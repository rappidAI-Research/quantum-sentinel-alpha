"""Rights-review preparation for upstream repositories (evidence collection and review queue).

This module never approves anything. It gathers the facts a human reviewer needs and orders the
work so the most valuable, lowest-risk repositories come first:

- the root license files (path + sha256) at EVERY vulnerable and fixed revision the dataset uses,
  so a license change between revisions is visible;
- an independent license signal per package from deps.dev (PyPI, npm, Go);
- record counts per split, advisory links and direct links to the license file at the pinned
  revision.

Tiers are recommendations for the order and depth of review, not decisions:

  A  permissive license (MIT, Apache-2.0, BSD-2/3-Clause, ISC), identical license file at every
     used revision, deps.dev agrees
  B  permissive, but something needs a closer look (deps.dev disagrees or is silent, the license
     file changed between revisions, several permissive licenses, other permissive licenses)
  C  weak copyleft (MPL, LGPL, EPL) - needs an explicit policy decision
  D  strong copyleft (GPL, AGPL), unknown, missing or conflicting license - blocked by default;
     the governance gate refuses training without a written justification in the approval
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from sentinel.garden.gitsource import GitError, UpstreamRepo

DEPSDEV = "https://api.deps.dev/v3/systems"
DEPSDEV_SYSTEM = {"PyPI": "pypi", "npm": "npm", "Go": "go"}
TIER_A_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
BATCH_SIZE = 25

_FAMILY = [
    ("AGPL", re.compile(r"\bAGPL|affero", re.I)),
    ("LGPL", re.compile(r"\bLGPL|lesser general", re.I)),
    ("GPL", re.compile(r"\bGPL|general public license", re.I)),
    ("MPL", re.compile(r"\bMPL|mozilla", re.I)),
    ("Apache", re.compile(r"apache", re.I)),
    ("MIT", re.compile(r"\bMIT\b|\bexpat\b", re.I)),
    ("BSD", re.compile(r"\bBSD", re.I)),
    ("ISC", re.compile(r"\bISC\b", re.I)),
    ("Unlicense", re.compile(r"unlicense", re.I)),
]


def license_families(value: str) -> set[str]:
    """Coarse license families named in an SPDX expression or free-text license string."""
    families = set()
    for family, pattern in _FAMILY:
        if pattern.search(value):
            families.add(family)
    if "LGPL" in families or "AGPL" in families:
        families.discard("GPL")
    return families


def _http_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "quantum-sentinel-alpha-sentinel-garden/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


class DepsDev:
    """deps.dev package metadata with an on-disk cache (default version's licenses)."""

    def __init__(self, cache_dir: Path, fetch: Callable[[str], Any] = _http_json) -> None:
        self.cache_dir = cache_dir
        self.fetch = fetch

    def licenses(self, ecosystem: str, package: str) -> list[str] | None:
        cache = self.cache_dir / ecosystem / (urllib.parse.quote(package, safe="") + ".json")
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))["licenses"]
        system = DEPSDEV_SYSTEM[ecosystem]
        name = urllib.parse.quote(package if ecosystem != "PyPI" else package.lower(), safe="")
        try:
            info = self.fetch(f"{DEPSDEV}/{system}/packages/{name}")
            default = next((v for v in info.get("versions", []) if v.get("isDefault")), None)
            if default is None:
                result = None
            else:
                version = urllib.parse.quote(default["versionKey"]["version"], safe="")
                result = sorted(self.fetch(f"{DEPSDEV}/{system}/packages/{name}/versions/{version}").get("licenses") or [])
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            result = None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"licenses": result}), encoding="utf-8")
        return result


def tier_of(evidence: dict[str, Any]) -> tuple[str, list[str]]:
    """Recommended review tier and the reasons for it."""
    reasons: list[str] = []
    detected = evidence["license_detected"]
    category = evidence["license_category"]
    if category in {"strong_copyleft", "unknown", "none"}:
        return "D", ["license_conflicting" if detected.startswith("multiple:") else f"license_{category}"]
    if category == "weak_copyleft":
        return "C", ["weak_copyleft"]
    if not evidence["license_files_identical_across_revisions"]:
        reasons.append("license_file_changed_between_revisions")
    if detected not in TIER_A_LICENSES:
        reasons.append(f"other_permissive_license:{detected}")
    agreement = evidence["registry_agreement"]
    if agreement != "agrees":
        reasons.append(f"registry_{agreement}")
    return ("A" if not reasons else "B"), reasons


def registry_agreement(detected: str, registry: dict[str, list[str] | None]) -> str:
    known = [values for values in registry.values() if values]
    if not known:
        return "unavailable"
    detected_families = license_families(detected)
    named = set().union(*(license_families(" ".join(values)) for values in known))
    if not named:
        return "unrecognised"
    return "agrees" if named <= detected_families or detected_families & named == named else "disagrees"


def collect_evidence(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], git_cache: Path, depsdev: DepsDev, workers: int = 8) -> list[dict[str, Any]]:
    """Evidence for every repository with accepted train/validation records."""
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["quality"]["status"] == "accepted" and row["split"] in {"train", "validation"}:
            by_repo[row["provenance"]["repository_key"]].append(row)
    packages: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for candidate in candidates:
        for package in candidate["packages"]:
            packages[candidate["repository_key"]].add((candidate["ecosystem"], package))

    def one(repo_key: str) -> dict[str, Any]:
        repo_rows = by_repo[repo_key]
        provenance = max(repo_rows, key=lambda r: r["provenance"]["fix_committed_at"])["provenance"]
        repo = UpstreamRepo(git_cache, provenance["repository"], repo_key)
        revisions = sorted({r["provenance"][k] for r in repo_rows for k in ("vulnerable_revision", "fixed_revision")})
        files_by_revision: dict[str, list[dict[str, str]]] = {}
        for revision in revisions:
            try:
                files_by_revision[revision] = [{"path": f.path, "sha256": f.sha256} for f in repo.root_license_files(revision)]
            except GitError:
                files_by_revision[revision] = []
        distinct = {json.dumps(v, sort_keys=True) for v in files_by_revision.values()}
        registry = {f"{eco}:{name}": depsdev.licenses(eco, name) for eco, name in sorted(packages[repo_key])}
        detected = provenance["license"]["detected"]
        evidence = {
            "repository_key": repo_key,
            "repository": provenance["repository"],
            "license_detected": detected,
            "license_category": provenance["license"]["category"],
            "license_files_latest": provenance["license"]["files"],
            "latest_fixed_revision": provenance["fixed_revision"],
            "license_files_by_revision": files_by_revision,
            "license_files_identical_across_revisions": len(distinct) == 1,
            "registry_licenses": registry,
            "registry_agreement": registry_agreement(detected, registry),
            "records": dict(sorted(Counter(f"{r['split']}/{r['kind']}" for r in repo_rows).items())),
            "record_count": len(repo_rows),
            "advisories": sorted({r["provenance"]["advisory_id"] for r in repo_rows}),
        }
        tier, reasons = tier_of(evidence)
        evidence["tier"] = tier
        evidence["tier_reasons"] = reasons
        evidence["evidence_urls"] = _evidence_urls(evidence)
        return evidence

    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(one, sorted(by_repo)))
    return sorted(result, key=lambda e: (e["tier"], -e["record_count"], e["repository_key"]))


def _evidence_urls(evidence: dict[str, Any]) -> list[str]:
    urls = [evidence["repository"]]
    for license_file in evidence["license_files_latest"]:
        urls.append(f"{evidence['repository']}/blob/{evidence['latest_fixed_revision']}/{license_file['path']}")
    for key in evidence["registry_licenses"]:
        ecosystem, package = key.split(":", 1)
        urls.append(f"https://deps.dev/{DEPSDEV_SYSTEM[ecosystem]}/{urllib.parse.quote(package, safe='')}")
    urls += [f"https://github.com/advisories/{advisory}" for advisory in evidence["advisories"]]
    return urls


def approval_template(evidence: dict[str, Any]) -> str:
    """A YAML entry with all facts prefilled and the decision left open (not loadable as-is)."""
    files = evidence["license_files_latest"]
    license_file = files[0] if files else {"path": "<NONE FOUND>", "sha256": "<NONE>"}
    lines = [
        f"  # tier {evidence['tier']}: {', '.join(evidence['tier_reasons']) or 'no open questions found by tooling'}",
        f"  # records: {evidence['record_count']} {evidence['records']}",
        f"  # registry (deps.dev): {evidence['registry_licenses']}",
        f"  - repository_key: {evidence['repository_key']}",
        '    decision: "<REQUIRED: approved_for_training | blocked>"',
        f'    license_spdx: "<REQUIRED: reviewer-confirmed SPDX id; detected {evidence["license_detected"]}>"',
        f"    license_file: {license_file['path']}",
        f"    license_file_sha256: {license_file['sha256']}",
        '    terms_status: "<REQUIRED: approved_for_project | blocked>"',
        '    redistribution_status: "<REQUIRED: not_redistributed | approved | blocked>"',
        '    reviewer: "<REQUIRED: name of the authorized reviewer>"',
        '    reviewed_at: "<REQUIRED: YYYY-MM-DD>"',
        "    evidence_urls:",
        *[f"      - {url}" for url in evidence["evidence_urls"][:2]],
        '    notes: ""',
    ]
    if evidence["tier"] in {"C", "D"}:
        lines.insert(-1, '    non_permissive_justification: "<REQUIRED for non-permissive licenses: written policy/legal basis>"')
    return "\n".join(lines)


def write_review_outputs(evidence: list[dict[str, Any]], committed: Path, manifest_sha256: str) -> dict[str, Any]:
    review_dir = committed / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    for old in review_dir.glob("approvals_template_batch_*.yaml"):
        old.unlink()
    with (review_dir / "review_queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in evidence:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    reviewable = [e for e in evidence if e["tier"] in {"A", "B"}]
    batches = [reviewable[i : i + BATCH_SIZE] for i in range(0, len(reviewable), BATCH_SIZE)]
    for number, batch in enumerate(batches, start=1):
        body = "\n\n".join(approval_template(e) for e in batch)
        (review_dir / f"approvals_template_batch_{number:02d}.yaml").write_text(
            "# Review batch {n}: approval TEMPLATES, not approvals. Each field marked <REQUIRED> must be filled by an\n"
            "# authorized reviewer after reading the license file at the linked revision; then move the entry into\n"
            "# data/garden/approvals.yaml. These templates fail schema validation on purpose until completed.\n"
            "repositories:\n{body}\n".format(n=number, body=body),
            encoding="utf-8",
        )
    summary = {
        "manifest_sha256": manifest_sha256,
        "repositories": len(evidence),
        "by_tier": dict(sorted(Counter(e["tier"] for e in evidence).items())),
        "records_by_tier": {t: sum(e["record_count"] for e in evidence if e["tier"] == t) for t in sorted({e["tier"] for e in evidence})},
        "registry_agreement": dict(sorted(Counter(e["registry_agreement"] for e in evidence).items())),
        "license_changed_between_revisions": sum(1 for e in evidence if not e["license_files_identical_across_revisions"]),
        "batches": len(batches),
    }
    (review_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (committed / "REVIEW_QUEUE.md").write_text(review_queue_markdown(evidence, summary), encoding="utf-8")
    return summary


def review_queue_markdown(evidence: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    tier_names = {
        "A": "A - permissive, consistent evidence",
        "B": "B - permissive, open questions",
        "C": "C - weak copyleft (policy decision needed)",
        "D": "D - blocked by default (strong copyleft, unknown, missing or conflicting license)",
    }
    out = [
        "# Sentinel Garden repository review queue",
        "",
        "Generated by `python -m sentinel.garden review`. **Nothing here is an approval.** Detected licenses and",
        "deps.dev metadata are evidence for a human reviewer. Workflow: `docs/RIGHTS_REVIEW.md`.",
        "",
        f"Repositories with accepted train/validation records: {summary['repositories']} "
        f"({', '.join(f'tier {t}: {n}' for t, n in summary['by_tier'].items())}).",
        f"Records by tier: {', '.join(f'{t}: {n}' for t, n in summary['records_by_tier'].items())}.",
        f"deps.dev agreement: {', '.join(f'{k}: {v}' for k, v in summary['registry_agreement'].items())}. "
        f"License file changed between used revisions: {summary['license_changed_between_revisions']} repositories.",
        f"Approval templates for tiers A and B: `review/approvals_template_batch_01..{summary['batches']:02d}.yaml` "
        f"(batches of {BATCH_SIZE}, most records first).",
        "",
    ]
    for tier in "ABCD":
        rows = [e for e in evidence if e["tier"] == tier]
        if not rows:
            continue
        out += [f"## Tier {tier_names[tier]}", "", "| Repository | Records | License | deps.dev | Open questions | License file @ revision | sha256 |", "| --- | ---: | --- | --- | --- | --- | --- |"]
        for e in rows:
            registry = "; ".join(f"{k.split(':', 1)[1]}: {'/'.join(v) if v else 'n/a'}" for k, v in e["registry_licenses"].items()) or "n/a"
            if e["license_files_latest"]:
                f = e["license_files_latest"][0]
                link = f"[{f['path']}]({e['repository']}/blob/{e['latest_fixed_revision']}/{f['path']}) @ `{e['latest_fixed_revision'][:12]}`"
                sha = f"`{f['sha256'][:16]}…`"
            else:
                link, sha = "(none)", "-"
            out.append(
                f"| [{e['repository_key']}]({e['repository']}) | {e['record_count']} | {e['license_detected']} | {registry} | "
                f"{', '.join(e['tier_reasons']) or '-'} | {link} | {sha} |"
            )
        out.append("")
    out.append("Full hashes, all revisions and evidence URLs: `review/review_queue.jsonl`.")
    out.append("")
    return "\n".join(out)
