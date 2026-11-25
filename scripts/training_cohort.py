"""Prepare the training rights-review cohort for a Garden dataset version (proposal only).

Uses the hardened inspection of scripts/pilot_cohort.py (evidence bound to the exact records,
revisions and paths; a failed read is never absence) on every tier A repository of the dataset's
review queue, i.e. every repository whose root license is MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause or ISC, identical at every used revision, and confirmed by the package registry.
GPL/AGPL, unknown, missing and conflicting licenses (tiers C and D) and tier B (an open question
already at the root) are outside this cohort. So is every repository that also has holdout or
temporal-quarantine records: the training loader (sentinel/train/data.py) refuses all records of
such a repository, so approving it would make training fail; it is listed under `excluded`.

It never approves anything: data/garden/approvals.yaml and data/registry.yaml are not touched,
and the proposed approvals it writes are templates that fail schema validation until a reviewer
fills them in.

    python scripts/training_cohort.py --config configs/garden/seed_v2.yaml            # offline
    python scripts/training_cohort.py --config configs/garden/seed_v2.yaml --inspect  # Git cache

Outputs (<committed_dir>/cohort/):
    training_cohort.json      machine-readable evidence manifest (one entry per tier A repository)
    proposed_approvals.yaml   approval TEMPLATES for the repositories recommended for review
    COHORT_REVIEW.md          the owner's summary: decision groups, obligations, volume
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.garden.pipeline import load_config  # noqa: E402
from sentinel.garden.review import TIER_A_LICENSES  # noqa: E402

_spec = importlib.util.spec_from_file_location("pilot_cohort", ROOT / "scripts" / "pilot_cohort.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

COHORT_VERSION = "1"
OBLIGATIONS = {
    "Apache-2.0": "keep the license and any NOTICE file with any redistributed copy of the code; state changes (Apache-2.0 §4)",
    "MIT": "keep the copyright and permission notice with any redistributed copy of the code",
    "ISC": "keep the copyright and permission notice with any redistributed copy of the code",
    "BSD-2-Clause": "keep the copyright notice, conditions and disclaimer with any redistributed copy of the code",
    "BSD-3-Clause": "keep the copyright notice, conditions and disclaimer with any redistributed copy; no endorsement using the authors' names",
    "Zlib": "do not misrepresent the origin; mark altered source versions; keep the notice in source distributions",
    "0BSD": "none (public-domain-equivalent permission)",
    "Unlicense": "none (public-domain dedication)",
}
# Tier B extension (--extension): repositories whose only open questions are formal - no package
# registry cross-check, a registry that lists further (bundled) licenses next to the root license,
# a license file whose text changed between revisions while the license stayed the same, or another
# permissive license. Each is inspected exactly like tier A and gets its own decision group.
EXTENSION_LICENSES = set(OBLIGATIONS)


def registry_finding(detected: str, registry: dict[str, list[str] | None]) -> str:
    """agrees | lists_root_and_others | names_other_license | unavailable | unrecognised (for one repository)."""
    from sentinel.garden.review import license_families, registry_agreement

    agreement = registry_agreement(detected, registry)
    if agreement != "disagrees":
        return agreement
    root = license_families(detected)
    known = [values for values in registry.values() if values]
    # deps.dev lists every license found in a module (bundled/vendored code included); the root
    # license appearing for every package means the extra entries describe other files
    if root and all(root & license_families(" ".join(values)) for values in known):
        return "lists_root_and_others"
    return "names_other_license"


def license_variants(repo_key: str, entry: dict[str, Any], path: str) -> dict[str, Any]:
    """{sha256: detected SPDX} for every version of the root license file at the used revisions, or a problem."""
    from sentinel.garden.licenses import detect_license_text

    repo = pilot.CACHE / repo_key.replace("/", "__")
    variants: dict[str, str] = {}
    for revision, files in sorted(entry["license_files_by_revision"].items()):
        match = [f for f in files if f["path"] == path]
        if not match:
            return {"problem": f"{path} is missing at {revision[:12]}"}
        sha = match[0]["sha256"]
        if sha in variants:
            continue
        try:
            content = pilot._git(repo, "show", f"{revision}:{path}")
        except RuntimeError as exc:
            return {"problem": f"cannot read {path} at {revision[:12]}: {exc}"}
        if hashlib.sha256(content).hexdigest() != sha:
            return {"problem": f"{path} at {revision[:12]} does not match the recorded hash"}
        variants[sha] = detect_license_text(content.decode("utf-8", "replace"))
    return {"variants": dict(sorted(variants.items()))}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question_group(question: str) -> str:
    """Coarse type of an open question, so the owner decides similar cases together."""
    if "declares license" in question:
        return "package_metadata_names_another_license"
    if "may add terms" in question:
        return "notice_or_additional_license_file"
    if "run with --inspect" in question:
        return "evidence_not_current"
    if "package registry names another license" in question:
        return "registry_names_other_license"
    if "changed in substance" in question:
        return "license_changed_between_revisions"
    if "is missing at" in question or "does not match the recorded hash" in question:
        return "license_file_not_constant"
    if "not a single permissive license" in question:
        return "root_license_not_single_permissive"
    return "inspection_incomplete"


def approval_binding(entry: dict[str, Any]) -> dict[str, Any]:
    """What an approval of this cohort entry binds: the license file and every reviewed version of it."""
    binding = {"decision": "approved_for_training", "license_spdx": entry["license_spdx_detected"],
               "license_file": entry["license_file"], "license_file_sha256": entry["license_file_sha256"]}
    if entry.get("additional_license_file_sha256s"):
        binding["additional_license_file_sha256s"] = entry["additional_license_file_sha256s"]
    return binding


def projected_volume(entries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`pilot.simulated_volume` for explicit approval bindings (incl. further reviewed license versions)."""
    from sentinel.data.registry import load_registry
    from sentinel.garden.audit import family_of
    from sentinel.garden.governance import gate

    sources = {}
    for source in load_registry()["sources"]:
        source = dict(source)
        if source["source_id"] in {"ghsa-osv-fix-commits", "osv", "github-advisory-db"}:
            source["production_approved"] = True  # hypothetical, in memory only
        sources[source["source_id"]] = source
    approvals = {e["repository_key"]: approval_binding(e) for e in entries}
    would = [r for r in rows if r["provenance"]["repository_key"] in approvals and gate(r, sources, approvals)[0] == "training_approved"]

    def count(key):
        return dict(sorted(collections.Counter(key(r) for r in would).items()))

    vulnerable = [r for r in would if r["kind"] == "vulnerable"]
    return {
        "records": len(would),
        "repositories": len({r["provenance"]["repository_key"] for r in would}),
        "by_split": count(lambda r: r["split"]),
        "by_language": count(lambda r: r["label"]["language"]),
        "by_decision": count(lambda r: r["label"]["decision"]),
        "by_split_and_decision": count(lambda r: f"{r['split']}/{r['label']['decision']}"),
        "cwe": dict(sorted(collections.Counter(r["label"]["cwe"] for r in vulnerable).items())),
        "cwe_family": dict(sorted(collections.Counter(family_of(r["label"]["cwe"]) for r in vulnerable).items())),
        "patch_examples_by_split": dict(sorted(collections.Counter(r["split"] for r in vulnerable if r["supervision"]["patch_eligibility"] == "eligible").items())),
        "record_ids_sha256": hashlib.sha256("\n".join(sorted(r["record_id"] for r in would)).encode()).hexdigest(),
    }


def extension_entry(key: str, entry: dict[str, Any], records: list[dict[str, Any]], inspect: bool, previous: Any) -> dict[str, Any]:
    """A tier B repository inspected like tier A, with its formal open question classified."""
    reasons = [r.split(":")[0] for r in entry["tier_reasons"]]
    detected = entry["license_detected"]
    path = entry["license_files_latest"][0]["path"]
    latest = entry["license_files_latest"][0]["sha256"]
    blockers: list[str] = []
    groups: list[str] = []
    if "license_lookup_failed" in reasons:
        blockers.append("the root license could not be read at every used revision")
    if detected not in EXTENSION_LICENSES:
        blockers.append(f"root license {detected} is not a single permissive license")
    registry = registry_finding(detected, entry["registry_licenses"])
    if registry == "names_other_license":
        blockers.append(f"the package registry names another license: {entry['registry_licenses']}")
    elif registry in {"unavailable", "unrecognised"}:
        groups.append("no_registry_crosscheck")
    elif registry == "lists_root_and_others":
        groups.append("registry_lists_bundled_licenses")
    if detected in EXTENSION_LICENSES and detected not in TIER_A_LICENSES:
        groups.append(f"other_permissive_{detected}")

    state = pilot.source_state(records)
    evidence = pilot.inspect_repository(key, records, detected) if inspect else previous
    additional: list[str] = []
    if "license_file_changed_between_revisions" in reasons:
        groups.append("license_text_updated")
        if inspect and evidence is not None:
            evidence["license_variants"] = license_variants(key, entry, path)
        variants = (evidence or {}).get("license_variants") or {"problem": "license file versions not inspected (run with --inspect)"}
        if "problem" in variants:
            blockers.append(variants["problem"])
        elif set(variants["variants"].values()) != {detected}:
            blockers.append(f"the license file changed in substance between revisions: {variants['variants']}")
        else:
            additional = sorted(sha for sha in variants["variants"] if sha != latest)
    decision, confirmed, questions = pilot.assess(entry, evidence, state)
    if pilot.stale_reason(evidence, state):
        evidence = None
    confirmed[0] = (f"root license file {path} detected as {detected} in every version used "
                    f"({1 + len(additional)} version(s))" if "license_file_changed_between_revisions" in reasons else confirmed[0])
    confirmed[1] = f"package registry (deps.dev): {registry} {entry['registry_licenses']}"
    questions = sorted(set(questions + blockers))
    if questions:
        decision = "needs_clarification"
    vulnerable = [r for r in records if r["kind"] == "vulnerable"]
    return {
        "repository_key": key,
        "url": entry["repository"],
        "tier": entry["tier"],
        "tier_reasons": entry["tier_reasons"],
        "registry_finding": registry,
        "languages": sorted({r["label"]["language"] for r in records}),
        "splits": sorted({r["split"] for r in records}),
        "license_spdx_detected": detected,
        "license_file": path,
        "license_file_sha256": latest,
        "additional_license_file_sha256s": additional,
        "license_file_url": f"{entry['repository']}/blob/{entry['latest_fixed_revision']}/{path}",
        "registry_licenses": entry["registry_licenses"],
        "revisions": state["revisions"],
        "records": dict(sorted(collections.Counter(r["label"]["decision"] for r in records).items())),
        "record_ids": [r["record_id"] for r in records],
        "cwes": sorted({r["label"]["cwe"] for r in vulnerable}),
        "advisories": entry["advisories"],
        "obligations": [OBLIGATIONS.get(detected, "see the license text")],
        "proposed_decision": decision,
        "decision_group": ("extension:" + "+".join(sorted(groups)) if decision == "recommended_for_review"
                           else "clarify_extension:" + "+".join(sorted({question_group(q) for q in questions}))),
        "confirmed_evidence": confirmed,
        "open_questions": questions,
        "evidence": evidence,
    }


def build(config_path: Path, inspect: bool, extension: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    pilot.CACHE = config.workdir / "cache" / "git"  # the selected dataset's own upstream cache
    data = config.committed_dir
    out = data / "cohort"
    rows = _jsonl(data / "manifest.jsonl")
    queue = {e["repository_key"]: e for e in _jsonl(data / "review" / "review_queue.jsonl")}
    by_repo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["quality"]["status"] == "accepted" and row["split"] in {"train", "validation"}:
            by_repo[row["provenance"]["repository_key"]].append(row)
    previous: dict[str, Any] = {}
    if (out / "training_cohort.json").is_file():
        document = json.loads((out / "training_cohort.json").read_text())
        for r in document["repositories"] + document.get("extension_repositories", []):
            previous[r["repository_key"]] = r.get("evidence")

    held_out = {r["provenance"]["repository_key"] for r in rows if r["split"] in {"holdout_reserved", "temporal_quarantine"}}
    repositories = []
    excluded = []
    for key, entry in sorted(queue.items()):
        if entry["tier"] != "A" or key not in by_repo:
            continue
        if key in held_out:
            excluded.append({"repository_key": key, "records": len(by_repo[key]),
                             "reason": "repository also has holdout/temporal-quarantine records; the training loader refuses it"})
            continue
        records = sorted(by_repo[key], key=lambda r: r["record_id"])
        state = pilot.source_state(records)
        evidence = pilot.inspect_repository(key, records, entry["license_detected"]) if inspect else previous.get(key)
        decision, confirmed, questions = pilot.assess(entry, evidence, state)
        if pilot.stale_reason(evidence, state):
            evidence = None  # never carried forward for another source state
        vulnerable = [r for r in records if r["kind"] == "vulnerable"]
        repositories.append(
            {
                "repository_key": key,
                "url": entry["repository"],
                "languages": sorted({r["label"]["language"] for r in records}),
                "splits": sorted({r["split"] for r in records}),
                "license_spdx_detected": entry["license_detected"],
                "license_file": entry["license_files_latest"][0]["path"],
                "license_file_sha256": entry["license_files_latest"][0]["sha256"],
                "license_file_url": f"{entry['repository']}/blob/{entry['latest_fixed_revision']}/{entry['license_files_latest'][0]['path']}",
                "registry_licenses": entry["registry_licenses"],
                "revisions": state["revisions"],
                "records": dict(sorted(collections.Counter(r["label"]["decision"] for r in records).items())),
                "record_ids": [r["record_id"] for r in records],
                "cwes": sorted({r["label"]["cwe"] for r in vulnerable}),
                "advisories": entry["advisories"],
                "obligations": [OBLIGATIONS[entry["license_detected"]]],
                "proposed_decision": decision,
                "decision_group": (
                    f"recommended:{entry['license_detected']}" if decision == "recommended_for_review"
                    else "clarify:" + "+".join(sorted({question_group(q) for q in questions}))
                ),
                "confirmed_evidence": confirmed,
                "open_questions": questions,
                "evidence": evidence,
            }
        )
    recommended = {r["repository_key"] for r in repositories if r["proposed_decision"] == "recommended_for_review"}
    volume = pilot.simulated_volume(recommended, rows)
    volume["repositories"] = len(recommended)
    result: dict[str, Any] = {}
    if extension:
        extension_repositories, extension_excluded = [], []
        for key, entry in sorted(queue.items()):
            if entry["tier"] != "B" or key not in by_repo:
                continue
            if key in held_out:
                extension_excluded.append({"repository_key": key, "records": len(by_repo[key]),
                                           "reason": "repository also has holdout/temporal-quarantine records; the training loader refuses it"})
                continue
            records = sorted(by_repo[key], key=lambda r: r["record_id"])
            extension_repositories.append(extension_entry(key, entry, records, inspect, previous.get(key)))
        clean = [r for r in repositories + extension_repositories if r["proposed_decision"] == "recommended_for_review"]
        result = {
            "extension": "tier B repositories whose open questions are formal, inspected like tier A (--extension)",
            "extension_repositories": extension_repositories,
            "extension_excluded": extension_excluded,
            "extension_decision_groups": dict(sorted(collections.Counter(r["decision_group"] for r in extension_repositories).items())),
            "if_extension_is_also_approved": projected_volume(clean, rows),
        }
    return result | {
        "cohort_version": COHORT_VERSION,
        "status": "PROPOSAL - not an approval; see COHORT_REVIEW.md",
        "dataset_id": config.dataset_id,
        "config": str(config_path.relative_to(ROOT)) if config_path.is_absolute() else str(config_path),
        "manifest_sha256": hashlib.sha256((data / "manifest.jsonl").read_bytes()).hexdigest(),
        "review_queue_sha256": hashlib.sha256((data / "review" / "review_queue.jsonl").read_bytes()).hexdigest(),
        "selection": "every tier A repository of the review queue with accepted train/validation records",
        "decision_groups": dict(sorted(collections.Counter(r["decision_group"] for r in repositories).items())),
        "repositories": repositories,
        "excluded": excluded,
        "if_recommended_are_approved": volume,
    }


def _table(mapping: dict[str, Any], key: str, value: str = "Records") -> list[str]:
    return [f"| {key} | {value} |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in mapping.items()]


def summary_markdown(cohort: dict[str, Any]) -> str:
    repos = cohort["repositories"]
    volume = cohort["if_recommended_are_approved"]
    recommended = [r for r in repos if r["proposed_decision"] == "recommended_for_review"]
    clarify = [r for r in repos if r["proposed_decision"] != "recommended_for_review"]
    families = volume["cwe_family"]
    lines = [
        f"# Training cohort rights review - {cohort['dataset_id']} (PROPOSAL, nothing approved)",
        "",
        "Generated by `scripts/training_cohort.py --inspect`; machine-readable evidence in `training_cohort.json`,",
        "approval templates (not loadable) in `proposed_approvals.yaml`. This is an engineering evidence",
        "summary, not a legal opinion. The method and the open legal questions are in",
        "`docs/TRAINING_COHORT_RIGHTS_REVIEW.md`.",
        "",
        f"Manifest sha256 `{cohort['manifest_sha256']}`; review queue sha256 `{cohort['review_queue_sha256']}`.",
        "",
        "## Decision groups",
        "",
        "| Group | Repositories | Records (train/validation) | What the reviewer decides |",
        "| --- | ---: | ---: | --- |",
    ]
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in repos:
        groups[r["decision_group"]].append(r)
    for name, members in sorted(groups.items(), key=lambda kv: (not kv[0].startswith("recommended"), -len(kv[1]), kv[0])):
        records = sum(sum(r["records"].values()) for r in members)
        if name.startswith("recommended:"):
            spdx = name.split(":", 1)[1]
            what = f"confirm {spdx} for training use; obligation if code is redistributed: {OBLIGATIONS[spdx]}"
        else:
            what = "resolve the listed question per repository (table below) or leave it out"
        lines.append(f"| `{name}` | {len(members)} | {records} | {what} |")
    lines += [
        "",
        f"Recommended for review: **{len(recommended)} repositories**; needing clarification: **{len(clarify)}**. "
        f"Not in the cohort: {len(cohort['excluded'])} tier A repositories ({sum(e['records'] for e in cohort['excluded'])} records) that also have "
        "holdout or temporal-quarantine records, because the training loader refuses every record of such a repository.",
        "",
        "## Volume if the three sources and every recommended repository were approved",
        "",
        "Computed with the real governance gate in memory (`pilot_cohort.simulated_volume`); nothing is",
        "written and nothing is approved. Holdout and temporal-quarantine records stay blocked by the gate.",
        "",
        "| | train | validation | total |",
        "| --- | ---: | ---: | ---: |",
    ]
    sd = volume["by_split_and_decision"]
    for decision in ("finding", "no_finding", "abstain"):
        t, v = sd.get(f"train/{decision}", 0), sd.get(f"validation/{decision}", 0)
        lines.append(f"| {decision} | {t} | {v} | {t + v} |")
    lines.append(f"| **detection examples** | **{volume['by_split'].get('train', 0)}** | **{volume['by_split'].get('validation', 0)}** | **{volume['records']}** |")
    p = volume["patch_examples_by_split"]
    lines.append(f"| patch examples | {p.get('train', 0)} | {p.get('validation', 0)} | {sum(p.values())} |")
    lines += ["", f"Record-ID set sha256: `{volume['record_ids_sha256']}`.", "", "### By language", ""]
    lines += _table(volume["by_language"], "Language")
    lines += ["", f"### CWE families of the findings ({len(volume['cwe'])} CWEs)", ""]
    lines += _table(dict(sorted(families.items(), key=lambda kv: -kv[1])), "Family", "Findings")
    lines += ["", "## Repositories needing clarification", "", "| Repository | License | Records | Open questions |", "| --- | --- | ---: | --- |"]
    for r in clarify:
        questions = "; ".join(r["open_questions"][:3]) + (f"; ... ({len(r['open_questions'])} total)" if len(r["open_questions"]) > 3 else "")
        lines.append(f"| [{r['repository_key']}]({r['url']}) | {r['license_spdx_detected']} | {sum(r['records'].values())} | {questions.replace('|', '/')} |")
    lines += ["", "## Recommended repositories", "", "| Repository | License (file @ revision) | Languages | Records F/N/A | CWEs |", "| --- | --- | --- | --- | --- |"]
    for r in sorted(recommended, key=lambda r: (r["license_spdx_detected"], r["repository_key"])):
        rec = r["records"]
        lines.append(
            f"| [{r['repository_key']}]({r['url']}) | [{r['license_spdx_detected']}]({r['license_file_url']}) (`{r['license_file_sha256'][:12]}`) "
            f"| {', '.join(r['languages'])} | {rec.get('finding', 0)}/{rec.get('no_finding', 0)}/{rec.get('abstain', 0)} | {', '.join(r['cwes'])} |"
        )
    if "extension_repositories" in cohort:
        lines += extension_markdown(cohort)
    return "\n".join(lines) + "\n"


EXTENSION_GROUP_TEXT = {
    "no_registry_crosscheck": "deps.dev has no usable license for the package; the decision rests on the root license file alone",
    "registry_lists_bundled_licenses": "deps.dev lists the root license plus further licenses (typically of bundled/vendored code in the module); the used files carry no other license",
    "license_text_updated": "the license file text changed between the used revisions (e.g. a copyright year) but every version is the same license; the approval lists every version",
}


def extension_markdown(cohort: dict[str, Any]) -> list[str]:
    repos = cohort["extension_repositories"]
    clean = [r for r in repos if r["proposed_decision"] == "recommended_for_review"]
    volume = cohort["if_extension_is_also_approved"]
    lines = ["", "## Tier B extension (optional second decision)", "",
             "Tier B repositories whose open questions are formal, inspected exactly like tier A. Templates:",
             "`proposed_approvals_extension.yaml`.", "",
             "| Group | Repositories | Records (train/validation) | What it means |", "| --- | ---: | ---: | --- |"]
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in repos:
        groups[r["decision_group"]].append(r)
    for name, members in sorted(groups.items(), key=lambda kv: (not kv[0].startswith("extension:"), -len(kv[1]), kv[0])):
        parts = name.split(":", 1)[1].split("+")
        text = "; ".join(EXTENSION_GROUP_TEXT.get(p, p.replace("_", " ")) for p in parts) if name.startswith("extension:") else "not recommended: see the open questions"
        lines.append(f"| `{name}` | {len(members)} | {sum(sum(r['records'].values()) for r in members)} | {text} |")
    sd = volume["by_split"]
    lines += ["", f"Recommended in the extension: **{len(clean)} repositories**. If they were approved as well as the tier A "
              f"recommendation: **{volume['records']} records** (train {sd.get('train', 0)}, validation {sd.get('validation', 0)}), "
              f"{sum(volume['patch_examples_by_split'].values())} patch examples, {len(volume['cwe'])} CWEs, "
              f"from {volume['repositories']} repositories (record-ID set sha256 `{volume['record_ids_sha256'][:12]}`).", "",
              "| Repository | License | Group | Records F/N/A | Open questions |", "| --- | --- | --- | --- | --- |"]
    for r in sorted(repos, key=lambda r: (r["proposed_decision"] != "recommended_for_review", r["decision_group"], r["repository_key"])):
        rec = r["records"]
        q = "; ".join(r["open_questions"][:2]).replace("|", "/")
        lines.append(f"| [{r['repository_key']}]({r['url']}) | [{r['license_spdx_detected']}]({r['license_file_url']}) | `{r['decision_group']}` "
                     f"| {rec.get('finding', 0)}/{rec.get('no_finding', 0)}/{rec.get('abstain', 0)} | {q} |")
    return lines


def extension_templates(cohort: dict[str, Any]) -> str:
    import yaml

    entries = []
    for repo in cohort["extension_repositories"]:
        if repo["proposed_decision"] != "recommended_for_review":
            continue
        entry = {
            "repository_key": repo["repository_key"],
            "decision": "<REQUIRED: approved_for_training | blocked>",
            "license_spdx": f"<REQUIRED: reviewer-confirmed SPDX id; detected {repo['license_spdx_detected']}>",
            "license_file": repo["license_file"],
            "license_file_sha256": repo["license_file_sha256"],
        }
        if repo["additional_license_file_sha256s"]:
            entry["additional_license_file_sha256s"] = repo["additional_license_file_sha256s"]
        entry.update({
            "terms_status": "<REQUIRED: approved_for_project | blocked>",
            "redistribution_status": "<REQUIRED: not_redistributed | approved | blocked>",
            "reviewer": "<REQUIRED: name of the authorized reviewer>",
            "reviewed_at": "<REQUIRED: YYYY-MM-DD>",
            "evidence_urls": [repo["url"], repo["license_file_url"]],
            "notes": f"tier B extension, group {repo['decision_group']}",
        })
        entries.append(entry)
    return (
        f"# PROPOSED tier B extension approvals ({cohort['dataset_id']}) - TEMPLATES, NOT APPROVALS. Generated by scripts/training_cohort.py --extension.\n"
        "# Every <REQUIRED> field must be filled by the authorized reviewer after reading the linked license file (and, where listed,\n"
        "# every further version in additional_license_file_sha256s); only then may an entry be copied into data/garden/approvals.yaml.\n"
        + yaml.safe_dump({"approvals_version": "1.0", "repositories": entries}, sort_keys=False, width=200)
    )


def owner_decision(cohort: dict[str, Any]) -> str:
    """The open decisions only: recommended repositories (tier A, and the tier B extension) that have no
    entry in data/garden/approvals.yaml yet, with the evidence a decision needs. Nothing is approved."""
    from sentinel.garden.governance import load_approvals

    approved = load_approvals()
    rows = [
        (group, repo)
        for group, repos in (("A", cohort["repositories"]), ("B", cohort.get("extension_repositories", [])))
        for repo in repos
        if str(repo.get("decision_group", "")).startswith("recommended") and repo["repository_key"] not in approved
    ]
    records = sum(len(repo["record_ids"]) for _, repo in rows)
    lines = [
        f"# Open approval decisions - {cohort['dataset_id']} (PROPOSAL, nothing approved)",
        "",
        f"Generated by `scripts/training_cohort.py --inspect --extension`. {len(rows)} recommended repositories without an entry in",
        f"`data/garden/approvals.yaml`, {records} records. Every repository was inspected at every used revision: the root license",
        "file (bound by sha256), NOTICE and additional license files, nested licenses and the package registry's license.",
        "To approve, copy the repository's template from `proposed_approvals.yaml` (tier A) or `proposed_approvals_extension.yaml`",
        "(tier B) into `data/garden/approvals.yaml`, fill every `<REQUIRED>` field yourself and run `curate`.",
        "",
        "| Tier | Repository | License | License file (sha256) | Records | Languages | Open questions |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for group, repo in sorted(rows, key=lambda r: (r[0], -len(r[1]["record_ids"]), r[1]["repository_key"])):
        questions = "; ".join(repo.get("open_questions") or []) or "-"
        lines.append(
            f"| {group} | [{repo['repository_key']}]({repo['url']}) | {repo['license_spdx_detected']} | "
            f"[{repo['license_file']}]({repo['license_file_url']}) `{repo['license_file_sha256'][:16]}…` | {len(repo['record_ids'])} | "
            f"{', '.join(sorted(repo.get('languages') or []))} | {questions} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/garden/seed_v3.yaml"))
    parser.add_argument("--inspect", action="store_true", help="read NOTICE/nested license/package metadata from the Git cache")
    parser.add_argument("--extension", action="store_true", help="also inspect tier B repositories whose open questions are formal")
    args = parser.parse_args(argv)
    cohort = build(args.config, args.inspect, args.extension)
    out = load_config(args.config).committed_dir / "cohort"
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_cohort.json").write_text(json.dumps(cohort, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    templates = pilot.proposed_approvals(cohort).replace("PROPOSED pilot approvals", f"PROPOSED training-cohort approvals ({cohort['dataset_id']})")
    templates = templates.replace("Generated by scripts/pilot_cohort.py", "Generated by scripts/training_cohort.py")
    (out / "proposed_approvals.yaml").write_text(templates, encoding="utf-8")
    (out / "COHORT_REVIEW.md").write_text(summary_markdown(cohort), encoding="utf-8")
    if "extension_repositories" in cohort:
        (out / "proposed_approvals_extension.yaml").write_text(extension_templates(cohort), encoding="utf-8")
    decision = owner_decision(cohort)
    (out / "OWNER_DECISION.md").write_text(decision, encoding="utf-8")
    print(f"training cohort: {len(cohort['repositories'])} tier A repositories {cohort['decision_groups']}, "
          f"{len(cohort['excluded'])} excluded (quarantine/holdout overlap); "
          f"if the recommended ones are approved: {cohort['if_recommended_are_approved']['by_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
