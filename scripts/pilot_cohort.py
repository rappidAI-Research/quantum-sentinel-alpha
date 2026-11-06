"""Select and document a small pilot cohort for the Sentinel Garden rights review.

This tool prepares evidence for a human decision. It never approves anything: it does not touch
data/garden/approvals.yaml or data/registry.yaml, and the proposed approvals it writes are templates
that fail schema validation until a reviewer fills them in.

    python scripts/pilot_cohort.py              # select + statistics from committed data (offline)
    python scripts/pilot_cohort.py --inspect    # also read NOTICE / nested license / package metadata
                                                # at every used revision from the Git cache (network
                                                # only for commit trees not yet in the cache)

Outputs (data/garden/seed_v1/pilot/):
    cohort.json               machine-readable evidence manifest (one entry per selected repository)
    proposed_approvals.yaml   approval TEMPLATES for the recommended repositories (not active)

Selection (deterministic; docs/PILOT_DATASET_RIGHTS_REVIEW.md explains it):
    tier A of the review queue (permissive license, identical license file at every used revision,
    deps.dev agrees), at least one complete vulnerable/fixed pair, no weak advisory-code correspondence,
    no repository that also has temporal_quarantine records, at most 20 records, a quality score of at
    least 3, then per language and split a round-robin over CWE families.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.data.registry import load_registry  # noqa: E402
from sentinel.garden.audit import family_of  # noqa: E402
from sentinel.garden.governance import gate  # noqa: E402
from sentinel.garden.licenses import detect_license_text  # noqa: E402

SEED = ROOT / "data" / "garden" / "seed_v1"
OUT = SEED / "pilot"
CACHE = ROOT / "artifacts" / "garden" / "seed_v1" / "cache" / "git"
COHORT_VERSION = "1"
PLAN = {
    ("python", "train"): 10,
    ("javascript", "train"): 10,
    ("go", "train"): 10,
    ("typescript", "train"): 8,
    ("python", "validation"): 3,
    ("javascript", "validation"): 3,
    ("go", "validation"): 3,
    ("typescript", "validation"): 3,
}
MAX_RECORDS = 20
MIN_SCORE = 3.0
NOTICE_NAME = re.compile(r"^(NOTICE|NOTICES|COPYING|COPYRIGHT|PATENTS|LICEN[CS]E|THIRD[_-]?PARTY[A-Z_-]*|ADDITIONAL[_-]TERMS)(\.(md|txt|rst))?$", re.I)
SPDX_ALIASES = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd license": "BSD",
    "isc": "ISC",
    "isc license": "ISC",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------
# Selection (committed data only)
# --------------------------------------------------------------------------------------------


def score_repositories() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = _jsonl(SEED / "manifest.jsonl")
    queue = {e["repository_key"]: e for e in _jsonl(SEED / "review" / "review_queue.jsonl")}
    automated = {g["group_id"]: g for g in _jsonl(SEED / "audit" / "automated.jsonl")}
    manual = yaml.safe_load((SEED / "audit" / "manual_review.yaml").read_text(encoding="utf-8"))["items"]
    ai_doubt = {i["group_id"] for i in manual if i.get("cwe") in {"questionable", "wrong"} or i.get("localization") in {"imprecise", "wrong"}}
    quarantine = {r["provenance"]["repository_key"] for r in rows if r["split"] == "temporal_quarantine" and r["quality"]["status"] == "accepted"}
    by_repo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["quality"]["status"] == "accepted" and row["split"] in {"train", "validation"}:
            by_repo[row["provenance"]["repository_key"]].append(row)

    scored = []
    for key, entry in sorted(queue.items()):
        if entry["tier"] != "A" or key not in by_repo:
            continue
        records = by_repo[key]
        groups: dict[str, dict[str, Any]] = collections.defaultdict(dict)
        for row in records:
            groups[row["group_id"]][row["kind"]] = row
        group_quality = []
        for group_id, group in sorted(groups.items()):
            if "vulnerable" not in group:
                continue
            audit = automated.get(group_id, {})
            strength = audit.get("correspondence", {}).get("strength")
            span = audit.get("span", {}).get("span_lines", 99)
            vulnerable = group["vulnerable"]
            points = {"strong": 2, "moderate": 1, "weak": -3}.get(strength, 0)
            points += 1 if audit.get("fix_pattern", {}).get("consistent") else 0
            points -= 1 if any("coarse_cwe" in f for f in vulnerable["quality"]["flags"]) else 0
            points += 1 if span <= 6 else (0 if span <= 12 else -1)
            points += 1 if vulnerable["supervision"]["patch_eligibility"] == "eligible" else 0
            points -= 2 if group_id in ai_doubt else 0
            group_quality.append(
                {
                    "group_id": group_id,
                    "score": points,
                    "correspondence": strength,
                    "fix_pattern_consistent": audit.get("fix_pattern", {}).get("consistent"),
                    "span_lines": span,
                    "cwe": vulnerable["label"]["cwe"],
                    "patch_eligible": vulnerable["supervision"]["patch_eligibility"] == "eligible",
                    "has_abstention": "caller_abstention" in group,
                    "has_hard_negative": "post_fix" in group,
                }
            )
        if not group_quality:
            continue
        splits = sorted({r["split"] for r in records})
        scored.append(
            {
                "repository_key": key,
                "language": sorted({r["label"]["language"] for r in records}),
                "split": splits[0] if len(splits) == 1 else "mixed",
                "records": len(records),
                "complete_pairs": sum(1 for g in groups.values() if {"vulnerable", "post_fix"} <= set(g)),
                "score": round(sum(g["score"] for g in group_quality) / len(group_quality), 2),
                "weak_groups": sum(1 for g in group_quality if g["correspondence"] == "weak"),
                "families": sorted({family_of(g["cwe"]) for g in group_quality}),
                "groups": group_quality,
                "quarantine_overlap": key in quarantine,
                "registry_agreement": entry["registry_agreement"],
            }
        )
    return scored, by_repo


def select(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        s for s in scored
        if s["weak_groups"] == 0 and not s["quarantine_overlap"] and s["complete_pairs"] >= 1 and s["score"] >= MIN_SCORE
        and s["registry_agreement"] == "agrees" and s["records"] <= MAX_RECORDS and s["split"] != "mixed" and len(s["language"]) == 1
    ]
    chosen_all = []
    for (language, split), count in PLAN.items():
        candidates = sorted(
            (s for s in eligible if s["language"] == [language] and s["split"] == split),
            key=lambda s: (-s["score"], -any(g["has_abstention"] for g in s["groups"]), hashlib.sha256(s["repository_key"].encode()).hexdigest()),
        )
        families: collections.Counter[str] = collections.Counter()
        chosen: list[dict[str, Any]] = []
        for round_ in range(3):  # round-robin: prefer CWE families not yet covered
            for candidate in candidates:
                if candidate in chosen or len(chosen) >= count:
                    continue
                if round_ == 2 or any(families[f] <= round_ for f in candidate["families"]):
                    chosen.append(candidate)
                    families.update(candidate["families"])
        chosen_all += chosen
    return chosen_all


# --------------------------------------------------------------------------------------------
# Evidence at the exact revisions (Git cache)
# --------------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> bytes:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=300, check=False)
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace")[:200])
    return done.stdout


def _blobs(repo: Path, revision: str, directory: str) -> list[str]:
    spec = f"{revision}:{directory}" if directory else revision
    out = _git(repo, "ls-tree", spec).decode("utf-8", "replace")
    return [line.split("\t", 1)[1] for line in out.splitlines() if "\t" in line and line.split()[1] == "blob"]


def _package_license(repo: Path, revision: str, file_path: str, language: str) -> list[dict[str, Any]]:
    names = {"javascript": ["package.json"], "typescript": ["package.json"], "python": ["pyproject.toml", "setup.cfg", "setup.py"], "go": []}[language]
    directory = posixpath.dirname(file_path)
    while True:
        found = []
        try:
            present = _blobs(repo, revision, directory)
        except RuntimeError:
            present = []
        for name in names:
            if name not in present:
                continue
            path = posixpath.join(directory, name) if directory else name
            text = _git(repo, "show", f"{revision}:{path}").decode("utf-8", "replace")
            value = None
            if name == "package.json":
                try:
                    value = json.loads(text).get("license")
                except ValueError:
                    value = "unparseable"
            else:
                match = re.search(r"(?im)^\s*license\s*[=:]\s*(?:\{\s*text\s*=\s*)?[\"']?([^\"'\n,}]+)", text) or re.search(r"license\s*=\s*[\"']([^\"']+)", text)
                classifier = re.search(r"License :: OSI Approved :: ([^\"'\n]+)", text)
                value = (match.group(1).strip() if match else None) or (classifier.group(1).strip() if classifier else None)
            found.append({"file": path, "license": value})
        if found or not directory:
            return found
        directory = posixpath.dirname(directory)


def inspect_repository(key: str, records: list[dict[str, Any]], root_license: str) -> dict[str, Any]:
    repo = CACHE / key.replace("/", "__")
    revisions = sorted({r["provenance"][k] for r in records for k in ("vulnerable_revision", "fixed_revision")})
    files = sorted({r["provenance"]["file_path"] for r in records})
    language = records[0]["label"]["language"]
    evidence: dict[str, Any] = {"revisions": {}, "problems": []}
    for revision in revisions:
        try:
            root = _blobs(repo, revision, "")
        except RuntimeError as exc:
            evidence["problems"].append(f"cannot read {revision}: {exc}")
            continue
        notices = {}
        for name in root:
            if NOTICE_NAME.match(name):
                content = _git(repo, "show", f"{revision}:{name}")
                notices[name] = {"sha256": hashlib.sha256(content).hexdigest(), "detected": detect_license_text(content.decode("utf-8", "replace"))}
        nested = {}
        for file_path in files:
            directory = posixpath.dirname(file_path)
            while directory:
                try:
                    for name in _blobs(repo, revision, directory):
                        if NOTICE_NAME.match(name):
                            path = posixpath.join(directory, name)
                            content = _git(repo, "show", f"{revision}:{path}")
                            nested[path] = {"sha256": hashlib.sha256(content).hexdigest(), "detected": detect_license_text(content.decode("utf-8", "replace"))}
                except RuntimeError:
                    pass
                directory = posixpath.dirname(directory)
        evidence["revisions"][revision] = {
            "root_notice_and_license_files": notices,
            "license_files_on_record_paths": nested,
            "package_metadata": [m for f in files for m in _package_license(repo, revision, f, language)],
        }
    return evidence


def assess(entry: dict[str, Any], evidence: dict[str, Any] | None) -> tuple[str, list[str], list[str]]:
    """recommended_for_review | needs_clarification | exclude, with confirmed facts and open questions."""
    confirmed = [
        f"root license file(s) {', '.join(f['path'] for f in entry['license_files_latest'])} detected as {entry['license_detected']}, "
        f"identical at all {len(entry['license_files_by_revision'])} used revisions",
        f"package registry (deps.dev) agrees: {entry['registry_licenses']}",
    ]
    questions: list[str] = []
    if evidence is None:
        return "needs_clarification", confirmed, ["NOTICE/nested-license inspection not run"]
    if evidence["problems"]:
        questions += evidence["problems"]
    root_spdx = entry["license_detected"]
    for revision, data in evidence["revisions"].items():
        for name, info in {**data["root_notice_and_license_files"], **data["license_files_on_record_paths"]}.items():
            if posixpath.basename(name).lower().split(".")[0] in {"license", "licence", "copying"} and info["detected"] == root_spdx:
                continue
            if re.search(r"\.(py|go|js|ts|tsx|hcl|json|ya?ml)$", name):
                continue  # source/config files whose names only resemble a license file
            questions.append(f"{name} at {revision[:12]} (detected {info['detected']}) may add terms; check which code it covers")
        for meta in data["package_metadata"]:
            value = meta["license"]
            if value is None:
                continue
            normalized = SPDX_ALIASES.get(str(value).strip().lower(), str(value).strip())
            if normalized not in {root_spdx, "BSD"}:
                questions.append(f"{meta['file']} at {revision[:12]} declares license {value!r}, root license is {root_spdx}")
    questions = sorted(set(questions))
    if questions:
        return "needs_clarification", confirmed, questions
    confirmed.append("no NOTICE file and no other license file at the root or on the paths of the used files, at every used revision")
    return "recommended_for_review", confirmed, []


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------


def simulated_volume(cohort_keys: set[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What `curate` WOULD export if sources and exactly these repositories were approved (nothing written)."""
    registry = load_registry()
    sources = {}
    for source in registry["sources"]:
        source = dict(source)
        if source["source_id"] in {"ghsa-osv-fix-commits", "osv", "github-advisory-db"}:
            source["production_approved"] = True  # hypothetical, in memory only
        sources[source["source_id"]] = source
    approvals = {}
    for row in rows:
        key = row["provenance"]["repository_key"]
        if key in cohort_keys and row["provenance"]["license"]["files"]:
            first = row["provenance"]["license"]["files"][0]
            approvals.setdefault(key, {"decision": "approved_for_training", "license_spdx": row["provenance"]["license"]["detected"], "license_file": first["path"], "license_file_sha256": first["sha256"]})
    would = [r for r in rows if r["provenance"]["repository_key"] in cohort_keys and gate(r, sources, approvals)[0] == "training_approved"]

    def count(key):
        return dict(sorted(collections.Counter(key(r) for r in would).items()))

    vulnerable = [r for r in would if r["kind"] == "vulnerable"]
    return {
        "records": len(would),
        "by_split": count(lambda r: r["split"]),
        "by_language": count(lambda r: r["label"]["language"]),
        "by_decision": count(lambda r: r["label"]["decision"]),
        "by_split_and_decision": count(lambda r: f"{r['split']}/{r['label']['decision']}"),
        "cwe": dict(sorted(collections.Counter(r["label"]["cwe"] for r in vulnerable).items())),
        "cwe_family": dict(sorted(collections.Counter(family_of(r["label"]["cwe"]) for r in vulnerable).items())),
        "patch_examples_by_split": dict(sorted(collections.Counter(r["split"] for r in vulnerable if r["supervision"]["patch_eligibility"] == "eligible").items())),
        "record_ids_sha256": hashlib.sha256("\n".join(sorted(r["record_id"] for r in would)).encode()).hexdigest(),
    }


def build(inspect: bool) -> dict[str, Any]:
    scored, by_repo = score_repositories()
    queue = {e["repository_key"]: e for e in _jsonl(SEED / "review" / "review_queue.jsonl")}
    rows = _jsonl(SEED / "manifest.jsonl")
    selected = select(scored)
    previous = {}
    if (OUT / "cohort.json").is_file():
        previous = {r["repository_key"]: r.get("evidence") for r in json.loads((OUT / "cohort.json").read_text())["repositories"]}
    repositories = []
    for s in selected:
        key = s["repository_key"]
        entry = queue[key]
        records = by_repo[key]
        evidence = inspect_repository(key, records, entry["license_detected"]) if inspect else previous.get(key)
        decision, confirmed, questions = assess(entry, evidence)
        revisions = sorted({r["provenance"][k] for r in records for k in ("vulnerable_revision", "fixed_revision")})
        repositories.append(
            {
                "repository_key": key,
                "url": entry["repository"],
                "language": s["language"][0],
                "split": s["split"],
                "license_spdx_detected": entry["license_detected"],
                "license_file": entry["license_files_latest"][0]["path"],
                "license_file_sha256": entry["license_files_latest"][0]["sha256"],
                "license_file_url": f"{entry['repository']}/blob/{entry['latest_fixed_revision']}/{entry['license_files_latest'][0]['path']}",
                "revisions": revisions,
                "records": {k: v for k, v in sorted(collections.Counter(r["kind"] for r in records).items())},
                "record_ids": sorted(r["record_id"] for r in records),
                "cwes": sorted({g["cwe"] for g in s["groups"]}),
                "cwe_families": s["families"],
                "quality": {"score": s["score"], "groups": s["groups"]},
                "advisories": entry["advisories"],
                "evidence_urls": entry["evidence_urls"],
                "obligations": (
                    ["keep the copyright and permission notice (MIT/ISC/BSD) with any redistributed copy of the code"]
                    if entry["license_detected"] != "Apache-2.0"
                    else ["keep the license and any NOTICE file with any redistributed copy of the code; state changes (Apache-2.0 §4)"]
                ),
                "proposed_decision": decision,
                "confirmed_evidence": confirmed,
                "open_questions": questions,
                "evidence": evidence,
            }
        )
    recommended = {r["repository_key"] for r in repositories if r["proposed_decision"] == "recommended_for_review"}
    return {
        "cohort_version": COHORT_VERSION,
        "status": "PROPOSAL - not an approval; see docs/PILOT_DATASET_RIGHTS_REVIEW.md",
        "manifest_sha256": hashlib.sha256((SEED / "manifest.jsonl").read_bytes()).hexdigest(),
        "review_queue_sha256": hashlib.sha256((SEED / "review" / "review_queue.jsonl").read_bytes()).hexdigest(),
        "selection": {"plan": {f"{k[0]}/{k[1]}": v for k, v in PLAN.items()}, "max_records": MAX_RECORDS, "min_score": MIN_SCORE,
                      "tier_a_scored": len(scored)},
        "repositories": repositories,
        "if_recommended_are_approved": simulated_volume(recommended, rows),
    }


def proposed_approvals(cohort: dict[str, Any]) -> str:
    entries = []
    for repo in cohort["repositories"]:
        if repo["proposed_decision"] != "recommended_for_review":
            continue
        entries.append(
            {
                "repository_key": repo["repository_key"],
                "decision": "<REQUIRED: approved_for_training | blocked>",
                "license_spdx": f"<REQUIRED: reviewer-confirmed SPDX id; detected {repo['license_spdx_detected']}>",
                "license_file": repo["license_file"],
                "license_file_sha256": repo["license_file_sha256"],
                "terms_status": "<REQUIRED: approved_for_project | blocked>",
                "redistribution_status": "<REQUIRED: not_redistributed | approved | blocked>",
                "reviewer": "<REQUIRED: name of the authorized reviewer>",
                "reviewed_at": "<REQUIRED: YYYY-MM-DD>",
                "evidence_urls": [repo["url"], repo["license_file_url"]],
                "notes": "",
            }
        )
    return (
        "# PROPOSED pilot approvals - TEMPLATES, NOT APPROVALS. Generated by scripts/pilot_cohort.py.\n"
        "# Every <REQUIRED> field must be filled by the authorized reviewer after reading the linked license file;\n"
        "# only then may an entry be copied into data/garden/approvals.yaml. This file is never loaded.\n"
        + yaml.safe_dump({"approvals_version": "1.0", "repositories": entries}, sort_keys=False, width=200)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", action="store_true", help="read NOTICE/nested license/package metadata from the Git cache")
    args = parser.parse_args(argv)
    cohort = build(args.inspect)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cohort.json").write_text(json.dumps(cohort, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "proposed_approvals.yaml").write_text(proposed_approvals(cohort), encoding="utf-8")
    decisions = collections.Counter(r["proposed_decision"] for r in cohort["repositories"])
    print(f"pilot cohort: {len(cohort['repositories'])} repositories {dict(decisions)}; "
          f"if the recommended ones are approved: {cohort['if_recommended_are_approved']['by_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
