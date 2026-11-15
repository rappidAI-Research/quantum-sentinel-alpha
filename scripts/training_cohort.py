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
}


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
    return "inspection_incomplete"


def build(config_path: Path, inspect: bool) -> dict[str, Any]:
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
        previous = {r["repository_key"]: r.get("evidence") for r in json.loads((out / "training_cohort.json").read_text())["repositories"]}

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
    return {
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
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/garden/seed_v2.yaml"))
    parser.add_argument("--inspect", action="store_true", help="read NOTICE/nested license/package metadata from the Git cache")
    args = parser.parse_args(argv)
    cohort = build(args.config, args.inspect)
    out = load_config(args.config).committed_dir / "cohort"
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_cohort.json").write_text(json.dumps(cohort, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    templates = pilot.proposed_approvals(cohort).replace("PROPOSED pilot approvals", f"PROPOSED training-cohort approvals ({cohort['dataset_id']})")
    templates = templates.replace("Generated by scripts/pilot_cohort.py", "Generated by scripts/training_cohort.py")
    (out / "proposed_approvals.yaml").write_text(templates, encoding="utf-8")
    (out / "COHORT_REVIEW.md").write_text(summary_markdown(cohort), encoding="utf-8")
    print(f"training cohort: {len(cohort['repositories'])} tier A repositories {cohort['decision_groups']}, "
          f"{len(cohort['excluded'])} excluded (quarantine/holdout overlap); "
          f"if the recommended ones are approved: {cohort['if_recommended_are_approved']['by_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
