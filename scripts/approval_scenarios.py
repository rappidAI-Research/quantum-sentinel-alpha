"""Training yield under hypothetical owner decisions (in memory only; nothing is approved).

Runs the real governance (`pipeline.apply_governance`: registry gate, repository approvals, holdout and
quarantine repositories, source coverage, negative-kind validation, evaluation roles, external
benchmarks) on the curated records of a dataset version with cumulative decisions added in memory, and
counts what the training exports would contain (detection examples, patch examples of verified pairs).

    python scripts/approval_scenarios.py --config configs/garden/seed_v6.yaml

Writes <committed_dir>/cohort/approval_scenarios.json. data/registry.yaml and data/garden/approvals.yaml
are never touched; a scenario is a projection, not an approval.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.data.registry import load_registry
from sentinel.garden import supervision
from sentinel.garden.analysis import source_of
from sentinel.garden.governance import load_approvals
from sentinel.garden.pipeline import (
    _read_jsonl,
    apply_governance,
    load_config,
)

REAL_WORLD_SOURCES = ("morefixes", "nvd-cve", "morefixes-cve-fix-commits")
TOOL_TEST_SOURCES = ("codeql-query-tests",)


def _binding(entry: dict[str, Any]) -> dict[str, Any]:
    binding = {
        "repository_key": entry["repository_key"],
        "decision": "approved_for_training",
        "license_spdx": entry["license_spdx_detected"],
        "license_file": entry["license_file"],
        "license_file_sha256": entry["license_file_sha256"],
    }
    if entry.get("additional_license_file_sha256s"):
        binding["additional_license_file_sha256s"] = entry["additional_license_file_sha256s"]
    return binding


def recommended_repositories(cohort_path: Path) -> dict[str, dict[str, Any]]:
    """Approval bindings of every repository the cohort recommends (tier A and the tier B extension)."""
    if not cohort_path.is_file():
        return {}
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    out = {}
    for repo in cohort["repositories"] + cohort.get("extension_repositories", []):
        if str(repo.get("decision_group", "")).startswith(("recommended", "extension:")):
            out[repo["repository_key"]] = _binding(repo)
    return out


def tool_test_approval(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """github/codeql bound to the LICENSE file its tool-test records recorded (MIT)."""
    for record in records:
        if record["provenance"].get("source_kind") == "tool_test":
            files = record["provenance"]["license"]["files"]
            if files:
                return {
                    "github/codeql": {
                        "repository_key": "github/codeql",
                        "decision": "approved_for_training",
                        "license_spdx": "MIT",
                        "license_file": files[0]["path"],
                        "license_file_sha256": files[0]["sha256"],
                    }
                }
    return {}


def copyleft_approvals(records: list[dict[str, Any]], approved: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Hypothetical approvals of every repository whose single root license is weak or strong copyleft, bound to
    that license file, with a placeholder written justification (the gate requires one for non-permissive code).
    Repositories without any license or with conflicting license files are never included: no permission exists."""
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["provenance"]["repository_key"]
        lic = record["provenance"]["license"]
        if key in approved or key in out or lic["category"] not in {"weak_copyleft", "strong_copyleft"} or len(lic["files"]) != 1:
            continue
        out[key] = {
            "repository_key": key,
            "decision": "approved_for_training",
            "license_spdx": lic["detected"],
            "license_file": lic["files"][0]["path"],
            "license_file_sha256": lic["files"][0]["sha256"],
            "non_permissive_justification": "HYPOTHETICAL (scenario only): the owner's written basis for training on copyleft code",
        }
    return out


def yield_of(config: Any, records: list[dict[str, Any]], sources: dict[str, Any], approvals: dict[str, Any], validated: set[str] | None) -> dict[str, Any]:
    governed = copy.deepcopy(records)
    apply_governance(config, governed, sources, approvals, validated)
    approved = [r for r in governed if r["governance"]["approval_status"] == "training_approved"]
    train = [r for r in approved if r["split"] == "train"]
    validation = [r for r in approved if r["split"] == "validation"]
    patch = supervision.patch_examples(train)

    return {
        "training_approved_records": len(approved),
        "repositories": len({r["provenance"]["repository_key"] for r in approved}),
        "train_detection_examples": len(train),
        "train_patch_examples": len(patch),
        "train_examples_total": len(train) + len(patch),
        "validation_detection_examples": len(validation),
        "train_by_decision": dict(sorted(collections.Counter(r["target"]["decision"] for r in train).items())),
        "train_by_kind": dict(sorted(collections.Counter(r["kind"] for r in train).items())),
        "train_by_source": dict(sorted(collections.Counter(source_of(r) for r in train).items())),
        "train_by_language": dict(sorted(collections.Counter(r["input"]["language"] for r in train).items())),
        "unique_vulnerabilities_in_train": len({r["provenance"]["advisory_id"] for r in train if r["kind"] == "vulnerable"}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/garden/seed_v6.yaml"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    records = _read_jsonl(config.workdir / "records.jsonl")
    registry = {s["source_id"]: s for s in load_registry()["sources"]}
    approvals = load_approvals()
    recommended = recommended_repositories(config.committed_dir / "cohort" / "training_cohort.json")

    def with_sources(ids: tuple[str, ...], base: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(base)
        for source_id in ids:
            out[source_id] = {**out[source_id], "production_approved": True}  # hypothetical, in memory only
        return out

    s1 = with_sources(REAL_WORLD_SOURCES, registry)
    a2 = {**recommended, **approvals}
    s3 = with_sources(TOOL_TEST_SOURCES, s1)
    a3 = {**tool_test_approval(records), **a2}
    scenarios = [
        ("0_current", "today's registry and approvals", registry, approvals, None),
        ("1_real_world_sources", "+ MoreFixes, NVD and the MoreFixes fix-commit source approved", s1, approvals, None),
        ("2_recommended_repositories", f"+ the {len(recommended)} repositories the v6 cohort recommends (tier A and B extension)", s1, a2, None),
        ("3_tool_tests", "+ the CodeQL query-test source and github/codeql (MIT)", s3, a3, None),
        ("4_negative_kinds_validated", "+ defended_sibling and query_test_negative validated by the independent review", s3, a3, {"defended_sibling", "query_test_negative"}),
    ]
    a5 = {**copyleft_approvals(records, a3), **a3}
    scenarios.append(
        (
            "5_copyleft_with_justification",
            f"+ the {len(a5) - len(a3)} copyleft repositories (single GPL/LGPL/AGPL/MPL/EPL root license), each with a written justification; never repositories without a license",
            s3,
            a5,
            {"defended_sibling", "query_test_negative"},
        )
    )
    result = {"dataset_id": config.dataset_id, "scenarios": []}
    for name, description, sources, approved, validated in scenarios:
        projection = yield_of(config, records, sources, approved, validated)
        result["scenarios"].append({"scenario": name, "decisions": description, **projection})
        print(f"{name}: {projection['train_examples_total']} train examples ({projection['train_detection_examples']} detection + {projection['train_patch_examples']} patch)")
    out = config.committed_dir / "cohort" / "approval_scenarios.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
