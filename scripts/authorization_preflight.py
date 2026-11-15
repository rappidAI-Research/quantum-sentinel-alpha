"""Simulate the post-authorization data path for a proposed cohort, without approving anything.

Answers "if the owner approved the three sources and exactly these repositories, would curate
export them and would the training loader accept them?" by running the real code end to end on a
scratch copy:

1. hypothetical approvals (in memory) for the cohort's recommended repositories and the sources;
2. `curate_records` + `write_exports` into a Git-ignored scratch directory;
3. `load_training_data` (the training preflight, including the supervision-hash and checksum
   checks) on that scratch export, for the detection and patch tasks;
4. optionally, token lengths with the pinned foundation's tokenizer (small files only, no weights).

data/garden/approvals.yaml and data/registry.yaml are never read for writing, and nothing under
data/ is modified. The scratch directory lives under artifacts/ (Git-ignored).

    python scripts/authorization_preflight.py --config configs/garden/seed_v2.yaml \
        --cohort data/garden/seed_v2/cohort/training_cohort.json [--tokenizer DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.data.registry import load_registry  # noqa: E402
from sentinel.garden import pipeline, supervision  # noqa: E402
from sentinel.train import data as train_data  # noqa: E402

SOURCES = {"ghsa-osv-fix-commits", "osv", "github-advisory-db"}
SCRATCH = ROOT / "artifacts" / "garden" / "authorization_preflight"


def hypothetical_registry() -> dict[str, Any]:
    """The registry as it would look after the owner's source decision (in memory only).

    Every field `validate_registry` requires for an approved source is set the way an authorized
    reviewer would set it; the values are placeholders and marked as a simulation."""
    registry = load_registry()
    for source in registry["sources"]:
        if source["source_id"] in SOURCES:
            source.update(
                production_approved=True,
                revision="SIMULATION-osv-snapshot-2026-11-01",
                license_status="reviewed",
                terms_status="approved_for_project",
                redistribution_status="not_redistributed",
                reviewer="SIMULATION (no reviewer has approved anything)",
            )
    return registry


def hypothetical_approvals(cohort: dict[str, Any]) -> dict[str, Any]:
    return {
        r["repository_key"]: {
            "repository_key": r["repository_key"],
            "decision": "approved_for_training",
            "license_spdx": r["license_spdx_detected"],
            "license_file": r["license_file"],
            "license_file_sha256": r["license_file_sha256"],
        }
        for r in cohort["repositories"]
        if r["proposed_decision"] == "recommended_for_review"
    }


def run(config_path: Path, cohort_path: Path, tokenizer: Path | None) -> dict[str, Any]:
    config = pipeline.load_config(config_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    approvals = hypothetical_approvals(cohort)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "committed").mkdir(parents=True)
    raw = config.raw | {"workdir": str((SCRATCH / "work").relative_to(ROOT)), "committed_dir": str((SCRATCH / "committed").relative_to(ROOT))}
    (SCRATCH / "garden.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    shutil.copytree(config.committed_dir / "audit", SCRATCH / "committed" / "audit")  # same audit exclusions
    scratch_config = pipeline.load_config(SCRATCH / "garden.yaml")

    original_pipeline, original_train = pipeline.load_registry, train_data.load_registry
    pipeline.load_registry = train_data.load_registry = hypothetical_registry
    try:
        records = pipeline.curate_records(scratch_config, pipeline._read_jsonl(config.workdir / "records.raw.jsonl"), approvals=approvals)
        pairs = supervision.pairs_by_id(records)
        pipeline._write_jsonl(SCRATCH / "committed" / "manifest.jsonl", [pipeline.manifest_row(r, pairs) for r in records])
        pipeline.write_exports(scratch_config.workdir, records)
        loaded = train_data.load_training_data(SCRATCH / "garden.yaml", approvals=approvals, tasks=("detection", "patch"))
    finally:
        pipeline.load_registry, train_data.load_registry = original_pipeline, original_train

    committed = {r["record_id"]: r for r in pipeline._read_jsonl(config.committed_dir / "manifest.jsonl")}
    simulated = {r["record_id"]: r for r in records}
    drift = [rid for rid in committed if committed[rid]["quality"]["status"] != simulated[rid]["quality"]["status"]
             or committed[rid]["split"] != simulated[rid]["split"]]
    approved_ids = sorted(rid for rid, r in simulated.items() if r["governance"]["approval_status"] == "training_approved")
    result: dict[str, Any] = {
        "approved_repositories": len(approvals),
        "training_approved_records": len(approved_ids),
        "matches_cohort_projection": len(approved_ids) == cohort["if_recommended_are_approved"]["records"],
        "quality_or_split_drift_vs_committed_manifest": len(drift),
        "loader": {"train_examples": len(loaded.train), "validation_examples": len(loaded.validation),
                   "examples_by_task": loaded.examples_by_task, "repositories": loaded.repositories, "data_sha256": loaded.sha256},
    }
    if tokenizer is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tokenizer))
        lengths = {}
        for split, examples in (("train", loaded.train), ("validation", loaded.validation)):
            for example in examples:
                text = tok.apply_chat_template(example["messages"], tokenize=False) if hasattr(tok, "apply_chat_template") else ""
                lengths.setdefault(f"{split}/{example['task']}", []).append(len(tok(text)["input_ids"]))
        result["tokens"] = {k: {"n": len(v), "median": int(statistics.median(v)), "max": max(v), "over_4096": sum(1 for x in v if x > 4096)}
                            for k, v in sorted(lengths.items())}
    (SCRATCH / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "garden" / "seed_v2.yaml")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "garden" / "seed_v2" / "cohort" / "training_cohort.json")
    parser.add_argument("--tokenizer", type=Path, help="directory with the pinned foundation's tokenizer files (no weights)")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.cohort, args.tokenizer), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
