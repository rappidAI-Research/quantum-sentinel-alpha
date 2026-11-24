"""Development evaluation set: the training-approved Garden v3 validation split, for checkpoint selection.

    python -m sentinel.eval.devset --config configs/garden/seed_v3.yaml --output-dir runs/devset

Evaluation roles (docs/EVALUATION.md):

- `train` split: training only.
- `validation` split, approved records (repository-disjoint from train and from `holdout_reserved`):
  the **development set**. Checkpoints, thresholds and any other choice are made on it. It was only
  used for the validation loss during training, never for gradient updates.
- `holdout_reserved`: the **final independent test**, repository-disjoint from both. Evaluated once,
  after the choice is fixed; never used to choose (`sentinel.eval.select_checkpoint` refuses it).
- QuickEval v1: its results have been inspected, so it is a regression check only.

The set is built from the exports `materialize` writes, after the training loader has verified them
(approvals, manifest supervision hashes, SHA256SUMS). Upstream code stays in the Git-ignored output
directory; the cases are marked `source_kind: development` and are never committable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sentinel.eval.io import PROJECT_ROOT, case_set_sha256, load_jsonl, validate_cases
from sentinel.garden import render, supervision
from sentinel.garden.pipeline import load_config
from sentinel.train.data import GovernanceError, load_training_data

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "garden" / "seed_v3.yaml"
DEVSET_VERSION = "1"


def build_devset(
    config_path: Path = DEFAULT_CONFIG, approvals: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Detection cases, patch cases and a manifest for the approved validation split."""
    # The training loader's checks (approvals, supervision hashes, SHA256SUMS) gate the set; GovernanceError on any mismatch.
    data = load_training_data(config_path, approvals=approvals, tasks=("detection", "patch"))
    config = load_config(config_path)
    validation_path = config.workdir / "validation.jsonl"
    records = sorted(load_jsonl(validation_path), key=lambda r: r["record_id"])
    approved = set(data.record_ids)
    for record in records:
        if record["split"] != "validation" or record["record_id"] not in approved:
            raise GovernanceError(f"{record['record_id']} is not an approved validation record")
    cases = []
    for record in records:
        case = render.to_eval_case(record)
        case["source_kind"] = "development"
        cases.append(case)
    validate_cases(cases)
    patch_cases = supervision.patch_cases(records)
    manifest = {
        "devset_version": DEVSET_VERSION,
        "dataset_id": config.dataset_id,
        "role": "development (checkpoint selection); not the final independent test",
        "source": "training-approved validation split",
        "validation_jsonl_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "training_data_sha256": data.sha256,
        "records": len(records),
        "repositories": len({r["provenance"]["repository_key"] for r in records}),
        "detection": {
            "cases_sha256": case_set_sha256(cases),
            "count": len(cases),
            "by_expected_decision": dict(sorted(Counter(c["expected_decision"] for c in cases).items())),
            "by_language": dict(sorted(Counter(c["language"] for c in cases).items())),
        },
        "patch": {"cases_sha256": case_set_sha256(patch_cases), "count": len(patch_cases)},
    }
    return cases, patch_cases, manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "devset")
    args = parser.parse_args(argv)
    try:
        cases, patch_cases, manifest = build_devset(args.config)
    except GovernanceError as exc:
        print(f"refused (governance): {exc}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "dev.eval_cases.jsonl", cases)
    _write_jsonl(args.output_dir / "dev.patch_cases.jsonl", patch_cases)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    detection = manifest["detection"]
    print(
        f"development set: {detection['count']} detection cases {detection['by_expected_decision']}, "
        f"{manifest['patch']['count']} patch cases, {manifest['repositories']} repositories -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
