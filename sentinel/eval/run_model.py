"""Run QuickEval against one model and write a machine-readable run artifact.

Direct:   python -m sentinel.eval.run_model --model Qwen/Qwen3.5-4B --revision <sha> \\
              --cases eval/sentinelbench/quickeval_v1.jsonl --output runs/qwen35-4b.json
Bake-off: python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml \\
              --candidate qwen35-4b --revision <sha> --output runs/qwen35-4b.json
No GPU:   add --backend baseline to exercise the full path with the regex baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentinel.eval.backends import (
    DTYPES,
    QUANTIZATION_MODES,
    BaselineBackend,
    GenerationSettings,
    HFBackend,
    ModelBackend,
    RuntimeSettings,
)
from sentinel.eval.bakeoff import load_bakeoff
from sentinel.eval.coding_regression import SUITES
from sentinel.eval.io import PROJECT_ROOT, load_jsonl
from sentinel.eval.runner import is_immutable_revision, run_quickeval, write_artifact

# Flags that would change the shared bake-off setup; rejected together with --bakeoff.
SHARED_SETUP_FLAGS = ("model", "cases", "device", "dtype", "quantization", "max_new_tokens")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Quantum Sentinel QuickEval against a model.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bakeoff", type=Path, help="bake-off config; fixes cases, prompt and generation settings")
    parser.add_argument("--candidate", help="candidate name inside the bake-off config")
    parser.add_argument("--model", help="Hugging Face model ID (without --bakeoff)")
    parser.add_argument("--revision", help="immutable 40-character model commit SHA")
    parser.add_argument("--cases", type=Path, help="QuickEval case JSONL (without --bakeoff)")
    parser.add_argument("--backend", choices=("hf", "baseline"), default="hf")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=DTYPES)
    parser.add_argument("--quantization", choices=QUANTIZATION_MODES)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="smoke test on the first N cases; never reportable")
    parser.add_argument("--allow-unpinned", action="store_true", help="permit a non-reportable run without a pinned revision")
    parser.add_argument("--coding-suite", help=f"coding-regression suite ({', '.join(SUITES) or 'none registered yet'})")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    bakeoff_meta = None
    if args.bakeoff:
        conflicting = [f"--{name.replace('_', '-')}" for name in SHARED_SETUP_FLAGS if getattr(args, name) is not None]
        if conflicting:
            parser.error(f"{', '.join(conflicting)} cannot be combined with --bakeoff; edit the shared config instead")
        if not args.candidate:
            parser.error("--candidate is required with --bakeoff")
        config = load_bakeoff(args.bakeoff)
        if args.candidate not in config.candidates:
            parser.error(f"unknown candidate {args.candidate!r}; choose from {sorted(config.candidates)}")
        candidate = config.candidates[args.candidate]
        if args.revision and candidate.revision and args.revision != candidate.revision:
            parser.error(f"--revision {args.revision} contradicts the pinned config revision {candidate.revision}")
        model_id, revision = candidate.model_id, args.revision or candidate.revision
        cases_path, generation, runtime = config.cases, config.generation, candidate.runtime
        bakeoff_meta = {
            "id": config.bakeoff_id,
            "candidate": candidate.name,
            "config_sha256": config.sha256,
            "runtime_override_reason": candidate.override_reason,
        }
    else:
        if args.backend == "hf" and not args.model:
            parser.error("--model is required")
        if not args.cases:
            parser.error("--cases is required")
        model_id = args.model or "baseline/regex"
        revision = args.revision
        cases_path = args.cases
        generation = GenerationSettings(**({"max_new_tokens": args.max_new_tokens} if args.max_new_tokens else {}))
        runtime = RuntimeSettings(
            **{key: value for key in ("device", "dtype", "quantization") if (value := getattr(args, key)) is not None}
        )

    if args.backend == "baseline":
        model_id, revision = "baseline/regex", None
    elif not is_immutable_revision(revision) and not args.allow_unpinned:
        parser.error("a real model run requires --revision <40-char commit SHA> (or --allow-unpinned for a non-reportable smoke run)")

    cases = load_jsonl(cases_path)
    full_case_set = args.limit is None or args.limit >= len(cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    coding_suite = None
    if args.coding_suite:
        if args.coding_suite not in SUITES:
            parser.error(f"unknown coding suite {args.coding_suite!r}")
        coding_suite = SUITES[args.coding_suite]()

    backend: ModelBackend
    if args.backend == "baseline":
        backend = BaselineBackend()
    else:
        backend = HFBackend(model_id, revision, runtime, generation)

    try:
        display_path = str(cases_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(cases_path)
    artifact = run_quickeval(
        cases,
        backend,
        model_id=model_id,
        revision=revision,
        generation=generation,
        runtime=runtime,
        cases_path=display_path,
        full_case_set=full_case_set,
        bakeoff=bakeoff_meta,
        coding_suite=coding_suite,
    )
    write_artifact(args.output, artifact)

    metrics = artifact["metrics"]
    print(f"wrote {args.output} ({artifact['cases']['count']} cases, reportable={artifact['reportable']})")
    for blocker in artifact["reportable_blockers"]:
        print(f"  not reportable: {blocker}", file=sys.stderr)
    print(
        f"precision={metrics['precision']:.3f} fpr={metrics['false_positive_rate']:.3f} "
        f"recall={metrics['recall']:.3f} invalid_rate={metrics['invalid_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
