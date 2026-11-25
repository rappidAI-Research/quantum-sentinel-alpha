"""Sentinel Garden CLI.

    python -m sentinel.garden morefixes-index --config configs/garden/seed_v6.yaml # v6: MoreFixes dump -> metadata index (offline)
    python -m sentinel.garden discover --config configs/garden/seed_v1.yaml   # OSV metadata (network)
    python -m sentinel.garden build    --config configs/garden/seed_v1.yaml   # upstream git (network)
    python -m sentinel.garden curate   --config configs/garden/seed_v1.yaml   # offline
    python -m sentinel.garden run      --config configs/garden/seed_v1.yaml   # build + curate
    python -m sentinel.garden audit    --config configs/garden/seed_v1.yaml   # quality audit (network: OSV, git)
    python -m sentinel.garden review   --config configs/garden/seed_v1.yaml   # rights-review evidence + queue (network)
    python -m sentinel.garden review-packages                                 # blind evidence packages for a human label review
    python -m sentinel.garden verify   --config configs/garden/seed_v1.yaml   # rebuild, compare hashes
    python -m sentinel.garden reconstruct                                     # fresh checkout -> full records + SHA256SUMS
    python -m sentinel.garden materialize --config configs/garden/seed_v3.yaml # training machine: approved records only
    python -m sentinel.garden materialize-eval --config configs/garden/seed_v4.yaml # evaluation sets only (v4)
    python -m sentinel.garden check    --config configs/garden/seed_v1.yaml   # offline manifest check
    python -m sentinel.garden analyze  --config configs/garden/seed_v4.yaml --output <file.json> [--runs ... --cases ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentinel.garden import pipeline

DEFAULT_CONFIG = pipeline.DEFAULT_CONFIG_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.garden", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["morefixes-index", "discover", "build", "curate", "run", "audit", "review", "review-packages", "verify", "reconstruct", "materialize", "materialize-eval", "check", "analyze"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true", help="discover: re-download the OSV advisory index")
    parser.add_argument("--workers", type=int, default=6, help="build: parallel repositories")
    parser.add_argument("--processes", type=int, default=0, help="build/run: build repositories in this many processes (CPU-bound; same result)")
    parser.add_argument("--output", type=Path, help="analyze: where to write the JSON report")
    parser.add_argument("--runs", nargs="*", type=Path, default=[], help="analyze: run_model artifacts on a development set")
    parser.add_argument("--cases", type=Path, help="analyze: the eval cases the runs were made on")
    args = parser.parse_args(argv)

    config = pipeline.load_config(args.config)
    if args.stage == "morefixes-index":
        pipeline.morefixes_index(config)
    elif args.stage == "discover":
        pipeline.discover(config, refresh=args.refresh)
    elif args.stage == "build":
        pipeline.build(config, workers=args.workers, processes=args.processes)
    elif args.stage == "curate":
        pipeline.curate(config)
    elif args.stage == "run":
        pipeline.build(config, workers=args.workers, processes=args.processes)
        pipeline.curate(config)
    elif args.stage == "audit":
        pipeline.audit(config)
    elif args.stage == "review":
        pipeline.review(config)
    elif args.stage == "review-packages":
        pipeline.review_packages(config)
    elif args.stage == "verify":
        problems = pipeline.verify(config)
        for problem in problems[:50]:
            print(f"  {problem}", file=sys.stderr)
        return 1 if problems else 0
    elif args.stage == "reconstruct":
        report = pipeline.reconstruct(config, workers=args.workers)
        for key in ("missing_records", "content_hash_mismatches", "unexpected_records", "split_drift", "status_drift"):
            if report[key]:
                print(f"  {key}: {len(report[key])} (see reconstruction_report.json)", file=sys.stderr)
        return 0 if report["ok"] else 1
    elif args.stage == "materialize":
        pipeline.materialize(config, workers=args.workers)
    elif args.stage == "materialize-eval":
        pipeline.materialize_evaluation(config, workers=args.workers)
    elif args.stage == "analyze":
        from sentinel.eval.runner import load_artifact
        from sentinel.garden import analysis

        if args.output is None:
            parser.error("analyze needs --output")
        records_path = config.workdir / "records.jsonl"
        cases = pipeline._read_jsonl(args.cases) if args.cases else []
        runs = [load_artifact(path) for path in args.runs]
        if not records_path.is_file():  # an evaluation machine (materialize-eval) has no curated records: runs only
            if not runs:
                parser.error(f"{records_path} is missing (curate or reconstruct first), and no --runs were given")
            analysis.write({"analysis_version": analysis.ANALYSIS_VERSION, "development_runs": [analysis.run_errors(r, {c["case_id"]: c for c in cases}) for r in runs]}, args.output)
            print(f"analyze: {len(runs)} runs -> {args.output}")
            return 0
        records = pipeline._read_jsonl(records_path)
        result = analysis.analyze(records, runs, cases)
        if "evaluation" in config.raw:  # Garden v4: cross-version contamination against every earlier evaluation set
            earlier = {
                Path(path).parent.name: pipeline._read_jsonl(pipeline.PROJECT_ROOT / path)
                for path in config.raw["evaluation"]["previous_holdout_manifests"]
            }
            workdirs = {name: pipeline.PROJECT_ROOT / "artifacts" / "garden" / name / "records.jsonl" for name in earlier}
            with_code = {name: pipeline._read_jsonl(path) for name, path in workdirs.items() if path.is_file()}
            result["contamination"] = analysis.contamination_report(records, earlier, with_code)
        analysis.write(result, args.output)
        print(f"analyze: {result['accepted']['records']} accepted records -> {args.output}")
    elif args.stage == "check":
        result = pipeline.check(config)
        print(f"garden manifest: {result['records']} records, {result['accepted']} accepted, {result['training_approved']} training-approved OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
