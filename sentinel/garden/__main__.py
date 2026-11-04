"""Sentinel Garden CLI.

    python -m sentinel.garden discover --config configs/garden/seed_v1.yaml   # OSV metadata (network)
    python -m sentinel.garden build    --config configs/garden/seed_v1.yaml   # upstream git (network)
    python -m sentinel.garden curate   --config configs/garden/seed_v1.yaml   # offline
    python -m sentinel.garden run      --config configs/garden/seed_v1.yaml   # build + curate
    python -m sentinel.garden verify   --config configs/garden/seed_v1.yaml   # rebuild, compare hashes
    python -m sentinel.garden check    --config configs/garden/seed_v1.yaml   # offline manifest check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentinel.garden import pipeline

DEFAULT_CONFIG = pipeline.DEFAULT_CONFIG_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.garden", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["discover", "build", "curate", "run", "verify", "check"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true", help="discover: re-download the OSV advisory index")
    parser.add_argument("--workers", type=int, default=6, help="build: parallel repositories")
    args = parser.parse_args(argv)

    config = pipeline.load_config(args.config)
    if args.stage == "discover":
        pipeline.discover(config, refresh=args.refresh)
    elif args.stage == "build":
        pipeline.build(config, workers=args.workers)
    elif args.stage == "curate":
        pipeline.curate(config)
    elif args.stage == "run":
        pipeline.build(config, workers=args.workers)
        pipeline.curate(config)
    elif args.stage == "verify":
        problems = pipeline.verify(config)
        for problem in problems[:50]:
            print(f"  {problem}", file=sys.stderr)
        return 1 if problems else 0
    elif args.stage == "check":
        result = pipeline.check(config)
        print(f"garden manifest: {result['records']} records, {result['accepted']} accepted, {result['training_approved']} training-approved OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
