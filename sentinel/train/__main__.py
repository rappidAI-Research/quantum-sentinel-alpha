"""QLoRA training CLI.

    python -m sentinel.train --dry-run          # config + governance + data checks, no model, no GPU
    python -m sentinel.train --dry-run --check-tokens   # + encode every example with the pinned tokenizer (small files only)
    python -m sentinel.train                    # train (CUDA GPU; downloads the pinned foundation)
    python -m sentinel.train --resume           # continue from the latest checkpoint

Exit codes: 0 ok, 1 failed, 2 governance/data refusal, 3 CUDA out of memory, 4 interrupted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentinel.train.config import DEFAULT_TRAIN_CONFIG, PROJECT_ROOT, ConfigError, load_train_config
from sentinel.train.data import GovernanceError, load_training_data


def token_report(config, data, tokenizer_dir: Path | None) -> str:  # type: ignore[no-untyped-def]
    """Encode every train/validation example exactly as the trainer does and report the token budget."""
    from statistics import median

    from transformers import AutoTokenizer

    from sentinel.train.data import encode_all

    source = str(tokenizer_dir) if tokenizer_dir else config.model_id
    tokenizer = AutoTokenizer.from_pretrained(source, revision=None if tokenizer_dir else config.revision)
    lines = []
    for split, examples in (("train", data.train), ("validation", data.validation)):
        kept, dropped = encode_all(tokenizer, examples, config.data.max_seq_len, config.data.end_of_turn)
        lengths = [len(e.input_ids) for e in kept]
        supervised = sum(sum(1 for t in e.labels if t >= 0) for e in kept)
        lines.append(
            f"tokens {split}: {len(kept)} kept, {len(dropped)} dropped at {config.data.max_seq_len}; "
            f"median {median(lengths) if lengths else 0:g}, max {max(lengths, default=0)}, total {sum(lengths):,}, "
            f"supervised {supervised:,}" + (f"; dropped {dropped[:5]}" if dropped else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sentinel.train", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="validate config, governance and data; load no model")
    parser.add_argument(
        "--check-tokens",
        action="store_true",
        help="with --dry-run: encode every example with the foundation tokenizer at the pinned revision (no weights)",
    )
    parser.add_argument("--tokenizer-dir", type=Path, help="local copy of the pinned tokenizer files for --check-tokens (offline)")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    parser.add_argument("--allow-dirty", action="store_true", help="permit a run from uncommitted code (recorded in run.json)")
    args = parser.parse_args(argv)

    try:
        config = load_train_config(args.config)
    except (ConfigError, ValueError) as exc:
        print(f"invalid training config: {exc}", file=sys.stderr)
        return 1
    try:
        data = load_training_data(PROJECT_ROOT / config.data.garden_config, tasks=config.data.tasks)
    except GovernanceError as exc:
        print(f"refused (governance): {exc}", file=sys.stderr)
        return 2
    print(
        f"foundation {config.model_id}@{config.revision}; train {len(data.train)} / validation {len(data.validation)} "
        f"approved examples from {data.repositories} repositories (tasks {', '.join(data.tasks)}: {data.examples_by_task}); "
        f"data sha256 {data.sha256[:12]}"
    )
    if args.check_tokens:
        try:
            print(token_report(config, data, args.tokenizer_dir))
        except ValueError as exc:
            print(f"refused (tokenization): {exc}", file=sys.stderr)
            return 1
    if config.init_adapter is not None:
        from sentinel.train.trainer import InitAdapterMismatch, check_init_adapter

        try:
            info = check_init_adapter(config, PROJECT_ROOT)
        except InitAdapterMismatch as exc:
            print(f"refused (initial adapter): {exc}", file=sys.stderr)
            return 2
        print(f"initial adapter {info['checkpoint']} (step {info['step']}, weights sha256 {info['weights_sha256'][:12]}); new optimizer and schedule")
    if args.dry_run:
        print("dry run: configuration, governance and data checks passed; no model loaded")
        return 0

    from sentinel.train.trainer import EXIT_INTERRUPTED, EXIT_OOM, git_state, train

    if git_state(PROJECT_ROOT)["dirty"] and not args.allow_dirty:
        print("refused: uncommitted changes; commit them or pass --allow-dirty (recorded in run.json)", file=sys.stderr)
        return 1
    import torch

    try:
        run = train(config, data, resume=args.resume, project_root=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("interrupted; resume with --resume", file=sys.stderr)
        return EXIT_INTERRUPTED
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA out of memory; see {config.output_dir / 'run.json'} for the hint and resume with --resume", file=sys.stderr)
        return EXIT_OOM
    except Exception as exc:  # noqa: BLE001 - recorded in run.json, reported here
        print(f"training failed: {type(exc).__name__}: {exc} (details in {config.output_dir / 'run.json'})", file=sys.stderr)
        return 1
    print(f"completed: adapter in {run['adapter']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
