"""Deterministic training diagnostics (CPU only, no model weights).

    python -m sentinel.train.diagnostics --config configs/train/qlora_ornith15_9b_v6.yaml \\
        [--tokenizer-dir DIR] [--run runs/train/<run>/run.json] [--window 126-175] --output report.json

Reproduces what a training run saw, from the same governance-checked data and the trainer's own
ordering (`sentinel.train.data.batches`, accumulation windows aligned to each epoch's start):
- the examples of every optimizer step, with their class (finding / no_finding / abstain / patch),
  and the class mix of any window of steps;
- the learning rate of every step (the trainer's cosine schedule with warmup);
- with `--tokenizer-dir`: how many supervised tokens each class contributes, where the decision token
  is, and how much of the training objective and of the validation loss it carries;
- with `--run`: the logged loss, gradient norm and validation loss next to the reconstruction.

Nothing here trains, loads weights or reads an evaluation set.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from sentinel.train.config import PROJECT_ROOT, load_train_config
from sentinel.train.data import DECISION_PREFIX, IGNORE_INDEX, EncodedExample, batches, load_training_data

CLASSES = ("finding", "no_finding", "abstain", "patch")


def example_class(example: dict[str, Any]) -> str:
    """finding / no_finding / abstain for a detection example (from its reply), patch for a patch example."""
    if example.get("task", "detection") != "detection":
        return str(example["task"])
    return str(json.loads(example["messages"][-1]["content"])["decision"])


@dataclass
class OptimizerStep:
    step: int  # 1-based optimizer step
    epoch: int  # 0-based
    examples: list[tuple[str, str]] = field(default_factory=list)  # (task, record_id) in micro-batch order


def optimizer_steps(
    examples: list[EncodedExample], micro_batch_size: int, accumulation_steps: int, epochs: int, seed: int, max_steps: int | None = None
) -> list[OptimizerStep]:
    """The examples of every optimizer step, exactly as `trainer.train` groups them: per-epoch shuffle,
    accumulation windows aligned to the epoch start, a partial last window per epoch."""
    steps: list[OptimizerStep] = []
    for epoch in range(epochs):
        micro = batches(examples, micro_batch_size, epoch, seed)
        for start in range(0, len(micro), accumulation_steps):
            if max_steps is not None and len(steps) >= max_steps:
                return steps
            window = micro[start : start + accumulation_steps]
            steps.append(OptimizerStep(len(steps) + 1, epoch, [(e.task, e.record_id) for batch in window for e in batch]))
    return steps


def learning_rate(step: int, total_steps: int, warmup_ratio: float, peak: float) -> float:
    """Learning rate applied at optimizer step `step` (1-based): transformers' cosine schedule with linear
    warmup (`get_cosine_schedule_with_warmup`), stepped after each optimizer step."""
    current = step - 1
    warmup = int(total_steps * warmup_ratio)
    if current < warmup:
        return peak * current / max(1, warmup)
    progress = (current - warmup) / max(1, total_steps - warmup)
    return peak * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def window_mix(steps: list[OptimizerStep], classes: dict[tuple[str, str], str], first: int, last: int) -> dict[str, Any]:
    chosen = [s for s in steps if first <= s.step <= last]
    counts = Counter(classes[key] for s in chosen for key in s.examples)
    total = sum(counts.values())
    return {
        "steps": f"{first}-{last}",
        "epochs": sorted({s.epoch for s in chosen}),
        "examples": total,
        "by_class": {c: counts.get(c, 0) for c in CLASSES},
        "share": {c: round(counts.get(c, 0) / total, 3) if total else None for c in CLASSES},
    }


def decision_position(tokenizer: Any, reply_ids: list[int]) -> int | None:
    """Index (within the reply) of the token that decides finding / no_finding / abstain: the first token
    after the reply prefix `{"decision": "`, or None when the reply does not start with that prefix."""
    prefix = list(tokenizer(DECISION_PREFIX, add_special_tokens=False)["input_ids"])
    return len(prefix) if reply_ids[: len(prefix)] == prefix and len(reply_ids) > len(prefix) else None


def token_report(tokenizer: Any, examples: list[dict[str, Any]], encoded: list[EncodedExample]) -> dict[str, Any]:
    """Supervised tokens per class, and the weight of the decision token in the objective.

    The trainer's loss is the mean over an example's supervised tokens (micro-batches of one example),
    averaged over the examples of a step; the validation loss is the token-weighted mean over all
    validation examples. Both are reported, so it is visible what each actually rewards."""
    by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    decision_tokens: dict[str, Counter[int]] = {c: Counter() for c in ("finding", "no_finding", "abstain")}
    missing_prefix = 0
    for example, enc in zip(examples, encoded):
        cls = example_class(example)
        supervised = [t for t in enc.labels if t != IGNORE_INDEX]
        by_class[cls].append(len(supervised))
        if cls in decision_tokens:
            position = decision_position(tokenizer, supervised)
            if position is None:
                missing_prefix += 1
            else:
                decision_tokens[cls][supervised[position]] += 1
    total_tokens = sum(sum(v) for v in by_class.values())
    detection = [n for c in ("finding", "no_finding", "abstain") for n in by_class[c]]
    return {
        "examples": {c: len(v) for c, v in by_class.items()},
        "median_reply_tokens": {c: median(v) if v else None for c, v in by_class.items()},
        "share_of_supervised_tokens": {c: round(sum(v) / total_tokens, 4) if total_tokens else None for c, v in by_class.items()},
        # one decisive token per detection example
        "decision_token_share_of_supervised_tokens": round(len(detection) / total_tokens, 4) if total_tokens else None,
        "decision_token_mean_weight_in_its_example": round(sum(1 / n for n in detection) / len(detection), 4) if detection else None,
        "decision_token_ids": {c: dict(v) for c, v in decision_tokens.items()},
        "detection_replies_without_decision_prefix": missing_prefix,
    }


def run_history(run: dict[str, Any], first: int, last: int, max_grad_norm: float) -> dict[str, Any]:
    """What the run logged inside the window (run.json keeps the last 50 log entries and all evaluations)."""
    history = [h for h in (run.get("state") or {}).get("history", []) if first <= h.get("step", 0) <= last]
    evaluations = (run.get("state") or {}).get("evaluations", [])
    return {
        "logged_steps": len(history),
        "loss": [h.get("loss") for h in history],
        "grad_norm": [h.get("grad_norm") for h in history],
        "clipped_steps": sum(1 for h in history if (h.get("grad_norm") or 0) > max_grad_norm),
        "validation_loss": [e for e in evaluations if first - 25 <= e["step"] <= last + 25],
    }


def report(config_path: Path, window: tuple[int, int], tokenizer_dir: Path | None = None, run_path: Path | None = None) -> dict[str, Any]:
    config = load_train_config(config_path)
    data = load_training_data(PROJECT_ROOT / config.data.garden_config, tasks=tuple(config.data.tasks))
    classes = {(e.get("task", "detection"), e["record_id"]): example_class(e) for e in data.train}
    t = config.training
    # ordering needs only (task, record_id); no example is dropped at max_seq_len in the V6 data (checked with --tokenizer-dir)
    keys = [EncodedExample(record_id=e["record_id"], input_ids=[0], labels=[0], task=e.get("task", "detection")) for e in data.train]
    steps = optimizer_steps(keys, t.micro_batch_size, t.gradient_accumulation_steps, t.num_epochs, t.seed, t.max_steps)
    total = len(steps)
    per_epoch = math.ceil(math.ceil(len(keys) / t.micro_batch_size) / t.gradient_accumulation_steps)
    first, last = window
    result: dict[str, Any] = {
        "config": str(config.path),
        "config_sha256": config.sha256,
        "data_sha256": data.sha256,
        "schedule": {
            "steps_per_epoch": per_epoch,
            "total_steps": total,
            "epoch_boundaries_after_steps": [per_epoch * (e + 1) for e in range(t.num_epochs - 1)],
            "last_window_of_epoch_micro_batches": math.ceil(len(keys) / t.micro_batch_size) - (per_epoch - 1) * t.gradient_accumulation_steps,
            "learning_rate": {str(s): learning_rate(s, total, t.warmup_ratio, t.learning_rate) for s in sorted({1, first, last, *range(25, total + 1, 25), total})},
        },
        "overall_mix": window_mix(steps, classes, 1, total),
        "windows": [window_mix(steps, classes, a, min(a + 24, total)) for a in range(1, total + 1, 25)],
        "focus_window": window_mix(steps, classes, first, last),
        "focus_steps": [
            {"step": s.step, "epoch": s.epoch, **{c: sum(1 for k in s.examples if classes[k] == c) for c in CLASSES}} for s in steps if first <= s.step <= last
        ],
    }
    if tokenizer_dir is not None:
        from transformers import AutoTokenizer

        from sentinel.train.data import encode_all

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        train_encoded, dropped = encode_all(tokenizer, data.train, config.data.max_seq_len, config.data.end_of_turn)
        if dropped:
            result["warning"] = f"{len(dropped)} train examples exceed max_seq_len; the reconstruction above includes them"
        validation_encoded, _ = encode_all(tokenizer, data.validation, config.data.max_seq_len, config.data.end_of_turn)
        result["tokens"] = {"train": token_report(tokenizer, data.train, train_encoded), "validation": token_report(tokenizer, data.validation, validation_encoded)}
    if run_path is not None:
        result["run"] = run_history(json.loads(run_path.read_text(encoding="utf-8")), first, last, t.max_grad_norm)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--window", default="126-175", help="optimizer steps to inspect, e.g. 126-175")
    parser.add_argument("--tokenizer-dir", type=Path, help="local copy of the pinned tokenizer files (token statistics)")
    parser.add_argument("--run", type=Path, help="the run's run.json (logged loss and gradient norms)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    first, _, last = args.window.partition("-")
    result = report(args.config, (int(first), int(last)), args.tokenizer_dir, args.run)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    s = result["schedule"]
    print(f"{s['total_steps']} steps, {s['steps_per_epoch']} per epoch (boundary after {s['epoch_boundaries_after_steps']}), last window {s['last_window_of_epoch_micro_batches']} micro-batches")
    for row in result["windows"] + [result["focus_window"]]:
        print(f"steps {row['steps']:>9} epochs {row['epochs']}: " + ", ".join(f"{c} {row['share'][c]:.0%}" for c in CLASSES))
    if "tokens" in result:
        tr = result["tokens"]["train"]
        print(f"decision token: {tr['decision_token_share_of_supervised_tokens']:.2%} of supervised train tokens, mean weight {tr['decision_token_mean_weight_in_its_example']:.2%} inside its example")
        print("validation loss token shares: " + ", ".join(f"{c} {v:.1%}" for c, v in result["tokens"]["validation"]["share_of_supervised_tokens"].items() if v is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
