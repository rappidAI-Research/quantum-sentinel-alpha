"""Exact average of LoRA checkpoints of one run (Training Optimization v2, no training).

    python -m sentinel.train.average --run runs/train/<run> --end-step 200 --window 5

averages the checkpoints step-000160 ... step-000200 (the `window` checkpoints that end at `end-step`) into
`runs/train/<run>/averaged/avg5-step-000200/adapter`, which run_model, decision_scores and select_checkpoint
evaluate like any checkpoint (label `<run>/avg5-step-000200`).

The mean of k LoRA updates is itself a LoRA update, exactly: with scaling s = alpha / r,

    (1/k) sum_i s * B_i A_i  =  s * [B_1 ... B_k] / k  @  [A_1; ...; A_k]

i.e. rank k*r with A stacked along the rank dimension, B concatenated and divided by k, and lora_alpha
multiplied by k so that alpha / r stays the same. Nothing is approximated (B / k is computed in float32 and
stored in the checkpoint's dtype). Averaging neighbouring checkpoints is a standard way to damp the
step-to-step swings that the decision showed in V6, E2 and E3 (docs/TRAINING_OPTIMIZATION_V2.md § 4).

Only checkpoints written by the trainer are accepted: same base model, revision, rank, alpha, target modules
and tensor shapes; each one verified against its checksums.json. Every source is recorded in averaging.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

AVERAGING_VERSION = "1"
_IDENTITY_KEYS = ("base_model_name_or_path", "revision", "r", "lora_alpha", "target_modules", "peft_type", "task_type", "bias")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_dirs(run: Path, end_step: int, window: int, every: int) -> list[Path]:
    """The `window` checkpoints ending at `end_step`, `every` steps apart; the final checkpoint of a run may
    lie off the grid (e.g. step 346), so the window ends there and the earlier ones are on the grid."""
    if window < 2:
        raise ValueError("averaging needs a window of at least 2 checkpoints")
    last_on_grid = end_step if end_step % every == 0 else end_step - end_step % every
    steps = sorted({end_step, *(last_on_grid - every * i for i in range(window - (0 if end_step % every == 0 else 1)))})
    if steps[0] <= 0:
        raise ValueError(f"a window of {window} checkpoints does not fit before step {end_step}")
    paths = [run / "checkpoints" / f"step-{s:06d}" for s in steps]
    missing = [str(p) for p in paths if not (p / "adapter" / "adapter_model.safetensors").is_file()]
    if missing:
        raise ValueError(f"missing checkpoints: {missing}")
    return paths


def verify_checkpoint(path: Path) -> dict[str, Any]:
    """The checkpoint's adapter files must be exactly what the trainer wrote (its checksums.json)."""
    recorded = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
    digests = {}
    for name in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json"):
        actual = _sha256(path / name)
        if recorded.get(name) != actual:
            raise ValueError(f"{path / name} does not match {path / 'checksums.json'}")
        digests[name] = actual
    return digests


def lora_pairs(names: list[str]) -> list[tuple[str, str]]:
    """(lora_A, lora_B) tensor names per adapted module; anything else is refused."""
    a_names = sorted(n for n in names if ".lora_A." in n)
    pairs = [(a, a.replace(".lora_A.", ".lora_B.")) for a in a_names]
    covered = {n for pair in pairs for n in pair}
    if covered != set(names):
        raise ValueError(f"not a plain LoRA adapter (unexpected tensors: {sorted(set(names) - covered)[:5]})")
    return pairs


def average_tensors(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """The rank-k*r LoRA tensors whose update is the mean of the sources' updates (see module docstring)."""
    import torch

    names = list(sources[0])
    if any(set(s) != set(names) for s in sources):
        raise ValueError("the checkpoints hold different LoRA tensors")
    k = len(sources)
    out: dict[str, Any] = {}
    for a_name, b_name in lora_pairs(names):
        shapes = {(tuple(s[a_name].shape), tuple(s[b_name].shape)) for s in sources}
        if len(shapes) != 1:
            raise ValueError(f"{a_name}: shapes differ between checkpoints: {shapes}")
        dtype = sources[0][b_name].dtype
        out[a_name] = torch.cat([s[a_name] for s in sources], dim=0).contiguous()
        out[b_name] = (torch.cat([s[b_name].float() for s in sources], dim=1) / k).to(dtype).contiguous()
    return out


def averaged_config(configs: list[dict[str, Any]]) -> dict[str, Any]:
    for key in ("use_rslora", "use_dora"):
        if any(c.get(key) for c in configs):
            raise ValueError(f"{key} adapters are not supported (scaling is not alpha / r)")
    for key in ("rank_pattern", "alpha_pattern", "modules_to_save"):
        if any(c.get(key) for c in configs):
            raise ValueError(f"adapters with {key} are not supported")
    reference = configs[0]
    different = sorted(key for key in _IDENTITY_KEYS for c in configs[1:] if c.get(key) != reference.get(key))
    if different:
        raise ValueError(f"the checkpoints differ in {sorted(set(different))}")
    k = len(configs)
    return {**reference, "r": reference["r"] * k, "lora_alpha": reference["lora_alpha"] * k}


def average(run: Path, end_step: int, window: int, every: int = 10, output: Path | None = None) -> Path:
    from safetensors.torch import load_file, save_file

    paths = checkpoint_dirs(run, end_step, window, every)
    digests = [verify_checkpoint(p) for p in paths]
    configs = [json.loads((p / "adapter" / "adapter_config.json").read_text(encoding="utf-8")) for p in paths]
    config = averaged_config(configs)
    tensors = average_tensors([load_file(str(p / "adapter" / "adapter_model.safetensors")) for p in paths])
    target = output or run / "averaged" / f"avg{window}-step-{end_step:06d}"
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    # written completely under a temporary name, then renamed (as trainer.save_checkpoint does): an interrupted
    # run leaves only `<name>.tmp`, which the next run replaces, never a partial adapter that looks finished
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    adapter = tmp / "adapter"
    adapter.mkdir(parents=True)
    save_file(tensors, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (tmp / "averaging.json").write_text(
        json.dumps(
            {
                "averaging_version": AVERAGING_VERSION,
                "method": "exact mean of the LoRA updates (rank concatenation)",
                "run": str(run),
                "window": window,
                "end_step": end_step,
                "sources": [{"checkpoint": str(p), **d} for p, d in zip(paths, digests)],
                "rank": config["r"],
                "lora_alpha": config["lora_alpha"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = {name: _sha256(tmp / name) for name in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json")}
    (tmp / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.rename(target)
    return target / "adapter"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="runs/train/<run>")
    parser.add_argument("--end-step", type=int, required=True, action="append", help="repeat for several windows")
    parser.add_argument("--window", type=int, default=5, help="number of checkpoints (pre-registered: 5)")
    parser.add_argument("--every", type=int, default=10, help="checkpoint spacing of the run (save_every_steps)")
    args = parser.parse_args(argv)
    for end_step in args.end_step:
        print(average(args.run, end_step, args.window, args.every))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
