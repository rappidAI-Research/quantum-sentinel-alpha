"""QLoRA training loop for the selected foundation (adapter-only output).

Heavy dependencies (torch, transformers, peft, bitsandbytes) are imported lazily. Every run writes
`run.json` with its configuration, data hash, foundation revision (requested and loaded), library
versions, seeds, metrics and final status - also when it fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sentinel.train.config import TrainConfig, matching_modules
from sentinel.train.data import EncodedExample, TrainingData, batches, encode_all, pad_batch

RUN_VERSION = "1"
EXIT_FAILED = 1
EXIT_GOVERNANCE = 2
EXIT_OOM = 3
EXIT_INTERRUPTED = 4


class FoundationMismatch(RuntimeError):
    pass


class TargetModuleMismatch(RuntimeError):
    pass


@dataclass
class RunState:
    step: int = 0  # optimizer steps completed
    epoch: int = 0
    batch_in_epoch: int = 0  # micro-batches of the current epoch already consumed
    history: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False)
            return done.stdout.strip() if done.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_foundation_model(config: TrainConfig) -> tuple[Any, Any]:
    """Tokenizer + model at the exact pinned revision, 4-bit NF4 when configured."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, config.quantization.compute_dtype)
    kwargs: dict[str, Any] = {"revision": config.revision, "dtype": dtype}
    if config.quantization.mode == "nf4":
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA needs a CUDA GPU (bitsandbytes); use quantization.mode: none only for CPU tests")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
    model, info = AutoModelForCausalLM.from_pretrained(config.model_id, output_loading_info=True, **kwargs)
    missing = list(info.get("missing_keys") or [])
    if missing:
        raise RuntimeError(f"checkpoint is missing weights ({len(missing)}), e.g. {missing[:5]}")
    return model, tokenizer


def verify_foundation(model: Any, config: TrainConfig) -> None:
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved != config.revision:
        raise FoundationMismatch(f"loaded revision {resolved!r}, expected pinned {config.revision}")


def apply_lora(model: Any, config: TrainConfig, gradient_checkpointing: bool) -> tuple[Any, list[str]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    linear_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) or type(module).__name__.startswith("Linear")]
    targets = matching_modules(config.lora.target_modules, linear_names)
    if len(targets) != config.lora.expected_target_count:
        raise TargetModuleMismatch(
            f"target_modules matched {len(targets)} linear modules, expected {config.lora.expected_target_count}; "
            "the architecture differs from the inspected one (configs/train/qwen35_9b_linear_modules.txt)"
        )
    if config.quantization.mode == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        revision=config.revision,
    )
    return get_peft_model(model, lora), targets


def _to_tensors(batch: list[EncodedExample], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    padded = pad_batch(batch, pad_id)
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in padded.items()}


def save_adapter(model: Any, directory: Path) -> list[str]:
    """Save only the LoRA weights and refuse to write anything that is not an adapter tensor."""
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model)
    foreign = [k for k in state if "lora_" not in k]
    if foreign:
        raise RuntimeError(f"adapter state contains non-LoRA tensors: {foreign[:5]}")
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    return sorted(state)


_CHECKPOINT_NAME = re.compile(r"^step-\d{6}$")


def _checkpoints(output_dir: Path) -> list[Path]:
    """Completed checkpoints only: an interrupted save leaves a `step-NNNNNN.tmp` directory, which is
    never resumed from or counted for retention."""
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and _CHECKPOINT_NAME.match(p.name) and (p / "state.json").is_file())


def accumulation_window(index: int, micro_batches: int, accumulation_steps: int) -> int:
    """Number of micro-batches in the optimizer step that micro-batch `index` belongs to.

    Steps are aligned to the start of each epoch, so the last step of an epoch can be a partial
    window; dividing its losses by the full accumulation count would under-weight that step (and
    change clipping and AdamW moments), so each loss is divided by its own window's size.
    """
    start = (index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, micro_batches - start)


def checkpoint_provenance(config: TrainConfig, data: TrainingData) -> dict[str, Any]:
    """What a checkpoint was trained on. Resuming is only allowed with exactly the same inputs."""
    return {
        "base_model": config.model_id,
        "base_revision": config.revision,
        "config_sha256": config.sha256,
        "data_sha256": data.sha256,
        "record_ids_sha256": hashlib.sha256("\n".join(data.record_ids).encode("utf-8")).hexdigest(),
    }


def check_resume_provenance(path: Path, expected: dict[str, Any]) -> None:
    file = path / "provenance.json"
    if not file.is_file():
        raise RuntimeError(f"{path} has no provenance.json; refusing to resume (start a new run)")
    found = json.loads(file.read_text(encoding="utf-8"))
    changed = sorted(k for k in expected if found.get(k) != expected[k])
    if changed:
        raise RuntimeError(
            f"cannot resume from {path}: it was trained with different {', '.join(changed)} "
            "(the training data or configuration changed; start a new run in a new output directory)"
        )


def save_checkpoint(model: Any, optimizer: Any, scheduler: Any, state: RunState, output_dir: Path, keep: int, provenance: dict[str, Any]) -> Path:
    import torch

    path = output_dir / "checkpoints" / f"step-{state.step:06d}"
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    save_adapter(model, tmp / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "trainer_state.pt",
    )
    (tmp / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    (tmp / "state.json").write_text(json.dumps(vars(state), indent=2), encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)  # a checkpoint directory only appears once it is complete
    for old in _checkpoints(output_dir)[:-keep]:
        shutil.rmtree(old)
    return path


def load_checkpoint(model: Any, optimizer: Any, scheduler: Any, path: Path) -> RunState:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(str(path / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, weights)
    blob = torch.load(path / "trainer_state.pt", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    random.setstate(blob["python_rng"])
    torch.set_rng_state(blob["torch_rng"])
    if blob["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return RunState(**json.loads((path / "state.json").read_text(encoding="utf-8")))


def evaluate_loss(model: Any, examples: list[EncodedExample], micro_batch_size: int, pad_id: int) -> float | None:
    import torch

    if not examples:
        return None
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in batches(examples, micro_batch_size, epoch=0, seed=0):
            inputs = _to_tensors(batch, pad_id, model.device)
            count = int((inputs["labels"][:, 1:] != -100).sum())
            loss = model(**inputs).loss
            total += float(loss) * count
            tokens += count
    model.train()
    return total / tokens if tokens else None


def _peak_vram_gib() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def _versions() -> dict[str, Any]:
    import importlib.metadata as md

    versions: dict[str, Any] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None
    try:
        import torch

        versions["cuda"] = torch.version.cuda
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return versions


def train(
    config: TrainConfig,
    data: TrainingData,
    *,
    resume: bool = False,
    model_loader: Callable[[TrainConfig], tuple[Any, Any]] = load_foundation_model,
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run QLoRA training. Writes run.json in every outcome and returns it."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    started = time.monotonic()
    run: dict[str, Any] = {
        "run_version": RUN_VERSION,
        "kind": "quantum-sentinel-alpha.qlora-run",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_root or config.path.parent),
        "config": {"path": str(config.path), "sha256": config.sha256, **config.as_dict()},
        "data": {
            "sha256": data.sha256,
            "train_examples": len(data.train),
            "validation_examples": len(data.validation),
            "repositories": data.repositories,
            "record_ids_sha256": hashlib.sha256("\n".join(data.record_ids).encode()).hexdigest(),
        },
        "foundation": {"model_id": config.model_id, "requested_revision": config.revision, "loaded_revision": None},
        "versions": _versions(),
        "seed": config.training.seed,
    }

    def write_run() -> None:
        run["elapsed_seconds"] = round(time.monotonic() - started, 3)
        run["peak_vram_gib"] = _peak_vram_gib()
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    model = optimizer = scheduler = None
    state = RunState()
    try:
        seed_everything(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        load_started = time.monotonic()
        base, tokenizer = model_loader(config)
        run["foundation"]["loaded_revision"] = getattr(base.config, "_commit_hash", None)
        verify_foundation(base, config)  # whatever loaded the model, it must be the pinned revision
        run["model_load_seconds"] = round(time.monotonic() - load_started, 3)

        train_examples, dropped_train = encode_all(tokenizer, data.train, config.data.max_seq_len, config.data.end_of_turn)
        validation_examples, dropped_validation = encode_all(tokenizer, data.validation, config.data.max_seq_len, config.data.end_of_turn)
        run["data"].update(
            train_encoded=len(train_examples),
            train_dropped_overlength=dropped_train,
            validation_encoded=len(validation_examples),
            validation_dropped_overlength=dropped_validation,
            train_tokens=sum(len(e.input_ids) for e in train_examples),
            train_supervised_tokens=sum(sum(1 for t in e.labels if t != -100) for e in train_examples),
        )
        if not train_examples:
            raise RuntimeError("no training example fits max_seq_len")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        model, targets = apply_lora(base, config, config.training.gradient_checkpointing)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run["lora"] = {"target_modules": len(targets), "trainable_parameters": trainable}

        micro_batches_per_epoch = math.ceil(len(train_examples) / config.training.micro_batch_size)
        steps_per_epoch = math.ceil(micro_batches_per_epoch / config.training.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.training.num_epochs
        if config.training.max_steps is not None:
            total_steps = min(total_steps, config.training.max_steps)
        run["schedule"] = {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps}

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=config.training.learning_rate, weight_decay=config.training.weight_decay
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * config.training.warmup_ratio), total_steps)

        provenance = checkpoint_provenance(config, data)
        if resume:
            checkpoints = _checkpoints(output_dir)
            if not checkpoints:
                raise RuntimeError(f"--resume given but no checkpoint in {output_dir / 'checkpoints'}")
            check_resume_provenance(checkpoints[-1], provenance)
            state = load_checkpoint(model, optimizer, scheduler, checkpoints[-1])
            run["resumed_from"] = str(checkpoints[-1])
            log(f"resumed from {checkpoints[-1]} at step {state.step}")

        model.train()
        device = model.device
        accumulated = 0
        window_tokens, window_started = 0, time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        while state.step < total_steps and state.epoch < config.training.num_epochs:
            epoch_batches = batches(train_examples, config.training.micro_batch_size, state.epoch, config.training.seed)
            for index in range(state.batch_in_epoch, len(epoch_batches)):
                inputs = _to_tensors(epoch_batches[index], pad_id, device)
                window = accumulation_window(index, len(epoch_batches), config.training.gradient_accumulation_steps)
                loss = model(**inputs).loss / window
                if not math.isfinite(float(loss.detach())):
                    raise RuntimeError(f"non-finite loss at step {state.step + 1} (micro-batch {index} of epoch {state.epoch})")
                loss.backward()
                accumulated += 1
                window_tokens += int(inputs["attention_mask"].sum())
                last_in_epoch = index == len(epoch_batches) - 1
                if accumulated < config.training.gradient_accumulation_steps and not last_in_epoch:
                    continue
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                state.step += 1
                # only advanced at an optimizer step, so an interrupt checkpoint never skips unapplied gradients
                state.batch_in_epoch = index + 1
                if state.step % config.training.logging_every_steps == 0:
                    elapsed = time.monotonic() - window_started
                    entry = {
                        "step": state.step,
                        "epoch": state.epoch,
                        "loss": round(float(loss.detach()) * window, 5),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": round(grad_norm, 5),
                        "tokens_per_second": round(window_tokens / elapsed, 2) if elapsed > 0 else None,
                        "peak_vram_gib": _peak_vram_gib(),
                    }
                    state.history.append(entry)
                    log(json.dumps(entry))
                    window_tokens, window_started = 0, time.monotonic()
                if state.step % config.training.eval_every_steps == 0:
                    state.evaluations.append({"step": state.step, "validation_loss": evaluate_loss(model, validation_examples, config.training.micro_batch_size, pad_id)})
                if state.step % config.training.save_every_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, state, output_dir, config.training.keep_last_checkpoints, provenance)
                if state.step >= total_steps:
                    break
            if state.batch_in_epoch >= len(epoch_batches):
                state.epoch += 1
                state.batch_in_epoch = 0

        state.evaluations.append({"step": state.step, "validation_loss": evaluate_loss(model, validation_examples, config.training.micro_batch_size, pad_id)})
        save_checkpoint(model, optimizer, scheduler, state, output_dir, config.training.keep_last_checkpoints, provenance)
        adapter_keys = save_adapter(model, output_dir / "adapter")
        (output_dir / "adapter" / "training_run.json").write_text(
            json.dumps({"run_name": config.run_name, "base_model": config.model_id, "base_revision": config.revision, "data_sha256": data.sha256}, indent=2) + "\n",
            encoding="utf-8",
        )
        run.update(status="completed", adapter={"path": str(output_dir / "adapter"), "tensors": len(adapter_keys)})
        return run
    except KeyboardInterrupt:
        run["status"] = "interrupted"
        if model is not None and optimizer is not None and state.step > 0:
            run["interrupt_checkpoint"] = str(save_checkpoint(model, optimizer, scheduler, state, output_dir, config.training.keep_last_checkpoints, provenance))
        raise
    except torch.cuda.OutOfMemoryError as exc:
        run.update(
            status="failed_oom",
            error=str(exc).splitlines()[0],
            hint="reduce training.micro_batch_size (and raise gradient_accumulation_steps) or data.max_seq_len; "
            "keep gradient_checkpointing on; resume from the last checkpoint with --resume",
        )
        raise
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run["state"] = {"step": state.step, "epoch": state.epoch, "history": state.history[-50:], "evaluations": state.evaluations}
        write_run()
