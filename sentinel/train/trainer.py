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

from sentinel.eval.patch import PATCH_PROMPT_VERSION
from sentinel.eval.prompt import PROMPT_VERSION
from sentinel.hf_revision import reset_cuda_peak_memory
from sentinel.train.config import PROJECT_ROOT, ObjectiveSettings, TrainConfig, matching_modules
from sentinel.train.data import IGNORE_INDEX, EncodedExample, TrainingData, batches, encode_all, pad_batch

# 2: checksummed checkpoints loaded with weights_only, window-mean loss, task mix recorded
# 3: validation reports loss per task and decision metrics; optional decision-token / task loss weights
RUN_VERSION = "3"
DECISIONS = ("finding", "no_finding", "abstain")
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


def verify_foundation(config: TrainConfig) -> dict[str, Any]:
    """Prove the loaded files are the pinned revision (content digests of the resolved cache snapshot).

    model.config._commit_hash is not used: it is None for the pinned Ornith checkpoint, and the
    requested revision is never taken as evidence of the loaded one.
    """
    from sentinel.hf_revision import RevisionUnverified, verify_loaded_revision

    try:
        return verify_loaded_revision(config.model_id, config.revision)
    except RevisionUnverified as exc:
        raise FoundationMismatch(f"loaded revision not verified as the pinned {config.revision}: {exc}") from exc


def apply_lora(model: Any, config: TrainConfig, gradient_checkpointing: bool) -> tuple[Any, list[str]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    linear_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) or type(module).__name__.startswith("Linear")]
    targets = matching_modules(config.lora.target_modules, linear_names)
    if len(targets) != config.lora.expected_target_count:
        raise TargetModuleMismatch(
            f"target_modules matched {len(targets)} linear modules, expected {config.lora.expected_target_count}; "
            "the architecture differs from the inspected one (configs/train/ornith15_9b_linear_modules.txt)"
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


def checkpoint_provenance(config: TrainConfig, data: TrainingData, init_adapter: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a checkpoint was trained on. Resuming is only allowed with exactly the same inputs."""
    provenance = {
        "base_model": config.model_id,
        "base_revision": config.revision,
        "config_sha256": config.sha256,
        "data_sha256": data.sha256,
        "record_ids_sha256": hashlib.sha256("\n".join(data.record_ids).encode("utf-8")).hexdigest(),
    }
    if init_adapter is not None:  # a continued run: its checkpoints descend from this adapter
        provenance["init_adapter_weights_sha256"] = init_adapter["weights_sha256"]
    return provenance


CHECKSUMS = "checksums.json"


def _file_digests(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != CHECKSUMS
    }


def verify_checkpoint_files(path: Path) -> None:
    """Every file of a checkpoint must be exactly what the trainer wrote (no extra, missing or changed file)."""
    file = path / CHECKSUMS
    if not file.is_file():
        raise RuntimeError(f"{path} has no {CHECKSUMS}; refusing to resume from an unverifiable checkpoint")
    recorded = json.loads(file.read_text(encoding="utf-8"))
    found = _file_digests(path)
    if found != recorded:
        changed = sorted(set(found) ^ set(recorded) | {k for k in set(found) & set(recorded) if found[k] != recorded[k]})
        raise RuntimeError(f"checkpoint {path} was modified after it was written ({', '.join(changed[:5])}); refusing to resume")


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


class InitAdapterMismatch(RuntimeError):
    pass


def check_init_adapter(config: TrainConfig, project_root: Path) -> dict[str, Any]:
    """Pure checks of the configured initial adapter (no torch); raises InitAdapterMismatch.

    The checkpoint must be unmodified (its checksums.json), written by the run the config names
    (provenance: pinned foundation, `source_config_sha256`, `source_data_sha256`), at `step`, and its
    LoRA shape must equal this config's, so the new LoRA modules receive every tensor unchanged."""
    settings = config.init_adapter
    assert settings is not None
    path = project_root / settings.checkpoint
    if not path.is_dir():
        raise InitAdapterMismatch(f"initial adapter checkpoint {path} does not exist")
    try:
        verify_checkpoint_files(path)
    except RuntimeError as exc:
        raise InitAdapterMismatch(str(exc)) from exc
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    expected = {
        "base_model": config.model_id,
        "base_revision": config.revision,
        "config_sha256": settings.source_config_sha256,
        "data_sha256": settings.source_data_sha256,
    }
    changed = sorted(k for k, v in expected.items() if provenance.get(k) != v)
    if changed:
        raise InitAdapterMismatch(f"{path} was not written by the configured source run ({', '.join(changed)} differ)")
    step = json.loads((path / "state.json").read_text(encoding="utf-8")).get("step")
    if step != settings.step:
        raise InitAdapterMismatch(f"{path} is at step {step}, expected {settings.step}")
    adapter_config = json.loads((path / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    shape = {
        "base_model_name_or_path": config.model_id,
        "revision": config.revision,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "target_modules": config.lora.target_modules,
    }
    differs = sorted(k for k, v in shape.items() if adapter_config.get(k) != v)
    if differs:
        raise InitAdapterMismatch(f"initial adapter LoRA settings differ from this config: {', '.join(differs)}")
    weights = path / "adapter" / "adapter_model.safetensors"
    return {
        "checkpoint": settings.checkpoint,
        "step": step,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "adapter_config_sha256": hashlib.sha256((path / "adapter" / "adapter_config.json").read_bytes()).hexdigest(),
        "source_run": provenance,
    }


def load_init_adapter(model: Any, config: TrainConfig, project_root: Path) -> dict[str, Any]:
    """Load the checked initial adapter into the fresh LoRA modules and prove every tensor arrived.

    Only adapter weights are taken: the optimizer, scheduler and RNG state of the source checkpoint
    are not loaded, so training continues with the new run's optimizer and learning-rate schedule."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    info = check_init_adapter(config, project_root)
    weights = load_file(str(project_root / config.init_adapter.checkpoint / "adapter" / "adapter_model.safetensors"))  # type: ignore[union-attr]
    set_peft_model_state_dict(model, weights)
    loaded = get_peft_model_state_dict(model)
    if set(loaded) != set(weights):
        missing, extra = sorted(set(weights) - set(loaded)), sorted(set(loaded) - set(weights))
        raise InitAdapterMismatch(f"initial adapter tensors do not match the LoRA modules (missing {missing[:3]}, extra {extra[:3]})")
    different = [k for k in weights if not torch.equal(loaded[k].detach().to("cpu", weights[k].dtype), weights[k])]
    if different:
        raise InitAdapterMismatch(f"{len(different)} initial adapter tensors were not loaded exactly (first: {different[0]})")
    return {**info, "tensors": len(weights), "parameters": sum(t.numel() for t in weights.values())}


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
    (tmp / CHECKSUMS).write_text(json.dumps(_file_digests(tmp), indent=2, sort_keys=True), encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)  # a checkpoint directory only appears once it is complete
    for old in _checkpoints(output_dir)[:-keep]:
        shutil.rmtree(old)
    return path


def load_checkpoint(model: Any, optimizer: Any, scheduler: Any, path: Path) -> RunState:
    """Resume state. The checkpoint is verified against its checksums, adapter weights come from
    safetensors and the optimizer/scheduler/RNG state is unpickled with `weights_only=True` (tensors
    and plain containers only), so a planted pickle cannot execute code on load."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    verify_checkpoint_files(path)
    weights = load_file(str(path / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, weights)
    blob = torch.load(path / "trainer_state.pt", weights_only=True)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    random.setstate(blob["python_rng"])
    torch.set_rng_state(blob["torch_rng"])
    if blob["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return RunState(**json.loads((path / "state.json").read_text(encoding="utf-8")))


def decision_token_ids(examples: list[EncodedExample]) -> dict[str, int]:
    """The token that decides each class, read from the training examples themselves (it must be one token
    per class, the same in every example)."""
    seen: dict[str, set[int]] = {}
    for e in examples:
        if e.decision is not None and e.decision_index is not None:
            seen.setdefault(e.decision, set()).add(e.input_ids[e.decision_index])
    inconsistent = {d: sorted(t) for d, t in seen.items() if len(t) != 1}
    if inconsistent:
        raise ValueError(f"a decision is spelled with different first tokens: {inconsistent}")
    ids = {d: next(iter(t)) for d, t in seen.items()}
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"decisions share their first token: {ids}")
    return ids


def example_losses(logits: Any, labels: Any, batch: list[EncodedExample], decision_token_weight: float = 1.0) -> list[tuple[Any, int]]:
    """(mean token loss, supervised tokens) of every example of a micro-batch. The decisive decision token
    of a detection example counts `decision_token_weight` times; 1.0 gives the plain mean token loss."""
    import torch
    import torch.nn.functional as F

    out = []
    for row, example in enumerate(batch):
        shifted = labels[row, 1:]
        positions = (shifted != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        ce = F.cross_entropy(logits[row, :-1][positions].float(), shifted[positions], reduction="none")
        weights = torch.ones_like(ce)
        if decision_token_weight != 1.0 and example.decision_index is not None:
            # the token at input position i is predicted by the logits at i - 1
            weights[positions == example.decision_index - 1] = decision_token_weight
        out.append(((weights * ce).sum() / weights.sum(), int(positions.numel())))
    return out


def objective_loss(model: Any, inputs: dict[str, Any], batch: list[EncodedExample], objective: ObjectiveSettings | None) -> Any:
    """The training loss of one micro-batch. Without an objective (or with both weights 1.0) it is the
    model's own mean token loss, exactly as before Training Optimization v1."""
    import torch

    if objective is None or objective.is_default():
        return model(**inputs).loss
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    losses = example_losses(logits, inputs["labels"], batch, objective.decision_token_weight)
    weighted = [loss * (objective.patch_task_weight if e.task == "patch" else 1.0) for (loss, _), e in zip(losses, batch)]
    return torch.stack(weighted).mean()


def decision_summary(rows: list[tuple[str, str, float]]) -> dict[str, Any]:
    """rows: (expected, predicted, finding-minus-no_finding logit margin) of the validation detection examples."""
    positives = [r for r in rows if r[0] == "finding"]
    negatives = [r for r in rows if r[0] == "no_finding"]
    tp = sum(1 for r in positives if r[1] == "finding")
    fp = sum(1 for r in negatives if r[1] == "finding")
    return {
        "examples": len(rows),
        "accuracy": round(sum(1 for r in rows if r[0] == r[1]) / len(rows), 4) if rows else None,
        "recall": round(tp / len(positives), 4) if positives else None,
        "false_positive_rate": round(fp / len(negatives), 4) if negatives else None,
        "finding_rate": round(sum(1 for r in rows if r[1] == "finding") / len(rows), 4) if rows else None,
        "mean_finding_margin": round(sum(r[2] for r in rows) / len(rows), 4) if rows else None,
        "mean_finding_margin_on_findings": round(sum(r[2] for r in positives) / len(positives), 4) if positives else None,
        "mean_finding_margin_on_negatives": round(sum(r[2] for r in negatives) / len(negatives), 4) if negatives else None,
    }


def evaluate_validation(model: Any, examples: list[EncodedExample], micro_batch_size: int, pad_id: int, decision_ids: dict[str, int] | None = None) -> dict[str, Any]:
    """Validation loss (token-weighted over all tasks, as before) plus what it hides: the loss of each task
    and the teacher-forced decisions of the detection examples (argmax of the decision tokens)."""
    import torch

    if not examples:
        return {"validation_loss": None}
    model.eval()
    total, tokens = 0.0, 0
    by_task: dict[str, list[float]] = {}
    decisions: list[tuple[str, str, float]] = []
    with torch.no_grad():
        for batch in batches(examples, micro_batch_size, epoch=0, seed=0):
            inputs = _to_tensors(batch, pad_id, model.device)
            count = int((inputs["labels"][:, 1:] != -100).sum())
            out = model(**inputs)
            total += float(out.loss) * count
            tokens += count
            for (loss, n), example in zip(example_losses(out.logits, inputs["labels"], batch), batch):
                acc = by_task.setdefault(example.task, [0.0, 0])
                acc[0] += float(loss) * n
                acc[1] += n
            if decision_ids and all(d in decision_ids for d in ("finding", "no_finding")):
                for row, example in enumerate(batch):
                    if example.decision is None or example.decision_index is None:
                        continue
                    logits = out.logits[row, example.decision_index - 1].float()
                    scores = {d: float(logits[i]) for d, i in decision_ids.items()}
                    decisions.append((example.decision, max(scores, key=lambda d: scores[d]), scores["finding"] - scores["no_finding"]))
    model.train()
    result: dict[str, Any] = {
        "validation_loss": total / tokens if tokens else None,
        "validation": {"loss_by_task": {task: round(s / n, 5) for task, (s, n) in sorted(by_task.items()) if n}},
    }
    if decisions:
        result["validation"]["decision"] = decision_summary(decisions)
    return result


def evaluate_loss(model: Any, examples: list[EncodedExample], micro_batch_size: int, pad_id: int) -> float | None:
    """The token-weighted validation loss over all tasks (kept for callers that need only this number)."""
    return evaluate_validation(model, examples, micro_batch_size, pad_id)["validation_loss"]


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
    revision_verifier: Callable[[TrainConfig], dict[str, Any]] | None = None,
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
            "tasks": list(data.tasks),
            "examples_by_task": data.examples_by_task,
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
        reset_cuda_peak_memory()  # initializes CUDA and each device's allocator before resetting
        load_started = time.monotonic()
        base, tokenizer = model_loader(config)
        # whatever loaded the model, its files must be the pinned revision (raises FoundationMismatch)
        verification = (revision_verifier or verify_foundation)(config)
        run["foundation"]["revision_verification"] = verification
        run["foundation"]["loaded_revision"] = verification["revision"]
        if verification["revision"] != config.revision:
            raise FoundationMismatch(f"loaded revision {verification['revision']}, expected pinned {config.revision}")
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
        decision_ids = decision_token_ids(train_examples)
        run["data"]["decision_token_ids"] = decision_ids
        if config.objective is not None:
            run["objective"] = {"decision_token_weight": config.objective.decision_token_weight, "patch_task_weight": config.objective.patch_task_weight}
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        model, targets = apply_lora(base, config, config.training.gradient_checkpointing)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run["lora"] = {"target_modules": len(targets), "trainable_parameters": trainable}
        init_adapter = None
        if config.init_adapter is not None and not resume:
            init_adapter = load_init_adapter(model, config, project_root or PROJECT_ROOT)
            run["init_adapter"] = init_adapter
            log(f"initial adapter {init_adapter['checkpoint']} loaded ({init_adapter['tensors']} tensors); new optimizer and schedule")
        elif config.init_adapter is not None:
            init_adapter = check_init_adapter(config, project_root or PROJECT_ROOT)  # resume: weights come from the run's own checkpoint
            run["init_adapter"] = init_adapter

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

        provenance = checkpoint_provenance(config, data, init_adapter)
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
        window_loss = 0.0  # mean loss of the current optimizer step (sum of loss / window)
        window_tokens, window_started = 0, time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        while state.step < total_steps and state.epoch < config.training.num_epochs:
            epoch_batches = batches(train_examples, config.training.micro_batch_size, state.epoch, config.training.seed)
            for index in range(state.batch_in_epoch, len(epoch_batches)):
                inputs = _to_tensors(epoch_batches[index], pad_id, device)
                window = accumulation_window(index, len(epoch_batches), config.training.gradient_accumulation_steps)
                loss = objective_loss(model, inputs, epoch_batches[index], config.objective) / window
                if not math.isfinite(float(loss.detach())):
                    raise RuntimeError(f"non-finite loss at step {state.step + 1} (micro-batch {index} of epoch {state.epoch})")
                loss.backward()
                window_loss += float(loss.detach())
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
                step_loss, window_loss = window_loss, 0.0
                state.step += 1
                # only advanced at an optimizer step, so an interrupt checkpoint never skips unapplied gradients
                state.batch_in_epoch = index + 1
                if state.step % config.training.logging_every_steps == 0:
                    elapsed = time.monotonic() - window_started
                    entry = {
                        "step": state.step,
                        "epoch": state.epoch,
                        "loss": round(step_loss, 5),  # mean over the step's micro-batches, not the last one
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": round(grad_norm, 5),
                        "tokens_per_second": round(window_tokens / elapsed, 2) if elapsed > 0 else None,
                        "peak_vram_gib": _peak_vram_gib(),
                    }
                    state.history.append(entry)
                    log(json.dumps(entry))
                    window_tokens, window_started = 0, time.monotonic()
                if state.step % config.training.eval_every_steps == 0:
                    evaluation = {"step": state.step, **evaluate_validation(model, validation_examples, config.training.micro_batch_size, pad_id, decision_ids)}
                    state.evaluations.append(evaluation)
                    log(json.dumps(evaluation))
                if state.step % config.training.save_every_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, state, output_dir, config.training.keep_last_checkpoints, provenance)
                if state.step >= total_steps:
                    break
            if state.batch_in_epoch >= len(epoch_batches):
                state.epoch += 1
                state.batch_in_epoch = 0

        evaluation = {"step": state.step, **evaluate_validation(model, validation_examples, config.training.micro_batch_size, pad_id, decision_ids)}
        state.evaluations.append(evaluation)
        log(json.dumps(evaluation))
        save_checkpoint(model, optimizer, scheduler, state, output_dir, config.training.keep_last_checkpoints, provenance)
        adapter_keys = save_adapter(model, output_dir / "adapter")
        (output_dir / "adapter" / "training_run.json").write_text(
            json.dumps(
                {
                    "run_name": config.run_name,
                    "base_model": config.model_id,
                    "base_revision": config.revision,
                    "data_sha256": data.sha256,
                    "tasks": list(data.tasks),
                    # evaluate this adapter with the prompt it was trained on (checked by the eval runner)
                    "prompt_version": PROMPT_VERSION,
                    "patch_prompt_version": PATCH_PROMPT_VERSION if "patch" in data.tasks else None,
                },
                indent=2,
            )
            + "\n",
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
