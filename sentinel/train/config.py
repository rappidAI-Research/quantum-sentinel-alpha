"""Training configuration: validated, hashed, and bound to the selected foundation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinel.foundation_model import load_foundation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train" / "qlora_qwen35_9b.yaml"
TOP_KEYS = {"schema_version", "run_name", "foundation", "data", "quantization", "lora", "training", "output_dir"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DataSettings:
    garden_config: str
    max_seq_len: int
    end_of_turn: str


@dataclass(frozen=True)
class QuantizationSettings:
    mode: str
    double_quant: bool
    compute_dtype: str


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: str
    expected_target_count: int


@dataclass(frozen=True)
class TrainingSettings:
    seed: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    num_epochs: int
    max_steps: int | None
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    gradient_checkpointing: bool
    eval_every_steps: int
    save_every_steps: int
    keep_last_checkpoints: int
    logging_every_steps: int


@dataclass(frozen=True)
class TrainConfig:
    path: Path
    sha256: str
    run_name: str
    model_id: str
    revision: str
    data: DataSettings
    quantization: QuantizationSettings
    lora: LoraSettings
    training: TrainingSettings
    output_dir: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "model_id": self.model_id,
            "revision": self.revision,
            "data": asdict(self.data),
            "quantization": asdict(self.quantization),
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "output_dir": str(self.output_dir),
        }


def _section(document: dict[str, Any], name: str, cls: type) -> Any:
    raw = document[name]
    expected = set(cls.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ConfigError(f"{name} must define exactly {sorted(expected)}")
    return cls(**raw)


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    data = path.read_bytes()
    document = yaml.safe_load(data)
    if not isinstance(document, dict) or set(document) != TOP_KEYS:
        raise ConfigError(f"training config must define exactly {sorted(TOP_KEYS)}")
    if document["schema_version"] != "1.0":
        raise ConfigError("unsupported training config schema_version")
    foundation = load_foundation(PROJECT_ROOT / document["foundation"])

    settings = _section(document, "data", DataSettings)
    quantization = _section(document, "quantization", QuantizationSettings)
    lora = _section(document, "lora", LoraSettings)
    training = _section(document, "training", TrainingSettings)

    if quantization.mode not in {"nf4", "none"}:
        raise ConfigError("quantization.mode must be nf4 or none")
    if quantization.compute_dtype not in {"bfloat16", "float16", "float32"}:
        raise ConfigError("quantization.compute_dtype must be bfloat16, float16 or float32")
    if settings.max_seq_len < 256:
        raise ConfigError("data.max_seq_len must be at least 256")
    try:
        re.compile(lora.target_modules)
    except re.error as exc:
        raise ConfigError(f"lora.target_modules is not a valid regex: {exc}") from exc
    if lora.rank <= 0 or lora.alpha <= 0 or not 0.0 <= lora.dropout < 1.0 or lora.expected_target_count <= 0:
        raise ConfigError("lora.rank/alpha/expected_target_count must be positive and dropout in [0, 1)")
    positive = ("learning_rate", "num_epochs", "micro_batch_size", "gradient_accumulation_steps", "eval_every_steps", "save_every_steps", "keep_last_checkpoints", "logging_every_steps")
    if any(getattr(training, name) <= 0 for name in positive):
        raise ConfigError(f"training.{'/'.join(positive)} must be positive")
    if not 0.0 <= training.warmup_ratio < 1.0 or training.max_grad_norm <= 0:
        raise ConfigError("training.warmup_ratio must be in [0, 1) and max_grad_norm positive")
    if training.max_steps is not None and training.max_steps <= 0:
        raise ConfigError("training.max_steps must be null or positive")
    output_dir = PROJECT_ROOT / document["output_dir"]
    if not str(document["output_dir"]).startswith(("runs/", "artifacts/")):
        raise ConfigError("output_dir must be under the Git-ignored runs/ or artifacts/ directory")

    return TrainConfig(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        run_name=document["run_name"],
        model_id=foundation.model_id,
        revision=foundation.revision,
        data=settings,
        quantization=quantization,
        lora=lora,
        training=training,
        output_dir=output_dir,
    )


def matching_modules(pattern: str, names: list[str]) -> list[str]:
    regex = re.compile(pattern)
    return [name for name in names if regex.fullmatch(name)]
