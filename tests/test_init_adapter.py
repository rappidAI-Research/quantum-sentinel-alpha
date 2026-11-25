"""Continuing from a checkpoint adapter (`init_adapter`): pure checks (no torch needed).

The owner decided on 2026-11-25 to train only fresh adapters; no committed config uses `init_adapter`.

The end-to-end continued run on a tiny model is in tests/test_train_smoke.py."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sentinel.train.config import (
    PROJECT_ROOT,
    ConfigError,
    InitAdapterSettings,
    load_train_config,
)
from sentinel.train.trainer import (
    CHECKSUMS,
    InitAdapterMismatch,
    _file_digests,
    check_init_adapter,
)


def write_checkpoint(root: Path, config, step: int = 200, **overrides) -> Path:
    path = root / "runs" / "train" / "src" / "checkpoints" / f"step-{step:06d}"
    adapter = path / "adapter"
    adapter.mkdir(parents=True)
    adapter_config = {
        "base_model_name_or_path": config.model_id,
        "revision": config.revision,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "target_modules": config.lora.target_modules,
        **overrides,
    }
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    provenance = {"base_model": config.model_id, "base_revision": config.revision, "config_sha256": "c" * 64, "data_sha256": "d" * 64}
    (path / "provenance.json").write_text(json.dumps(provenance))
    (path / "state.json").write_text(json.dumps({"step": step}))
    (path / "trainer_state.pt").write_bytes(b"optimizer state that is never loaded")
    (path / CHECKSUMS).write_text(json.dumps(_file_digests(path)))
    return path


def configured(root: Path, step: int = 200):
    base = load_train_config()
    return dataclasses.replace(base, init_adapter=InitAdapterSettings(f"runs/train/src/checkpoints/step-{step:06d}", step, "c" * 64, "d" * 64))


def test_a_checkpoint_of_the_source_run_is_accepted(tmp_path: Path) -> None:
    config = configured(tmp_path)
    write_checkpoint(tmp_path, config)
    info = check_init_adapter(config, tmp_path)
    assert info["step"] == 200 and info["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert info["source_run"]["data_sha256"] == "d" * 64


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda p: (p / "adapter" / "adapter_model.safetensors").write_bytes(b"other"), "modified"),
        (lambda p: (p / "provenance.json").write_text(json.dumps({"config_sha256": "e" * 64})), "modified"),
        (lambda p: (p / "extra.bin").write_bytes(b"x"), "modified"),
    ],
)
def test_a_modified_checkpoint_is_refused(tmp_path: Path, damage, message: str) -> None:
    config = configured(tmp_path)
    damage(write_checkpoint(tmp_path, config))
    with pytest.raises(InitAdapterMismatch, match=message):
        check_init_adapter(config, tmp_path)


def test_another_run_step_or_lora_shape_is_refused(tmp_path: Path) -> None:
    config = configured(tmp_path)
    write_checkpoint(tmp_path, config)
    with pytest.raises(InitAdapterMismatch, match="source run"):
        check_init_adapter(dataclasses.replace(config, init_adapter=dataclasses.replace(config.init_adapter, source_config_sha256="0" * 64)), tmp_path)
    with pytest.raises(InitAdapterMismatch, match="does not exist"):
        check_init_adapter(configured(tmp_path, step=150), tmp_path)
    other = tmp_path / "other"
    write_checkpoint(other, config, r=8)
    with pytest.raises(InitAdapterMismatch, match="LoRA settings differ from this config: r"):
        check_init_adapter(config, other)


def test_init_adapter_config_validation(tmp_path: Path) -> None:
    document = yaml.safe_load((PROJECT_ROOT / "configs" / "train" / "qlora_ornith15_9b.yaml").read_text())
    for bad in (
        {"checkpoint": "/abs/step-000200", "step": 200, "source_config_sha256": "c" * 64, "source_data_sha256": "d" * 64},
        {"checkpoint": "runs/r/checkpoints/step-000200", "step": 150, "source_config_sha256": "c" * 64, "source_data_sha256": "d" * 64},
        {"checkpoint": "runs/r/checkpoints/step-000200", "step": 200, "source_config_sha256": "short", "source_data_sha256": "d" * 64},
        {"checkpoint": "runs/r/checkpoints/step-000200", "step": 200},
    ):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({**document, "init_adapter": bad}))
        with pytest.raises(ConfigError):
            load_train_config(path)
