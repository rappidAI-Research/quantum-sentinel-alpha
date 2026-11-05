"""CPU smoke test of the real training loop on a tiny, randomly initialised Qwen3.5 text model.

Skipped unless torch, transformers and peft are installed (`pip install -e '.[train]'` plus a CPU
torch build). Uses the same model class and module layout as Qwen/Qwen3.5-9B (Gated DeltaNet +
full-attention layers), so the LoRA target regex, checkpointing, resume and adapter-only output
are exercised exactly as on the GPU - only smaller. No weights are downloaded.
"""

import dataclasses
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import load_file  # noqa: E402

import sentinel.train.data as train_data  # noqa: E402
from sentinel.train.config import load_train_config  # noqa: E402
from sentinel.train.data import load_training_data  # noqa: E402
from sentinel.train.trainer import FoundationMismatch, TargetModuleMismatch, train  # noqa: E402
from tests.train_fixtures import CharTokenizer, GardenWorkdir, approved_registry  # noqa: E402

# 3 linear-attention + 1 full-attention layer: 3 x 3 + 1 x 4 attention/DeltaNet + 4 x 3 MLP targets
TINY_TARGETS = 25


def tiny_model(revision: str):
    from transformers import AutoModelForCausalLM, Qwen3_5Config

    text = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "vocab_size": 256,
        "max_position_embeddings": 4096,
        "mtp_num_hidden_layers": 0,
    }
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(Qwen3_5Config(text_config=text))
    model.config._commit_hash = revision
    return model


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    garden = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    data = load_training_data(garden.write().config_path, approvals=garden.approvals)
    base = load_train_config()
    config = dataclasses.replace(
        base,
        quantization=dataclasses.replace(base.quantization, mode="none", compute_dtype="float32"),
        lora=dataclasses.replace(base.lora, expected_target_count=TINY_TARGETS),
        training=dataclasses.replace(
            base.training,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            num_epochs=50,
            max_steps=4,
            eval_every_steps=2,
            save_every_steps=2,
            keep_last_checkpoints=2,
            gradient_checkpointing=False,
            learning_rate=1e-3,
        ),
        output_dir=tmp_path / "run",
    )
    yield config, data
    garden.cleanup()


def loader(config):
    return tiny_model(config.revision), CharTokenizer()


def test_train_checkpoint_resume_and_adapter_only_output(setup) -> None:
    config, data = setup
    run = train(config, data, model_loader=loader, log=lambda _: None)
    assert run["status"] == "completed"
    assert run["lora"]["target_modules"] == TINY_TARGETS
    assert run["foundation"]["loaded_revision"] == config.revision
    assert run["state"]["step"] == 4 and run["data"]["train_dropped_overlength"] == []
    assert all(e["validation_loss"] is not None for e in run["state"]["evaluations"])

    adapter = config.output_dir / "adapter"
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    assert tensors and all("lora_" in key for key in tensors)
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    assert adapter_config["revision"] == config.revision
    assert not list(adapter.glob("model*.safetensors"))  # no base weights written

    checkpoints = sorted(p.name for p in (config.output_dir / "checkpoints").iterdir())
    assert checkpoints == ["step-000002", "step-000004"]
    stored = json.loads((config.output_dir / "run.json").read_text())
    assert stored["status"] == "completed" and stored["data"]["sha256"] == data.sha256

    resumed_config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=6))
    resumed = train(resumed_config, data, resume=True, model_loader=loader, log=lambda _: None)
    assert resumed["status"] == "completed" and resumed["resumed_from"].endswith("step-000004")
    assert resumed["state"]["step"] == 6


def test_wrong_foundation_revision_is_refused(setup) -> None:
    config, data = setup

    def wrong(cfg):
        model = tiny_model("0" * 40)
        return model, CharTokenizer()

    with pytest.raises(FoundationMismatch):
        train(config, data, model_loader=wrong, log=lambda _: None)
    assert json.loads((config.output_dir / "run.json").read_text())["status"] == "failed"


def test_target_module_drift_is_refused(setup) -> None:
    config, data = setup
    config = dataclasses.replace(config, lora=dataclasses.replace(config.lora, expected_target_count=200))
    with pytest.raises(TargetModuleMismatch):
        train(config, data, model_loader=loader, log=lambda _: None)


def test_out_of_memory_is_recorded_with_a_hint(setup) -> None:
    config, data = setup

    def oom(cfg):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        train(config, data, model_loader=oom, log=lambda _: None)
    run = json.loads((config.output_dir / "run.json").read_text())
    assert run["status"] == "failed_oom" and "micro_batch_size" in run["hint"]


def test_tampered_checkpoint_is_refused_and_pickles_never_execute(setup, tmp_path) -> None:
    config, data = setup
    train(config, data, model_loader=loader, log=lambda _: None)
    latest = sorted((config.output_dir / "checkpoints").iterdir())[-1]
    resumed_config = dataclasses.replace(config, training=dataclasses.replace(config.training, max_steps=6))

    # 1. any modified file is detected before anything is loaded
    state_file = latest / "state.json"
    original = state_file.read_text()
    state_file.write_text(original.replace('"step": 4', '"step": 1'))
    with pytest.raises(RuntimeError, match="modified after it was written"):
        train(resumed_config, data, resume=True, model_loader=loader, log=lambda _: None)
    state_file.write_text(original)

    # 2. an attacker who also rewrites checksums.json still cannot run code through the pickle
    marker = tmp_path / "pwned"

    class Planted:
        def __reduce__(self):
            return (Path.touch, (marker,))

    torch.save({"optimizer": Planted()}, latest / "trainer_state.pt")
    from sentinel.train.trainer import CHECKSUMS, _file_digests

    (latest / CHECKSUMS).write_text(json.dumps(_file_digests(latest)))
    with pytest.raises(Exception, match="(?i)weights_only|unpickl|unsupported"):
        train(resumed_config, data, resume=True, model_loader=loader, log=lambda _: None)
    assert not marker.exists()


def test_logged_loss_is_the_mean_of_the_optimizer_step(setup) -> None:
    from sentinel.train.data import batches, encode_all
    from sentinel.train.trainer import _to_tensors, apply_lora

    config, data = setup
    config = dataclasses.replace(
        config,
        lora=dataclasses.replace(config.lora, dropout=0.0),
        training=dataclasses.replace(config.training, max_steps=1, logging_every_steps=1, learning_rate=0.0, warmup_ratio=0.0),
    )
    run = train(config, data, model_loader=loader, log=lambda _: None)
    logged = run["state"]["history"][0]["loss"]

    # the same two micro-batches through an identical, unchanged model (learning rate 0)
    model, tokenizer = loader(config)
    model, _ = apply_lora(model, config, gradient_checkpointing=False)
    examples, _ = encode_all(tokenizer, data.train, config.data.max_seq_len, config.data.end_of_turn)
    first_step = batches(examples, 1, epoch=0, seed=config.training.seed)[:2]
    with torch.no_grad():
        losses = [float(model(**_to_tensors(b, tokenizer.pad_token_id, model.device)).loss) for b in first_step]
    assert losses[0] != pytest.approx(losses[1])
    assert logged == pytest.approx(sum(losses) / 2, abs=1e-4)


def test_patch_task_trains_and_is_recorded(setup, monkeypatch) -> None:
    config, _ = setup
    garden = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    try:
        data = load_training_data(garden.write().config_path, approvals=garden.approvals, tasks=("detection", "patch"))
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, tasks=("detection", "patch")))
        run = train(config, data, model_loader=loader, log=lambda _: None)
        assert run["status"] == "completed" and run["data"]["tasks"] == ["detection", "patch"]
        assert run["data"]["examples_by_task"]["train"]["patch"] == 1
        record = json.loads((config.output_dir / "adapter" / "training_run.json").read_text())
        assert record["tasks"] == ["detection", "patch"] and record["prompt_version"] == "quickeval-review-v2"
        assert record["patch_prompt_version"] == "sentinel-patch-v1"
    finally:
        garden.cleanup()
