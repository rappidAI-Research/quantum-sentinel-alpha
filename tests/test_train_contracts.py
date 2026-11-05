"""Training contracts that run without torch: config, target modules, governance gate, tokenization."""

import json
from pathlib import Path

import pytest
import yaml

import sentinel.train.data as train_data
from sentinel.foundation_model import load_foundation
from sentinel.train.__main__ import main as train_main
from sentinel.train.config import DEFAULT_TRAIN_CONFIG, PROJECT_ROOT, ConfigError, load_train_config, matching_modules
from sentinel.train.data import IGNORE_INDEX, GovernanceError, batches, encode_all, encode_example, load_training_data, pad_batch
from tests.train_fixtures import CharTokenizer, GardenWorkdir, approved_registry

MODULES = PROJECT_ROOT / "configs" / "train" / "qwen35_9b_linear_modules.txt"


def module_table() -> dict[str, tuple[int, int]]:
    table = {}
    for line in MODULES.read_text().splitlines():
        if line and not line.startswith("#"):
            name, fan_in, fan_out = line.split()
            table[name] = (int(fan_in), int(fan_out))
    return table


# --- configuration --------------------------------------------------------------------------------


def test_config_targets_the_pinned_foundation() -> None:
    config = load_train_config()
    foundation = load_foundation()
    assert (config.model_id, config.revision) == (foundation.model_id, foundation.revision)
    assert config.quantization.mode == "nf4" and config.quantization.compute_dtype == "bfloat16"


def test_lora_targets_match_the_inspected_architecture() -> None:
    config = load_train_config()
    table = module_table()
    targets = matching_modules(config.lora.target_modules, list(table))
    assert len(targets) == config.lora.expected_target_count == 200
    kinds = {t.rsplit(".", 1)[1] for t in targets}
    assert kinds == {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"}
    assert not any(t.endswith(("in_proj_a", "in_proj_b")) or t == "lm_head" for t in targets)
    # rank-16 LoRA parameters this configuration trains (documented in docs/TRAINING.md)
    trainable = sum(config.lora.rank * (table[t][0] + table[t][1]) for t in targets)
    # 16 x (8 x 30720 attention + 24 x 28672 Gated DeltaNet + 32 x 49152 MLP)
    assert trainable == 16 * (8 * 30720 + 24 * 28672 + 32 * 49152) == 40_108_032


def test_config_validation(tmp_path: Path) -> None:
    document = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text())
    for mutate, message in (
        (lambda d: d["quantization"].update(mode="int3"), "quantization.mode"),
        (lambda d: d.update(output_dir="/tmp/out"), "output_dir"),
        (lambda d: d["lora"].update(target_modules="("), "regex"),
        (lambda d: d["training"].update(micro_batch_size=0), "positive"),
        (lambda d: d["data"].update(max_seq_len=16), "max_seq_len"),
        (lambda d: d["lora"].pop("rank"), "exactly"),
    ):
        broken = yaml.safe_load(yaml.safe_dump(document))
        mutate(broken)
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(ConfigError, match=message):
            load_train_config(path)


# --- governance-gated data ------------------------------------------------------------------------


@pytest.fixture
def garden(monkeypatch: pytest.MonkeyPatch):
    workdir = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    yield workdir
    workdir.cleanup()


def test_approved_records_load(garden: GardenWorkdir) -> None:
    data = load_training_data(garden.write().config_path, approvals=garden.approvals)
    assert len(data.train) == len(garden.train) and len(data.validation) == len(garden.validation)
    assert data.repositories == 2 and len(data.sha256) == 64
    assert data == load_training_data(garden.config_path, approvals=garden.approvals)  # deterministic


def test_empty_training_export_fails_closed(garden: GardenWorkdir) -> None:
    garden.train = []
    with pytest.raises(GovernanceError, match="no training-approved records"):
        load_training_data(garden.write().config_path, approvals=garden.approvals)


def test_revoked_approval_stops_training(garden: GardenWorkdir) -> None:
    with pytest.raises(GovernanceError, match="no longer passes the governance gate"):
        load_training_data(garden.write().config_path, approvals={})


def test_unapproved_record_in_export_is_refused(garden: GardenWorkdir) -> None:
    garden.train[0]["governance"] = {"approval_status": "pending_review", "blockers": ["repository_not_reviewed"]}
    with pytest.raises(GovernanceError, match="not training_approved|approved train records"):
        load_training_data(garden.write().config_path, approvals=garden.approvals)


def test_holdout_record_cannot_enter_training(garden: GardenWorkdir) -> None:
    garden.train[0]["split"] = "holdout_reserved"
    with pytest.raises(GovernanceError):
        load_training_data(garden.write().config_path, approvals=garden.approvals)


def test_record_from_holdout_repository_is_refused(garden: GardenWorkdir) -> None:
    holdout = garden.copy_record(garden.train[0], record_id="sg-py-vulnerable-0000000000000000", split="holdout_reserved")
    from sentinel.garden.pipeline import manifest_row

    garden.extra_manifest.append(manifest_row(holdout))
    with pytest.raises(GovernanceError, match="holdout/quarantine repository"):
        load_training_data(garden.write().config_path, approvals=garden.approvals)


def test_tampered_chat_file_is_refused(garden: GardenWorkdir) -> None:
    garden.write()
    path = garden.workdir / "train.chat.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["messages"][2]["content"] = '{"decision": "no_finding", "evidence": "x"}'
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(GovernanceError, match="not the rendering"):
        load_training_data(garden.config_path, approvals=garden.approvals)


def test_code_must_match_manifest_hash(garden: GardenWorkdir) -> None:
    garden.write()
    path = garden.workdir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["input"]["code"] += "# injected\n"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(GovernanceError, match="content hash"):
        load_training_data(garden.config_path, approvals=garden.approvals)


def test_repository_in_train_and_validation_is_refused(garden: GardenWorkdir) -> None:
    for record in garden.validation:
        record["provenance"]["repository_key"] = "example/webapp"
    with pytest.raises(GovernanceError):
        load_training_data(garden.write().config_path, approvals=garden.approvals)


def test_cli_dry_run_refuses_without_approved_data() -> None:
    # The committed approvals are empty, so the real configuration must refuse (exit code 2).
    assert train_main(["--dry-run"]) == 2


# --- tokenization ---------------------------------------------------------------------------------


def example() -> dict:
    return {
        "record_id": "r1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "code"},
            {"role": "assistant", "content": '{"decision": "abstain", "evidence": "x"}'},
        ],
    }


def test_loss_only_on_reply_and_end_of_turn() -> None:
    tokenizer = CharTokenizer()
    encoded = encode_example(tokenizer, example(), 4096, "<|im_end|>")
    prompt = tokenizer(tokenizer.apply_chat_template(example()["messages"][:2], add_generation_prompt=True))["input_ids"]
    assert encoded.input_ids[: len(prompt)] == prompt
    assert encoded.labels[: len(prompt)] == [IGNORE_INDEX] * len(prompt)
    supervised = [t for t in encoded.labels if t != IGNORE_INDEX]
    assert supervised[-1] == 2 and len(supervised) == len('{"decision": "abstain", "evidence": "x"}') + 1


def test_overlength_examples_are_dropped_not_truncated() -> None:
    kept, dropped = encode_all(CharTokenizer(), [example()], 20, "<|im_end|>")
    assert kept == [] and dropped == ["r1"]


def test_end_of_turn_must_be_one_token() -> None:
    with pytest.raises(ValueError, match="single token"):
        encode_all(CharTokenizer(), [example()], 4096, "</s>")


def test_batches_are_deterministic_and_padded() -> None:
    tokenizer = CharTokenizer()
    examples = [encode_example(tokenizer, {**example(), "record_id": f"r{i}"}, 4096, "<|im_end|>") for i in range(5)]
    first = batches(examples, 2, epoch=0, seed=7)
    assert first == batches(list(reversed(examples)), 2, epoch=0, seed=7)
    assert first != batches(examples, 2, epoch=1, seed=7)
    padded = pad_batch([examples[0], examples[1]], pad_id=0)
    assert len({len(row) for row in padded["input_ids"]}) == 1


# --- adapter evaluation -----------------------------------------------------------------------------


def test_adapter_must_match_base_model_and_revision(tmp_path: Path) -> None:
    from sentinel.eval.backends import describe_adapter

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    config = load_train_config()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": config.model_id, "revision": config.revision}))
    info = describe_adapter(adapter, config.model_id, config.revision)
    assert info["base_revision"] == config.revision and len(info["weights_sha256"]) == 64
    with pytest.raises(ValueError, match="revision"):
        describe_adapter(adapter, config.model_id, "0" * 40)
    with pytest.raises(ValueError, match="trained for"):
        describe_adapter(adapter, "Qwen/Qwen3.5-4B", config.revision)


def test_run_model_refuses_mismatched_adapter(tmp_path: Path) -> None:
    from sentinel.eval.run_model import main as run_model_main

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B", "revision": "1" * 40}))
    with pytest.raises(SystemExit) as excinfo:
        run_model_main(
            ["--bakeoff", "configs/model/bakeoff_qwen35.yaml", "--candidate", "qwen35-9b", "--adapter", str(adapter), "--output", str(tmp_path / "r.json")]
        )
    assert excinfo.value.code == 2  # argparse error before any model download


def test_baseline_artifact_still_validates_without_adapter(tmp_path: Path) -> None:
    from sentinel.eval.run_model import main as run_model_main
    from sentinel.eval.runner import load_artifact

    out = tmp_path / "baseline.json"
    assert run_model_main(["--backend", "baseline", "--cases", "eval/sentinelbench/public_seed.jsonl", "--output", str(out)]) == 0
    assert "adapter" not in load_artifact(out)["model"]


def test_label_must_match_the_manifest(garden: GardenWorkdir) -> None:
    garden.write()
    path = garden.workdir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    vulnerable = next(r for r in rows if r["target"]["decision"] == "finding")
    vulnerable["target"].update(decision="no_finding", cwe=None, alternate_cwes=[], start_line=None, end_line=None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # a consistently re-rendered chat file must not launder the changed label
    from sentinel.garden import render

    (garden.workdir / "train.chat.jsonl").write_text("".join(json.dumps(render.to_chat_example(r)) + "\n" for r in rows))
    with pytest.raises(GovernanceError, match="does not match the manifest"):
        load_training_data(garden.config_path, approvals=garden.approvals)


def test_incomplete_checkpoints_are_never_resumed(tmp_path: Path) -> None:
    from sentinel.train.trainer import _checkpoints

    root = tmp_path / "checkpoints"
    for name, complete in (("step-000010", True), ("step-000020", True), ("step-000030.tmp", True), ("step-000040", False)):
        (root / name).mkdir(parents=True)
        if complete:
            (root / name / "state.json").write_text("{}")
    assert [p.name for p in _checkpoints(tmp_path)] == ["step-000010", "step-000020"]


def test_resume_requires_the_same_data_and_config(tmp_path: Path) -> None:
    from sentinel.train.trainer import check_resume_provenance

    expected = {"base_model": "m", "base_revision": "r", "config_sha256": "c", "data_sha256": "d", "record_ids_sha256": "i"}
    with pytest.raises(RuntimeError, match="no provenance.json"):
        check_resume_provenance(tmp_path, expected)
    (tmp_path / "provenance.json").write_text(json.dumps({**expected, "data_sha256": "other"}))
    with pytest.raises(RuntimeError, match="different data_sha256"):
        check_resume_provenance(tmp_path, expected)
    (tmp_path / "provenance.json").write_text(json.dumps(expected))
    check_resume_provenance(tmp_path, expected)


def test_partial_accumulation_window_is_averaged_over_its_own_size() -> None:
    from sentinel.train.trainer import accumulation_window

    sizes = [accumulation_window(i, 17, 16) for i in range(17)]
    assert sizes == [16] * 16 + [1]
    assert [accumulation_window(i, 10, 4) for i in range(10)] == [4] * 8 + [2, 2]
    # every optimizer step averages its micro-batch losses with weights summing to one
    for n, ga in ((17, 16), (10, 4), (8, 4), (3, 8)):
        steps: dict[int, float] = {}
        for i in range(n):
            steps[i // ga] = steps.get(i // ga, 0.0) + 1 / accumulation_window(i, n, ga)
        assert all(abs(total - 1.0) < 1e-12 for total in steps.values())
