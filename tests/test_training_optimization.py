"""Training Optimization v1: sample-order reconstruction, decision-aware objective and validation,
decision scores and calibrated decoding (docs/TRAINING_OPTIMIZATION_V1.md)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from sentinel.eval import decision_scores as ds
from sentinel.eval.backends import GenerationSettings
from sentinel.train.config import ConfigError, load_train_config
from sentinel.train.data import EncodedExample, batches, encode_example
from sentinel.train.diagnostics import learning_rate, optimizer_steps, window_mix
from tests.train_fixtures import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def detection_example(record_id: str, decision: str) -> dict[str, Any]:
    reply = {"decision": decision, "evidence": "x"}
    if decision == "finding":
        reply = {"decision": "finding", "cwe": "CWE-79", "start_line": 1, "end_line": 1, "evidence": "x"}
    return {
        "record_id": record_id,
        "task": "detection",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "code"}, {"role": "assistant", "content": json.dumps(reply)}],
    }


# ---------------------------------------------------------------------------------------------
# Order reconstruction and schedule
# ---------------------------------------------------------------------------------------------


def keys(n: int, task: str = "detection") -> list[EncodedExample]:
    return [EncodedExample(record_id=f"r{i:03d}", input_ids=[0], labels=[0], task=task) for i in range(n)]


def test_optimizer_steps_follow_the_trainer_grouping() -> None:
    examples = keys(37)
    steps = optimizer_steps(examples, micro_batch_size=1, accumulation_steps=16, epochs=2, seed=1234)
    # 37 micro-batches per epoch: windows of 16, 16, 5 in each epoch, aligned to the epoch start
    assert [len(s.examples) for s in steps] == [16, 16, 5, 16, 16, 5]
    assert [s.epoch for s in steps] == [0, 0, 0, 1, 1, 1]
    for epoch in (0, 1):
        expected = [(e.task, e.record_id) for batch in batches(examples, 1, epoch, 1234) for e in batch]
        assert [k for s in steps if s.epoch == epoch for k in s.examples] == expected
    assert len(optimizer_steps(examples, 1, 16, 2, 1234, max_steps=4)) == 4


def test_learning_rate_matches_the_transformers_schedule() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    total, warmup_ratio, peak = 346, 0.03, 1e-4
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, int(total * warmup_ratio), total)
    for step in range(1, total + 1):
        applied = optimizer.param_groups[0]["lr"]  # the rate optimizer.step() uses at this optimizer step
        assert math.isclose(learning_rate(step, total, warmup_ratio, peak), applied, rel_tol=1e-9, abs_tol=1e-15)
        optimizer.step()
        scheduler.step()


def test_window_mix_counts_classes() -> None:
    examples = keys(4) + keys(2, task="patch")
    classes = {(e.task, e.record_id): ("patch" if e.task == "patch" else ("finding" if e.record_id < "r002" else "no_finding")) for e in examples}
    steps = optimizer_steps(examples, 1, 3, 1, 7)
    mix = window_mix(steps, classes, 1, 2)
    assert mix["examples"] == 6 and mix["by_class"] == {"finding": 2, "no_finding": 2, "abstain": 0, "patch": 2}


# ---------------------------------------------------------------------------------------------
# Decision position and config
# ---------------------------------------------------------------------------------------------


def test_encoded_detection_examples_know_their_decision_token() -> None:
    tokenizer = CharTokenizer()
    for decision, first_char in (("finding", "f"), ("no_finding", "n"), ("abstain", "a")):
        encoded = encode_example(tokenizer, detection_example("r1", decision), 4096, "<|im_end|>")
        assert encoded is not None and encoded.decision == decision
        assert encoded.input_ids[encoded.decision_index] == tokenizer(first_char)["input_ids"][0]
        assert encoded.labels[encoded.decision_index] == encoded.input_ids[encoded.decision_index]  # supervised
    patch = {**detection_example("r2", "finding"), "task": "patch"}
    patch["messages"][-1]["content"] = "--- a/x\n+++ b/x\n"
    encoded = encode_example(tokenizer, patch, 4096, "<|im_end|>")
    assert encoded is not None and encoded.decision is None and encoded.decision_index is None


def write_config(tmp_path: Path, objective: dict[str, Any] | None, tasks: list[str] | None = None) -> Path:
    import yaml

    document = yaml.safe_load((ROOT / "configs/train/qlora_ornith15_9b_v6.yaml").read_text())
    if objective is not None:
        document["objective"] = objective
    if tasks is not None:
        document["data"]["tasks"] = tasks
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_objective_section_is_optional_and_validated(tmp_path: Path) -> None:
    assert load_train_config(ROOT / "configs/train/qlora_ornith15_9b_v6.yaml").objective is None  # V6 unchanged
    config = load_train_config(write_config(tmp_path, {"decision_token_weight": 10}))
    assert config.objective.decision_token_weight == 10.0 and config.objective.patch_task_weight == 1.0
    assert config.as_dict()["objective"] == {"decision_token_weight": 10.0, "patch_task_weight": 1.0}
    for bad in ({"decision_token_weight": 0}, {"decision_token_weight": -1}, {"decision_token_weight": 1000}, {"unknown": 1}):
        with pytest.raises(ConfigError):
            load_train_config(write_config(tmp_path, bad))
    with pytest.raises(ConfigError, match="patch task"):
        load_train_config(write_config(tmp_path, {"patch_task_weight": 0.5}, tasks=["detection"]))


def test_experiment_configs_change_one_variable_against_v6() -> None:
    """Each experiment config differs from the V6 config only in its variable, the observation settings
    (denser checkpoints and evaluations, which do not change the gradients) and its run name and output."""
    import yaml

    base = yaml.safe_load((ROOT / "configs/train/qlora_ornith15_9b_v6.yaml").read_text())
    observation = {"eval_every_steps", "save_every_steps", "keep_last_checkpoints"}
    expected = {
        "qlora_ornith15_9b_v6_to1_baseline_dense.yaml": set(),
        "qlora_ornith15_9b_v6_to1_decision_w10.yaml": {"objective"},
        "qlora_ornith15_9b_v6_to1_detection_only.yaml": {"data.tasks"},
        "qlora_ornith15_9b_v6_to1_lr5e-5.yaml": {"training.learning_rate"},
    }
    for name, variables in expected.items():
        document = yaml.safe_load((ROOT / "configs/train" / name).read_text())
        changed = set()
        for section in set(base) | set(document):
            if isinstance(base.get(section), dict) and isinstance(document.get(section), dict):
                changed |= {f"{section}.{k}" for k in set(base[section]) | set(document[section]) if base[section].get(k) != document[section].get(k)}
            elif base.get(section) != document.get(section):
                changed.add(section)
        assert changed - {f"training.{k}" for k in observation} - {"run_name", "output_dir"} == variables, name
        assert document["training"]["seed"] == base["training"]["seed"] and "init_adapter" not in document  # fresh adapter, same seed
        load_train_config(ROOT / "configs/train" / name)


# ---------------------------------------------------------------------------------------------
# Objective loss and validation on the tiny model
# ---------------------------------------------------------------------------------------------


def tiny_inputs(decision: str = "finding"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.test_train_smoke import tiny_model

    model = tiny_model("0" * 40)
    tokenizer = CharTokenizer()
    encoded = encode_example(tokenizer, detection_example("r1", decision), 4096, "<|im_end|>")
    from sentinel.train.trainer import _to_tensors

    return torch, model, encoded, _to_tensors([encoded], tokenizer.pad_token_id, model.device)


def test_default_objective_is_the_models_own_loss() -> None:
    from sentinel.train.config import ObjectiveSettings
    from sentinel.train.trainer import example_losses, objective_loss

    torch, model, encoded, inputs = tiny_inputs()
    with torch.no_grad():
        reference = float(model(**inputs).loss)
        assert float(objective_loss(model, inputs, [encoded], None)) == reference
        assert float(objective_loss(model, inputs, [encoded], ObjectiveSettings())) == reference
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        (loss, tokens), = example_losses(logits, inputs["labels"], [encoded])
    assert math.isclose(float(loss), reference, rel_tol=1e-5)
    assert tokens == sum(1 for t in encoded.labels if t != -100) - 0  # every supervised token after the first position


def test_decision_token_weight_reweights_exactly_one_token() -> None:
    import torch.nn.functional as F

    from sentinel.train.config import ObjectiveSettings
    from sentinel.train.trainer import objective_loss

    torch, model, encoded, inputs = tiny_inputs("no_finding")
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[0].float()
        labels = inputs["labels"][0]
        positions = [i for i in range(1, labels.shape[0]) if int(labels[i]) != -100]
        ce = {i: float(F.cross_entropy(logits[i - 1 : i], labels[i : i + 1])) for i in positions}
        w = 10.0
        expected = (sum(ce.values()) + (w - 1) * ce[encoded.decision_index]) / (len(ce) + w - 1)
        weighted = float(objective_loss(model, inputs, [encoded], ObjectiveSettings(decision_token_weight=w)))
    assert math.isclose(weighted, expected, rel_tol=1e-5)


def test_patch_task_weight_scales_patch_examples_only() -> None:
    from sentinel.train.config import ObjectiveSettings
    from sentinel.train.trainer import objective_loss

    torch, model, encoded, inputs = tiny_inputs()
    patch = dataclasses.replace(encoded, task="patch", decision=None, decision_index=None)
    with torch.no_grad():
        plain = float(objective_loss(model, inputs, [patch], ObjectiveSettings(decision_token_weight=2.0)))
        half = float(objective_loss(model, inputs, [patch], ObjectiveSettings(decision_token_weight=2.0, patch_task_weight=0.5)))
        detection = float(objective_loss(model, inputs, [encoded], ObjectiveSettings(decision_token_weight=1.0, patch_task_weight=0.5)))
        reference = float(model(**inputs).loss)
    assert math.isclose(half, plain * 0.5, rel_tol=1e-6)
    assert math.isclose(detection, reference, rel_tol=1e-5)


def test_validation_reports_task_losses_and_decisions() -> None:
    from sentinel.train.trainer import decision_token_ids, evaluate_loss, evaluate_validation

    torch, model, _, _ = tiny_inputs()
    tokenizer = CharTokenizer()
    examples = [encode_example(tokenizer, detection_example(f"r{i}", d), 4096, "<|im_end|>") for i, d in enumerate(("finding", "no_finding", "abstain", "finding"))]
    ids = decision_token_ids(examples)
    assert set(ids) == {"finding", "no_finding", "abstain"}
    result = evaluate_validation(model, examples, 1, tokenizer.pad_token_id, ids)
    assert result["validation_loss"] == pytest.approx(evaluate_loss(model, examples, 1, tokenizer.pad_token_id))
    assert set(result["validation"]["loss_by_task"]) == {"detection"}
    decision = result["validation"]["decision"]
    assert decision["examples"] == 4 and 0.0 <= decision["recall"] <= 1.0 and decision["false_positive_rate"] in (0.0, 1.0)


def test_training_run_with_a_decision_weighted_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    import sentinel.train.data as train_data
    from sentinel.train import trainer
    from sentinel.train.config import ObjectiveSettings
    from sentinel.train.data import load_training_data
    from tests.test_train_smoke import TINY_TARGETS, loader
    from tests.train_fixtures import GardenWorkdir, approved_registry

    monkeypatch.setattr(trainer, "verify_foundation", lambda config: {"revision": config.revision, "method": "test", "files_sha256": {}})
    garden = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    try:
        data = load_training_data(garden.write().config_path, approvals=garden.approvals)
        base = load_train_config()
        config = dataclasses.replace(
            base,
            quantization=dataclasses.replace(base.quantization, mode="none", compute_dtype="float32"),
            lora=dataclasses.replace(base.lora, expected_target_count=TINY_TARGETS),
            training=dataclasses.replace(
                base.training, micro_batch_size=1, gradient_accumulation_steps=2, num_epochs=50, max_steps=2, eval_every_steps=1,
                save_every_steps=2, keep_last_checkpoints=1, gradient_checkpointing=False, learning_rate=1e-3,
            ),
            output_dir=tmp_path / "run",
            objective=ObjectiveSettings(decision_token_weight=10.0),
        )
        run = trainer.train(config, data, model_loader=loader, log=lambda _: None)
    finally:
        garden.cleanup()
    assert run["status"] == "completed" and run["objective"] == {"decision_token_weight": 10.0, "patch_task_weight": 1.0}
    assert set(run["data"]["decision_token_ids"]) <= {"finding", "no_finding", "abstain"}
    evaluations = run["state"]["evaluations"]
    assert evaluations and all("loss_by_task" in e["validation"] for e in evaluations)


# ---------------------------------------------------------------------------------------------
# Decision scores (CPU)
# ---------------------------------------------------------------------------------------------


def case(case_id: str, expected: str, pair: str | None = None, language: str = "python") -> dict[str, Any]:
    c = {"case_id": case_id, "expected_decision": expected, "language": language, "ground_truth": {"cwe": "CWE-79" if expected == "finding" else None}}
    if pair:
        c["pair_id"] = pair
    return c


def scorer_from(table: dict[str, tuple[float, float, float]]):
    def score(c: dict[str, Any]) -> dict[str, Any]:
        f, n, a = table[c["case_id"]]
        logprob = {"finding": f, "no_finding": n, "abstain": a}
        return {"logprob": logprob, "greedy_token": max(logprob, key=logprob.get), "prefix_greedy": True}

    return score


def test_decision_token_ids_follow_the_reply_prefix() -> None:
    ids = ds.decision_token_ids(CharTokenizer())
    tokenizer = CharTokenizer()
    assert ids == {"finding": tokenizer("f")["input_ids"][0], "no_finding": tokenizer("n")["input_ids"][0], "abstain": tokenizer("a")["input_ids"][0]}


def test_scores_separate_ranking_from_threshold() -> None:
    cases = [case("v1", "finding", "p1"), case("f1", "no_finding", "p1"), case("v2", "finding", "p2"), case("f2", "no_finding", "p2"), case("a1", "abstain")]
    # the model ranks every vulnerable case above its fixed twin, but its threshold calls everything no_finding
    table = {"v1": (-2.0, -1.0, -5.0), "f1": (-3.0, -0.5, -5.0), "v2": (-1.5, -1.2, -5.0), "f2": (-4.0, -0.2, -5.0), "a1": (-6.0, -2.0, -0.5)}
    rows = ds.score_rows(cases, scorer_from(table))
    summary = ds.summarize(rows)
    assert summary["greedy"]["recall"] == 0.0 and summary["greedy"]["false_positive_rate"] == 0.0
    assert summary["scores"]["auc_finding_vs_no_finding"] == 1.0
    assert summary["pairs"]["both_missed"] == 2 and summary["pairs"]["score_margin_vulnerable_minus_fixed"]["share_positive"] == 1.0
    best = summary["scores"]["best_bias_at_fpr_0.2"]
    assert best["recall"] == 1.0 and best["false_positive_rate"] == 0.0 and best["bias"] > 0
    assert ds.with_bias(rows[0], best["bias"]) == "finding" and ds.with_bias(rows[1], best["bias"]) == "no_finding"
    estimate = ds.cross_fitted_bias(rows, 0.2)
    assert estimate["held_out"]["cases"] == len(rows)


def test_operating_points_cover_the_greedy_point() -> None:
    rows = ds.score_rows([case("v1", "finding"), case("f1", "no_finding")], scorer_from({"v1": (-1.0, -2.0, -9.0), "f1": (-1.0, -3.0, -9.0)}))
    points = {p["bias"]: p for p in ds.operating_points(rows)}
    assert points[0.0]["recall"] == 1.0 and points[0.0]["false_positive_rate"] == 1.0


def test_flips_detect_pairs_moving_together(tmp_path: Path) -> None:
    cases = [case("v1", "finding", "p1"), case("f1", "no_finding", "p1"), case("v2", "finding", "p2"), case("f2", "no_finding", "p2")]
    before = ds.score_rows(cases, scorer_from({"v1": (-1, -2, -9), "f1": (-1, -2, -9), "v2": (-1, -2, -9), "f2": (-3, -1, -9)}))
    after = ds.score_rows(cases, scorer_from({"v1": (-3, -1, -9), "f1": (-3, -1, -9), "v2": (-1, -2, -9), "f2": (-3, -1, -9)}))
    result = ds.flips(before, after)
    assert result["changed"] == 2 and result["pairs_flipping_together"] == 1
    assert result["by_direction"] == {"finding: finding->no_finding": 1, "no_finding: finding->no_finding": 1}


def test_analyze_reads_score_files_and_run_artifacts(tmp_path: Path) -> None:
    cases = [case("v1", "finding", "p1"), case("f1", "no_finding", "p1")]
    rows = ds.score_rows(cases, scorer_from({"v1": (-1, -2, -9), "f1": (-3, -1, -9)}))
    scores = tmp_path / "s.json"
    scores.write_text(json.dumps({"model": {"adapter": {"path": "runs/train/qsa-x/checkpoints/step-000150/adapter"}}, "rows": rows}))
    run = tmp_path / "r.json"
    run.write_text(json.dumps({
        "model": {"adapter": {"path": "runs/train/qsa-x/checkpoints/step-000175/adapter"}},
        "predictions": [
            {"case_id": "v1", "expected_decision": "finding", "prediction": {"decision": "no_finding"}},
            {"case_id": "f1", "expected_decision": "no_finding", "prediction": {"decision": "no_finding"}},
        ],
    }))
    result = ds.analyze([run, scores], cases)
    assert [c["label"] for c in result["checkpoints"]] == ["qsa-x/step-000150", "qsa-x/step-000175"]
    assert result["checkpoints"][0]["pairs"]["discriminated"] == 1 and "scores" not in result["checkpoints"][1]
    assert result["flips_between_consecutive"][0]["changed"] == 1
    with pytest.raises(ValueError, match="not made on these cases"):
        ds.analyze([scores], cases + [case("x", "finding")])


# ---------------------------------------------------------------------------------------------
# Calibrated decoding setting
# ---------------------------------------------------------------------------------------------


def test_finding_bias_is_recorded_only_when_used() -> None:
    assert "finding_logit_bias" not in GenerationSettings().as_dict()  # earlier runs stay comparable
    assert GenerationSettings(finding_logit_bias=1.5).as_dict()["finding_logit_bias"] == 1.5
    for bad in (float("nan"), 50.0):
        with pytest.raises(ValueError):
            GenerationSettings(finding_logit_bias=bad)
    with pytest.raises(ValueError, match="greedy"):
        GenerationSettings(do_sample=True, finding_logit_bias=1.0)


class FakeBackend:
    """The attributes HFDecisionScorer and HFBackend._biased_decision_prefix read from a loaded backend."""

    def __init__(self, bias: float = 0.0) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from tests.test_train_smoke import tiny_model

        self._model = tiny_model("0" * 40).eval()
        self._tokenizer = CharTokenizer()
        self.generation = GenerationSettings(finding_logit_bias=bias)


def eval_case() -> dict[str, Any]:
    return {"case_id": "c1", "expected_decision": "finding", "language": "python", "file": "a.py", "context": "", "code": "x = 1\n", "ground_truth": {"cwe": "CWE-79"}}


def test_hf_decision_scorer_reads_the_three_decision_tokens() -> None:
    import torch

    backend = FakeBackend()
    scored = ds.HFDecisionScorer(backend)(eval_case())
    assert set(scored["logprob"]) == {"finding", "no_finding", "abstain"}
    assert all(v <= 0.0 for v in scored["logprob"].values()) and isinstance(scored["prefix_greedy"], bool)
    # the same numbers as a direct forward pass over prompt + prefix
    from sentinel.eval.prompt import PROMPT_VERSION, build_messages
    from sentinel.train.data import DECISION_PREFIX

    tokenizer = backend._tokenizer
    prompt = tokenizer(tokenizer.apply_chat_template(build_messages(eval_case(), PROMPT_VERSION), add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt + tokenizer(DECISION_PREFIX)["input_ids"]])
    with torch.no_grad():
        logprobs = torch.log_softmax(backend._model(input_ids=ids).logits[0, -1].float(), dim=-1)
    finding = ds.decision_token_ids(tokenizer)["finding"]
    assert scored["logprob"]["finding"] == pytest.approx(float(logprobs[finding]), abs=1e-5)


def test_biased_decoding_forces_the_decision_only_when_greedy_would_start_a_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from sentinel.eval.backends import HFBackend

    fake = FakeBackend(bias=5.0)
    backend = HFBackend.__new__(HFBackend)
    backend._model, backend._tokenizer, backend.generation = fake._model, fake._tokenizer, fake.generation
    prompt = torch.tensor([[5, 6, 7]])
    # a random tiny model does not produce the reply prefix greedily: nothing is forced
    assert backend._biased_decision_prefix(prompt) == []

    # a model that always continues the prefix and prefers no_finding by 1 nat: a bias of 5 flips it to finding
    from sentinel.train.data import DECISION_PREFIX

    prefix = fake._tokenizer(DECISION_PREFIX)["input_ids"]
    ids = ds.decision_token_ids(fake._tokenizer)

    class Scripted(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids):
            n = input_ids.shape[1]
            logits = torch.full((1, n, 256), -10.0)
            start = n - len(prefix) - 1
            for k, token in enumerate(prefix):
                logits[0, start + k, token] = 10.0
            logits[0, -1, ids["no_finding"]] = 2.0
            logits[0, -1, ids["finding"]] = 1.0
            return type("Out", (), {"logits": logits})()

    backend._model = Scripted()
    assert backend._biased_decision_prefix(prompt) == prefix + [ids["finding"]]
    backend.generation = GenerationSettings(finding_logit_bias=0.5)
    assert backend._biased_decision_prefix(prompt) == prefix + [ids["no_finding"]]


def test_calibration_uses_the_validation_split_and_never_protected_sets() -> None:
    record = {
        "record_id": "sg-python-vulnerable-0001",
        "split": "validation",
        "evaluation_role": "retired_development",
        "pair_id": "pair-1",
        "input": {"language": "python", "file": "app.py", "context": "Function `f`", "code": "def f(x):\n    return eval(x)\n"},
        "target": {"decision": "finding", "cwe": "CWE-95", "start_line": 2, "end_line": 2, "alternate_cwes": []},
        "provenance": {
            "code_source_id": "ghsa_osv",
            "repository": "example/app",
            "excerpt_revision": "0" * 40,
            "file_path": "app.py",
            "license": {"detected": "MIT", "status": "reviewed"},
        },
    }
    cases = ds.calibration_cases([record])
    assert cases[0]["source_kind"] == "calibration" and cases[0]["case_id"] == record["record_id"]
    for role in ("development", "final_test", "previous_holdout", "external_benchmark"):
        with pytest.raises(ValueError, match="validation"):
            ds.calibration_cases([{**record, "evaluation_role": role}])
    with pytest.raises(ValueError, match="validation"):
        ds.calibration_cases([{**record, "split": "train"}])
    with pytest.raises(ValueError, match="final-test"):
        ds.refuse_protected_cases([{"case_id": "x", "source_kind": "private_holdout"}])
    ds.refuse_protected_cases([{"case_id": "x", "source_kind": "development"}])


def test_score_outputs_name_each_checkpoint_and_refuse_collisions(tmp_path: Path) -> None:
    single = ds.score_outputs(tmp_path / "base.json", [None])
    assert single == [tmp_path / "base.json"]
    run = "runs/train/qsa-qlora-ornith15-9b-v6"
    paths = ds.score_outputs(tmp_path, [Path(f"{run}/checkpoints/step-000150"), Path(f"{run}/checkpoints/step-000175")])
    assert [p.name for p in paths] == ["qsa-qlora-ornith15-9b-v6__step-000150.json", "qsa-qlora-ornith15-9b-v6__step-000175.json"]
    with pytest.raises(ValueError, match="same output"):
        ds.score_outputs(tmp_path, [Path(f"{run}/checkpoints/step-000150"), Path(f"{run}/checkpoints/step-000150")])


# --- calibrated decoding: the bias is chosen on the training validation split and carried with provenance ---

CALIBRATION = {
    "path": "runs/eval/calibration/step-000150.json",
    "sha256": "a" * 64,
    "calibration_version": "1",
    "bias": 2.0,
    "max_fpr": 0.15,
    "cases": {"path": "artifacts/garden/seed_v6/validation.jsonl", "sha256": "b" * 64, "count": 146, "kind": "calibration"},
}


def _score_document(kind: str) -> dict[str, Any]:
    rows = []
    for i in range(20):  # finding scores rise with i; positives are the upper half, with overlap
        expected = "finding" if i % 2 else "no_finding"
        rows.append({"case_id": f"c{i}", "expected": expected, "pair_id": f"p{i // 2}", "language": "python", "cwe": None,
                     "logprob": {"finding": -3.0 + i * 0.2 + (0.8 if expected == "finding" else 0.0), "no_finding": -0.5, "abstain": -6.0}})
    for row in rows:
        row["decision"] = ds.with_bias(row, 0.0)
    model = {"id": "ornith-ai/Ornith-1.5-9B", "revision": "c" * 40, "quantization": "none", "dtype": "bfloat16", "adapter": {"path": "x/adapter", "adapter_model_sha256": "d" * 64}}
    return {"model": model, "cases": {"path": "v.jsonl", "sha256": "e" * 64, "count": len(rows), "kind": kind}, "rows": rows}


def test_calibration_is_chosen_on_the_calibration_set_and_bound_to_its_model() -> None:
    with pytest.raises(ValueError, match="calibration set"):
        ds.calibration_document(_score_document("development"), 0.15)
    document = ds.calibration_document(_score_document("calibration"), 0.15)
    assert document["bias"] is not None and document["chosen"]["false_positive_rate"] <= 0.15
    setup = {"model_id": "ornith-ai/Ornith-1.5-9B", "revision": "c" * 40, "quantization": "none", "dtype": "bfloat16", "adapter_weights_sha256": "d" * 64}
    assert ds.check_calibration(document, **setup) == document["bias"]
    for key, other in (("adapter_weights_sha256", "f" * 64), ("quantization", "bnb-4bit"), ("revision", "0" * 40)):
        with pytest.raises(ValueError, match=key.split("_")[0]):
            ds.check_calibration(document, **{**setup, key: other})


def test_run_model_loads_a_calibration_only_for_the_adapter_it_was_made_for(tmp_path: Path) -> None:
    from sentinel.eval.backends import RuntimeSettings
    from sentinel.eval.run_model import load_calibration

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    document = ds.calibration_document(_score_document("calibration"), 0.15)
    document["model"]["adapter_weights_sha256"] = ds.adapter_digest(adapter)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    provenance, bias = load_calibration(path, "ornith-ai/Ornith-1.5-9B", "c" * 40, RuntimeSettings(), adapter)
    assert bias == document["bias"] == provenance["bias"] and provenance["cases"]["kind"] == "calibration"
    (adapter / "adapter_model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="adapter_weights_sha256"):
        load_calibration(path, "ornith-ai/Ornith-1.5-9B", "c" * 40, RuntimeSettings(), adapter)


def _adapter_artifact(monkeypatch: pytest.MonkeyPatch, step: int, bias: float = 0.0, calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    from sentinel.eval import runner
    from sentinel.eval.backends import RuntimeSettings
    from tests.test_adapter_dtypes import VERIFIED, AdapterBackend
    from tests.test_model_runner import CASES, PINNED

    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "f" * 40, "dirty": False})
    backend = AdapterBackend({"foundation_parameter_dtypes": {"bfloat16": 100}, "adapter_verification": {**VERIFIED, "ok": True}})
    backend.answers, backend.default = {}, '{"decision": "no_finding"}'
    backend.adapter_info = {**backend.adapter_info, "path": f"runs/train/r/checkpoints/step-{step:06d}/adapter"}
    generation = GenerationSettings(finding_logit_bias=bias) if bias else GenerationSettings()
    return runner.run_quickeval(CASES, backend, model_id="Qwen/Qwen3.5-4B", revision=PINNED, generation=generation, runtime=RuntimeSettings(),
                                cases_path="eval/sentinelbench/quickeval_v1.jsonl", calibration=calibration)


def test_a_biased_run_is_reportable_only_with_calibration_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    calibrated = _adapter_artifact(monkeypatch, 150, 2.0, CALIBRATION)  # schema-valid with the bias and its provenance
    assert calibrated["generation"]["finding_logit_bias"] == 2.0 and calibrated["calibration"] == CALIBRATION
    assert not any("finding_logit_bias" in b for b in calibrated["reportable_blockers"])
    hand_picked = _adapter_artifact(monkeypatch, 150, 2.0)
    assert any("finding_logit_bias" in b for b in hand_picked["reportable_blockers"]) and not hand_picked["reportable"]
    mismatched = _adapter_artifact(monkeypatch, 150, 3.0, CALIBRATION)
    assert not mismatched["reportable"]


def test_selection_accepts_per_checkpoint_calibrated_biases_with_provenance_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from sentinel.eval.io import case_set_sha256
    from sentinel.eval.select_checkpoint import Refused, select
    from tests.test_model_runner import CASES

    devset = {"detection": {"cases_sha256": case_set_sha256(CASES), "count": len(CASES), "by_expected_decision": {}}}
    greedy = _adapter_artifact(monkeypatch, 150)
    calibrated = _adapter_artifact(monkeypatch, 150, 2.0, CALIBRATION)
    other = _adapter_artifact(monkeypatch, 175, -1.5, {**CALIBRATION, "bias": -1.5})
    result = select(devset, [greedy, calibrated, other])
    by_label = {r["label"]: r for r in result["runs"]}
    assert set(by_label) == {"step-000150", "step-000150+calibrated", "step-000175+calibrated"}
    assert by_label["step-000150+calibrated"]["finding_logit_bias"] == 2.0 and "selected_finding_logit_bias" in result
    assert by_label["step-000150+calibrated"]["calibration"] == CALIBRATION["path"] and "selected_calibration" in result
    with pytest.raises(Refused, match="no calibration"):
        select(devset, [greedy, {**calibrated, "calibration": None}])
    with pytest.raises(Refused, match="generation"):
        select(devset, [greedy, {**calibrated, "generation": {**calibrated["generation"], "max_new_tokens": 7}}])


def test_screen_shortlists_greedy_and_calibrated_checkpoints_by_the_fixed_rule() -> None:
    dev = _score_document("development")
    dev["model"]["adapter"]["path"] = "runs/train/r/checkpoints/step-000150/adapter"
    calibration = ds.calibration_document(_score_document("calibration"), 0.15)
    result = ds.screen([(Path("dev/r__step-000150.json"), dev)], [(Path("cal/r__step-000150.json"), calibration)])
    entry = result["checkpoints"][0]
    assert entry["label"] == "r/step-000150" and entry["calibrated"]["bias"] == calibration["bias"]
    for c in result["shortlist"]:
        assert c["recall"] >= ds.SCREEN_RULE["min_recall"] and c["false_positive_rate"] <= ds.SCREEN_RULE["max_false_positive_rate"]
        assert (c["mode"] == "calibrated") == (c["calibration"] is not None)
    other = {**calibration, "model": {**calibration["model"], "adapter_weights_sha256": "0" * 64}}
    assert "calibrated" not in ds.screen([(Path("d.json"), dev)], [(Path("c.json"), other)])["checkpoints"][0]
    with pytest.raises(ValueError, match="development"):
        ds.screen([(Path("c.json"), _score_document("calibration"))], [])


def test_diagnostics_read_every_step_from_the_training_log() -> None:
    from sentinel.train.diagnostics import run_from_log, run_history

    lines = ["loading model", *(json.dumps({"step": s, "epoch": 0, "loss": 0.5, "learning_rate": 1e-4, "grad_norm": 1.5 if s == 150 else 0.4}) for s in range(1, 201))]
    lines.append(json.dumps({"step": 150, "validation_loss": 0.39678}))
    history = run_history(run_from_log("\n".join(lines)), 126, 175, 1.0)
    assert history["logged_steps"] == 50 and history["clipped_steps"] == 1
    assert history["validation_loss"] == [{"step": 150, "validation_loss": 0.39678}]


def test_gpu_pipeline_script_never_scores_or_calibrates_on_test_sets() -> None:
    """scripts/to1_gpu.sh (runbook § 16) in DRY_RUN: every stage in order, and the final test and holdout
    appear only in the final stage, never in decision scores, calibration, screening or selection."""
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    result = subprocess.run(["bash", str(ROOT / "scripts/to1_gpu.sh"), "all"], env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
                            capture_output=True, text=True, timeout=60, check=True)
    lines = result.stderr.splitlines()
    commands = [line for line in lines if line.startswith("+ ")]
    final_stage = next(i for i, line in enumerate(commands) if "final_test and holdout_reserved" in line)
    for line in commands[:final_stage]:
        assert "final_test" not in line and "holdout" not in line, line
    trained = [line.split("--config ")[1].split()[0] for line in commands if "sentinel.train --config" in line and "--dry-run" not in line]
    assert trained == [f"configs/train/qlora_ornith15_9b_v6_to1_{e}.yaml" for e in ("baseline_dense", "decision_w10", "detection_only", "lr5e-5")]
    calibration_sources = [line.split("--records ")[1].split()[0] for line in commands if "--records " in line]
    assert calibration_sources and set(calibration_sources) == {"artifacts/garden/seed_v6/validation.jsonl"}
    assert any("--quantization bnb-4bit" in line for line in commands) and any("select_checkpoint --rule 2" in line for line in commands)
    for config in trained:
        assert (ROOT / config).is_file()
