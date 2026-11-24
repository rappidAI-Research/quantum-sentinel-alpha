"""Development set and checkpoint selection: separated from training and from the final holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sentinel.train.data as train_data
from sentinel.eval.devset import build_devset
from sentinel.eval.io import case_set_sha256
from sentinel.eval.select_checkpoint import Refused, select
from tests.test_adapter_dtypes import VERIFIED, AdapterBackend
from tests.test_model_runner import CASES, FakeBackend, run
from tests.train_fixtures import GardenWorkdir, approved_registry

FINDING = '{"decision": "finding", "cwe": "CWE-89", "start_line": 1, "end_line": 2, "evidence": "Request data reaches the query."}'
DEVSET = {"detection": {"cases_sha256": case_set_sha256(CASES), "count": len(CASES), "by_expected_decision": {}}, "patch": {"cases_sha256": "0" * 64}}


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    from sentinel.eval import runner

    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "f" * 40, "dirty": False})


def adapter_run(path: str, default: str, answers: dict[str, str] | None = None, verified: bool = True) -> dict:
    extra = {"foundation_parameter_dtypes": {"bfloat16": 100}, "adapter_verification": {**VERIFIED, "ok": verified}}
    backend = AdapterBackend(extra)
    backend.answers, backend.default = answers or {}, default
    backend.adapter_info = {**backend.adapter_info, "path": path}
    return run(backend)


def foundation_run(default: str) -> dict:
    return run(FakeBackend({}, default=default))


def test_the_rule_selects_on_decision_accuracy_and_never_the_foundation() -> None:
    always_finding = adapter_run("runs/train/r/checkpoints/step-000234/adapter", FINDING)
    never_finding = adapter_run("runs/train/r/checkpoints/step-000150/adapter", '{"decision": "no_finding"}')
    result = select(DEVSET, [foundation_run('{"decision": "no_finding"}'), always_finding, never_finding])
    accuracy = {r["label"]: r["metrics"]["decision_accuracy"] for r in result["runs"]}
    best = max(("step-000150", "step-000234"), key=lambda name: accuracy[name])
    assert result["selected"] == best
    assert next(r for r in result["runs"] if r["label"] == "foundation")["eligible"] is False
    flagging = next(r for r in result["runs"] if r["label"] == "step-000234")
    assert flagging["false_positives"] and flagging["counts"]["false_positive"] == len(flagging["false_positives"])
    assert flagging["findings_on_abstention_cases"]
    if result["selected"] != "step-000234":  # compared case by case with the latest weights
        assert result["discordant_cases_vs_final"]["final"] == "step-000234"


def test_ties_prefer_fewer_false_positives_then_the_earlier_step() -> None:
    early = adapter_run("runs/train/r/checkpoints/step-000150/adapter", '{"decision": "no_finding"}')
    late = adapter_run("runs/train/r/checkpoints/step-000200/adapter", '{"decision": "no_finding"}')
    assert select(DEVSET, [late, early])["selected"] == "step-000150"


def test_unreportable_or_unverified_runs_are_not_eligible() -> None:
    unverified = adapter_run("runs/train/r/checkpoints/step-000150/adapter", '{"decision": "no_finding"}', verified=False)
    result = select(DEVSET, [unverified])
    assert result["selected"] is None and result["runs"][0]["reportable"] is False


def test_holdout_or_quickeval_runs_are_refused() -> None:
    other_set = {**DEVSET, "detection": {**DEVSET["detection"], "cases_sha256": "1" * 64}}
    with pytest.raises(Refused, match="not the development set"):
        select(other_set, [adapter_run("runs/train/r/adapter", FINDING)])


def test_runs_with_different_contracts_or_duplicates_are_refused() -> None:
    a = adapter_run("runs/train/r/checkpoints/step-000150/adapter", FINDING)
    b = adapter_run("runs/train/r/checkpoints/step-000200/adapter", FINDING)
    b["generation"] = {**b["generation"], "max_new_tokens": 999}
    with pytest.raises(Refused, match="generation"):
        select(DEVSET, [a, b])
    with pytest.raises(Refused, match="two runs"):
        select(DEVSET, [a, a])


@pytest.fixture
def garden(monkeypatch: pytest.MonkeyPatch):
    workdir = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    yield workdir
    workdir.cleanup()


def test_devset_is_the_verified_approved_validation_split(garden: GardenWorkdir) -> None:
    config = garden.write().config_path
    cases, patch_cases, manifest = build_devset(config, approvals=garden.approvals)
    validation_ids = {r["record_id"] for r in garden.validation}
    train_ids = {r["record_id"] for r in garden.train}
    assert {c["case_id"] for c in cases} == validation_ids and not validation_ids & train_ids
    assert all(c["source_kind"] == "development" for c in cases)
    assert manifest["detection"]["cases_sha256"] == case_set_sha256(cases)
    assert manifest["patch"]["count"] == len(patch_cases)
    assert build_devset(config, approvals=garden.approvals)[2] == manifest  # deterministic


def test_devset_refuses_tampered_exports(garden: GardenWorkdir) -> None:
    config = garden.write().config_path
    target = garden.workdir / "validation.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    rows[0]["input"]["code"] += "\n# edited"
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(train_data.GovernanceError):
        build_devset(config, approvals=garden.approvals)


def test_a_modified_checkpoint_adapter_is_refused(tmp_path: Path) -> None:
    import hashlib

    from sentinel.eval.backends import describe_adapter

    checkpoint = tmp_path / "checkpoints" / "step-000150"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "m/x", "revision": "1" * 40}))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    digests = {f"adapter/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()}
    (checkpoint / "checksums.json").write_text(json.dumps(digests))
    assert describe_adapter(adapter, "m/x", "1" * 40)["weights_sha256"] == digests["adapter/adapter_model.safetensors"]
    (adapter / "adapter_model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="does not match"):
        describe_adapter(adapter, "m/x", "1" * 40)
