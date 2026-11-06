"""No invented confidence supervision; versioned prompt/parser stay compatible; calibration metrics."""

from __future__ import annotations

import json

import pytest

from sentinel.eval import calibration, prompt
from sentinel.eval.parse import PARSER_VERSION, parse_model_output
from sentinel.garden import render
from tests.train_fixtures import _records

CASE = {"case_id": "c1", "code": "a\nb\nc\n"}


def finding(**extra):
    return json.dumps({"decision": "finding", "cwe": "CWE-89", "start_line": 2, "end_line": 2, "evidence": "x flows into SQL.", **extra})


def test_v1_prompt_is_frozen_and_v2_drops_confidence() -> None:
    # pinned: any artifact made with quickeval-review-v1 stays reproducible
    assert prompt.prompt_sha256("quickeval-review-v1") == "b08907afdc05142e882672be4f8f748c8e9fdff8e373505a94495dcfa4108095"
    assert prompt.PROMPT_VERSION == "quickeval-review-v2"
    assert '"confidence"' in prompt.system_prompt("quickeval-review-v1")
    assert '"confidence"' not in prompt.system_prompt("quickeval-review-v2")
    assert prompt.prompt_sha256("quickeval-review-v1") != prompt.prompt_sha256()
    with pytest.raises(ValueError, match="unknown prompt version"):
        prompt.build_messages({"language": "python", "file": "a.py", "code": "x"}, "quickeval-review-v9")


def test_parser_v3_accepts_findings_with_or_without_confidence() -> None:
    assert PARSER_VERSION == "3"
    without = parse_model_output(finding(), CASE).prediction
    assert without["decision"] == "finding" and without["finding"]["confidence"] is None
    explicit_null = parse_model_output(finding(confidence=None), CASE).prediction
    assert explicit_null["finding"]["confidence"] is None
    v1_style = parse_model_output(finding(confidence=0.8), CASE).prediction
    assert v1_style["finding"] == {"cwe": "CWE-89", "start_line": 2, "end_line": 2, "confidence": 0.8}
    for bad in (1.5, -0.1, "high", True):
        assert parse_model_output(finding(confidence=bad), CASE).prediction["decision"] == "invalid"
    assert parse_model_output(finding(severity="high"), CASE).prediction["decision"] == "invalid"


def test_training_replies_carry_no_confidence() -> None:
    for record in _records("example/webapp", "GHSA-2222-3333-4444", "train"):
        example = render.to_chat_example(record)
        reply = json.loads(example["messages"][-1]["content"])
        assert "confidence" not in reply
        assert example["prompt_version"] == "quickeval-review-v2" and example["render_version"] == "detection-v2"
        assert example["messages"][0]["content"] == prompt.system_prompt("quickeval-review-v2")
        assert render.reply_matches_contract(record)


def test_curate_rejects_a_reply_that_would_teach_a_confidence(monkeypatch) -> None:
    record = next(r for r in _records("example/webapp", "GHSA-2222-3333-4444", "train") if r["kind"] == "vulnerable")
    original = render.assistant_reply
    monkeypatch.setattr(render, "assistant_reply", lambda r: original(r)[:-1] + ', "confidence": 0.9}')
    assert render.reply_matches_contract(record) is False


def test_run_records_the_prompt_and_v1_stays_runnable(tmp_path) -> None:
    from sentinel.eval import run_model

    for version in ("quickeval-review-v1", None):
        out = tmp_path / f"{version}.json"
        args = ["--backend", "baseline", "--cases", "eval/sentinelbench/quickeval_v1.jsonl", "--output", str(out)]
        assert run_model.main(args + (["--prompt-version", version] if version else [])) == 0
        contract = json.loads(out.read_text())["contract"]
        assert contract["prompt_version"] == (version or prompt.PROMPT_VERSION)
        assert contract["prompt_sha256"] == prompt.prompt_sha256(contract["prompt_version"])
        assert contract["parser_version"] == "3"


def test_adapter_is_evaluated_with_its_training_prompt() -> None:
    from sentinel.eval.backends import BaselineBackend, GenerationSettings, RuntimeSettings
    from sentinel.eval.io import load_jsonl
    from sentinel.eval.runner import run_quickeval

    backend = BaselineBackend()
    backend.adapter_info = {
        "path": "runs/train/x/adapter",
        "weights_sha256": "0" * 64,
        "config_sha256": "1" * 64,
        "base_revision": None,
        "training_run": {"prompt_version": "quickeval-review-v2"},
    }
    from sentinel.eval.io import PROJECT_ROOT

    cases = load_jsonl(PROJECT_ROOT / "eval/sentinelbench/quickeval_v1.jsonl")[:3]
    kwargs = dict(model_id="baseline/regex", revision=None, generation=GenerationSettings(), runtime=RuntimeSettings(), cases_path="x", full_case_set=False)
    with pytest.raises(ValueError, match="trained with prompt"):
        run_quickeval(cases, backend, prompt_version="quickeval-review-v1", **kwargs)
    assert run_quickeval(cases, backend, prompt_version="quickeval-review-v2", **kwargs)["contract"]["prompt_version"] == "quickeval-review-v2"


def test_calibration_metrics() -> None:
    dist = calibration.decision_distribution({"finding": 0.0, "no_finding": 0.0, "abstain": 0.0})
    assert dist == pytest.approx({"finding": 1 / 3, "no_finding": 1 / 3, "abstain": 1 / 3})
    assert calibration.decision_distribution({"finding": -0.1, "no_finding": -5.0, "abstain": -9.0})["finding"] > 0.98
    with pytest.raises(ValueError):
        calibration.decision_distribution({"finding": 0.0, "no_finding": 0.0})
    assert calibration.decision_prefixes()["finding"] == '{"decision": "finding"'

    perfect = [(1.0, True), (0.0, False)]
    assert calibration.brier_score(perfect) == 0.0 and calibration.expected_calibration_error(perfect) == 0.0
    # every finding at the old constant 0.9 while half are wrong: badly calibrated
    constant = [(0.9, True), (0.9, False)] * 10
    assert calibration.brier_score(constant) == pytest.approx(0.41)
    assert calibration.expected_calibration_error(constant) == pytest.approx(0.4)
    report = calibration.calibration_report(constant)
    assert report["cases"] == 20 and report["positives"] == 10 and sum(b["count"] for b in report["bins"]) == 20
    assert calibration.brier_score([]) is None and calibration.expected_calibration_error([]) is None
    with pytest.raises(ValueError, match="measured confidence"):
        calibration.brier_score([(None, True)])
