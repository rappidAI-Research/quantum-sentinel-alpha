from pathlib import Path

import pytest

from sentinel.eval.io import load_jsonl, validate_cases, validate_predictions
from sentinel.eval.metrics import score

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "eval" / "sentinelbench" / "public_seed.jsonl"


def test_public_seed_contract_is_valid_and_balanced() -> None:
    cases = load_jsonl(SEED)
    validate_cases(cases)
    decisions = [case["expected_decision"] for case in cases]
    assert len(cases) == 12
    assert decisions.count("finding") == 5
    assert decisions.count("no_finding") == 5
    assert decisions.count("abstain") == 2
    assert {case["language"] for case in cases} == {"python", "javascript", "go"}


def test_metrics_reward_precision_localization_cwe_and_abstention() -> None:
    cases = [
        {
            "case_id": "positive",
            "expected_decision": "finding",
            "ground_truth": {"cwe": "CWE-79", "start_line": 10, "end_line": 12},
        },
        {"case_id": "negative", "expected_decision": "no_finding", "ground_truth": {}},
        {"case_id": "uncertain", "expected_decision": "abstain", "ground_truth": {}},
    ]
    predictions = [
        {
            "case_id": "positive",
            "decision": "finding",
            "finding": {"cwe": "CWE-79", "start_line": 11, "end_line": 11},
            "patch_verification": None,
        },
        {"case_id": "negative", "decision": "no_finding", "finding": None, "patch_verification": None},
        {"case_id": "uncertain", "decision": "abstain", "finding": None, "patch_verification": None},
    ]
    metrics = score(cases, predictions)
    assert metrics.precision == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.recall == 1.0
    assert metrics.localization_top1 == 1.0
    assert metrics.localization_mean_overlap == pytest.approx(1 / 3)
    assert metrics.cwe_accuracy == 1.0
    assert metrics.explicit_abstention_recall == 1.0


def test_prediction_set_must_cover_exact_case_ids() -> None:
    cases = [{"case_id": "a", "expected_decision": "no_finding", "ground_truth": {}}]
    with pytest.raises(ValueError, match="missing predictions"):
        score(cases, [])


def test_nonfinding_prediction_cannot_smuggle_finding_payload() -> None:
    with pytest.raises(ValueError, match="must not carry"):
        validate_predictions(
            [
                {
                    "schema_version": "1.0",
                    "case_id": "case-a",
                    "decision": "no_finding",
                    "finding": {"cwe": "CWE-79", "start_line": 1, "end_line": 1, "confidence": 0.9},
                    "patch_verification": None,
                }
            ]
        )


def _pred(case_id: str, decision: str, finding: dict | None = None) -> dict:
    return {"case_id": case_id, "decision": decision, "finding": finding, "patch_verification": None}


def test_hard_negative_findings_are_false_positives_and_break_pairs() -> None:
    cases = [
        {"case_id": "vuln", "expected_decision": "finding", "pair_id": "p",
         "ground_truth": {"cwe": "CWE-89", "start_line": 3, "end_line": 3}},
        {"case_id": "fixed", "expected_decision": "no_finding", "pair_id": "p", "ground_truth": {}},
    ]
    flag_everything = [
        _pred("vuln", "finding", {"cwe": "CWE-89", "start_line": 3, "end_line": 3}),
        _pred("fixed", "finding", {"cwe": "CWE-89", "start_line": 3, "end_line": 3}),
    ]
    metrics = score(cases, flag_everything)
    assert metrics.precision == 0.5
    assert metrics.false_positive_rate == 1.0
    assert metrics.hard_negative_accuracy == 0.0
    assert metrics.pair_accuracy == 0.0


def test_localization_and_cwe_scoring_accept_alternates_only() -> None:
    cases = [
        {"case_id": "a", "expected_decision": "finding",
         "ground_truth": {"cwe": "CWE-78", "alternate_cwes": ["CWE-77"], "start_line": 5, "end_line": 6}},
        {"case_id": "b", "expected_decision": "finding",
         "ground_truth": {"cwe": "CWE-22", "start_line": 2, "end_line": 2}},
    ]
    predictions = [
        _pred("a", "finding", {"cwe": "CWE-77", "start_line": 6, "end_line": 9}),
        _pred("b", "finding", {"cwe": "CWE-73", "start_line": 4, "end_line": 4}),
    ]
    metrics = score(cases, predictions)
    assert metrics.recall == 1.0
    assert metrics.cwe_accuracy == 0.5
    assert metrics.localization_top1 == 0.5
    assert metrics.localization_mean_overlap == pytest.approx((1 / 5) / 2)
    assert metrics.strict_recall == 0.5


def test_abstention_and_invalid_outputs_are_scored_as_failures_where_wrong() -> None:
    cases = [
        {"case_id": "vuln", "expected_decision": "finding",
         "ground_truth": {"cwe": "CWE-79", "start_line": 1, "end_line": 1}},
        {"case_id": "safe", "expected_decision": "no_finding", "ground_truth": {}},
        {"case_id": "unknown-1", "expected_decision": "abstain", "ground_truth": {}},
        {"case_id": "unknown-2", "expected_decision": "abstain", "ground_truth": {}},
    ]
    predictions = [
        _pred("vuln", "invalid"),
        _pred("safe", "abstain"),
        _pred("unknown-1", "abstain"),
        _pred("unknown-2", "finding", {"cwe": "CWE-79", "start_line": 1, "end_line": 1}),
    ]
    metrics = score(cases, predictions)
    assert metrics.false_negative == 1 and metrics.true_positive == 0
    assert metrics.false_positive == 0
    assert metrics.hard_negative_accuracy == 0.0
    assert metrics.invalid_rate == 0.25
    assert metrics.explicit_abstention_recall == 0.5
    assert metrics.abstention_precision == 0.5
    assert metrics.over_abstention_rate == 0.5
    assert metrics.abstention_case_finding_rate == 0.5
