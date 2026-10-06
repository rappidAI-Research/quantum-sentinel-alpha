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
