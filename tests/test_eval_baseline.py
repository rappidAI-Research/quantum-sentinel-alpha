from pathlib import Path

from sentinel.eval.baseline import predict
from sentinel.eval.io import load_jsonl, validate_predictions
from sentinel.eval.metrics import score

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "eval" / "sentinelbench" / "public_seed.jsonl"


def test_engineering_baseline_runs_end_to_end() -> None:
    cases = load_jsonl(SEED)
    predictions = [predict(case) for case in cases]
    validate_predictions(predictions)
    metrics = score(cases, predictions)
    assert metrics.cases == 12
    assert metrics.false_positive_rate == 0.0
    assert metrics.explicit_abstention_recall == 1.0
    assert metrics.precision >= 0.8
    assert metrics.recall >= 0.6
