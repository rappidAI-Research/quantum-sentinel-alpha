from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.eval.io import load_jsonl, validate_cases, validate_predictions
from sentinel.eval.metrics import score


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Quantum Sentinel QuickEval predictions.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    predictions = load_jsonl(args.predictions)
    validate_cases(cases)
    validate_predictions(predictions)
    metrics = score(cases, predictions).as_dict()

    if args.as_json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
