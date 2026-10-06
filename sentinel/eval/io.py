from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "eval_case.schema.json"
PREDICTION_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "eval_prediction.schema.json"


def _load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def validate_cases(cases: list[dict[str, Any]]) -> None:
    validator = Draft202012Validator(_load_schema(CASE_SCHEMA_PATH))
    seen: set[str] = set()
    for case in cases:
        validator.validate(case)
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)

        truth = case["ground_truth"]
        start, end = truth["start_line"], truth["end_line"]
        if case["expected_decision"] == "finding":
            if not truth["cwe"] or start is None or end is None:
                raise ValueError(f"finding case requires CWE and localization: {case_id}")
            if end < start:
                raise ValueError(f"invalid ground-truth span: {case_id}")
        elif any(value is not None for value in truth.values()):
            raise ValueError(f"non-finding case must not carry vulnerability ground truth: {case_id}")


def validate_predictions(predictions: list[dict[str, Any]]) -> None:
    validator = Draft202012Validator(_load_schema(PREDICTION_SCHEMA_PATH))
    seen: set[str] = set()
    for prediction in predictions:
        validator.validate(prediction)
        case_id = prediction["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate prediction case_id: {case_id}")
        seen.add(case_id)
        finding = prediction["finding"]
        if prediction["decision"] == "finding":
            if finding is None:
                raise ValueError(f"finding decision requires finding payload: {case_id}")
            if finding["end_line"] < finding["start_line"]:
                raise ValueError(f"invalid predicted span: {case_id}")
        elif finding is not None:
            raise ValueError(f"non-finding decision must not carry finding payload: {case_id}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
