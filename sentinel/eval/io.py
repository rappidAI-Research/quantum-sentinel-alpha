from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "eval_case.schema.json"
PREDICTION_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "eval_prediction.schema.json"

CASE_TYPE_DECISION = {"vulnerable": "finding", "hard_negative": "no_finding", "abstention": "abstain"}


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
    pairs: dict[str, list[str]] = {}
    for case in cases:
        validator.validate(case)
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)

        expected = case["expected_decision"]
        case_type = case.get("case_type")
        if case_type is not None and CASE_TYPE_DECISION[case_type] != expected:
            raise ValueError(f"case_type {case_type} contradicts expected_decision {expected}: {case_id}")

        truth = case["ground_truth"]
        start, end = truth["start_line"], truth["end_line"]
        if expected == "finding":
            if not truth["cwe"] or start is None or end is None:
                raise ValueError(f"finding case requires CWE and localization: {case_id}")
            if end < start or end > len(case["code"].splitlines()):
                raise ValueError(f"invalid ground-truth span: {case_id}")
            if truth["cwe"] in truth.get("alternate_cwes", []):
                raise ValueError(f"alternate_cwes must not repeat the primary CWE: {case_id}")
        elif any(truth.get(key) is not None for key in ("cwe", "start_line", "end_line")) or "alternate_cwes" in truth:
            raise ValueError(f"non-finding case must not carry vulnerability ground truth: {case_id}")

        _validate_provenance(case)
        if "pair_id" in case:
            pairs.setdefault(case["pair_id"], []).append(expected)

    # A subset (e.g. --limit) may hold one side of a pair; metrics only use complete pairs.
    for pair_id, decisions in pairs.items():
        if len(decisions) != len(set(decisions)) or not set(decisions) <= {"finding", "no_finding"}:
            raise ValueError(f"pair {pair_id} must contain at most one finding and one no_finding case")


def _validate_provenance(case: dict[str, Any]) -> None:
    case_id = case["case_id"]
    provenance = case.get("provenance")
    if provenance is None:
        if case["source_kind"] != "synthetic":
            raise ValueError(f"non-synthetic case requires provenance: {case_id}")
        return
    digest = hashlib.sha256(case["code"].encode("utf-8")).hexdigest()
    if provenance["content_sha256"] != digest:
        raise ValueError(f"provenance content_sha256 does not match code: {case_id}")
    has_upstream = "upstream" in provenance
    if provenance["origin"] == "upstream_derived" and not has_upstream:
        raise ValueError(f"upstream-derived case requires upstream provenance: {case_id}")
    if provenance["origin"] == "clean_room_synthetic" and has_upstream:
        raise ValueError(f"clean-room case must not reference upstream code: {case_id}")
    if (case["source_kind"] == "synthetic") != (provenance["origin"] == "clean_room_synthetic"):
        raise ValueError(f"source_kind and provenance origin disagree: {case_id}")


def assert_publicly_committable(cases: list[dict[str, Any]]) -> None:
    """Cases committed to the public repository must be clean-room synthetic."""
    for case in cases:
        if case["source_kind"] != "synthetic":
            raise ValueError(f"non-synthetic case must stay out of the public repository: {case['case_id']}")


def case_set_sha256(cases: list[dict[str, Any]]) -> str:
    """Order-independent hash of the full case content, including ground truth."""
    canonical = sorted(json.dumps(case, sort_keys=True, ensure_ascii=False) for case in cases)
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def case_set_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(cases),
        "by_expected_decision": dict(sorted(Counter(c["expected_decision"] for c in cases).items())),
        "by_language": dict(sorted(Counter(c["language"] for c in cases).items())),
        "by_source_kind": dict(sorted(Counter(c["source_kind"] for c in cases).items())),
    }


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
        if (prediction["decision"] == "invalid") != ("error" in prediction):
            raise ValueError(f"error is required for, and only allowed on, invalid predictions: {case_id}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
