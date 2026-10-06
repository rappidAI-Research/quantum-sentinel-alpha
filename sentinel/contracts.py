from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINDING_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "finding.schema.json"


def load_finding_schema() -> dict[str, Any]:
    with FINDING_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_finding(payload: dict[str, Any]) -> None:
    """Validate one evidence-backed Sentinel finding against the public contract."""
    Draft202012Validator(load_finding_schema()).validate(payload)

    if payload["end_line"] < payload["start_line"]:
        raise ValueError("end_line must be greater than or equal to start_line")

    path = PurePosixPath(payload["file"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("finding file must stay within the repository boundary")
