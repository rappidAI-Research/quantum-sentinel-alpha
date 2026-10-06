from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.yaml"
REGISTRY_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "source_registry.schema.json"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("source registry must be a mapping")
    return document


def load_registry_schema() -> dict[str, Any]:
    with REGISTRY_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_registry(document: dict[str, Any]) -> None:
    Draft202012Validator(load_registry_schema()).validate(document)

    seen: set[str] = set()
    for source in document["sources"]:
        source_id = source["source_id"]
        if source_id in seen:
            raise ValueError(f"duplicate source_id: {source_id}")
        seen.add(source_id)

        approved = source["production_approved"]
        if not approved:
            continue

        if source["role"] == "eval_only":
            raise ValueError(f"eval-only source cannot be training-approved: {source_id}")
        if not source.get("revision"):
            raise ValueError(f"approved source requires immutable revision: {source_id}")
        if source["terms_status"] != "approved_for_project":
            raise ValueError(f"approved source requires terms approval: {source_id}")
        if source["redistribution_status"] not in {"approved", "not_redistributed"}:
            raise ValueError(f"approved source requires resolved redistribution status: {source_id}")
        if source["license_status"] != "reviewed":
            raise ValueError(f"approved source requires reviewed license status: {source_id}")
        if not source.get("reviewer"):
            raise ValueError(f"approved source requires reviewer: {source_id}")
