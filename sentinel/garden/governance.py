"""Training-export gate (default-deny).

A record is `training_approved` only if ALL of the following hold:
- it passed every quality, deduplication and contamination check;
- its split is train or validation (never holdout_reserved / temporal_quarantine);
- every registry source it depends on is `production_approved` and none is eval-only;
- its repository has an `approved_for_training` entry in data/garden/approvals.yaml whose reviewed
  license file hash matches the license file recorded for the record's fixed revision;
- if the reviewer-confirmed license is not permissive (copyleft, unknown, custom), the approval
  carries a written `non_permissive_justification`.

Anything else is `pending_review` (could become trainable after review) or `excluded` (can never be
trained on in this dataset version). Blockers are listed by name; nothing is approved implicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from sentinel.garden.licenses import category_of

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROVALS_PATH = PROJECT_ROOT / "data" / "garden" / "approvals.yaml"
APPROVALS_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "garden_approvals.schema.json"


def load_approvals(path: Path = APPROVALS_PATH) -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = json.loads(APPROVALS_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    approvals: dict[str, dict[str, Any]] = {}
    for entry in document["repositories"]:
        key = entry["repository_key"]
        if key in approvals:
            raise ValueError(f"duplicate approval for repository: {key}")
        if entry["decision"] == "approved_for_training":
            if entry["terms_status"] != "approved_for_project" or entry["redistribution_status"] == "blocked":
                raise ValueError(f"approval for {key} contradicts its terms/redistribution status")
        approvals[key] = entry
    return approvals


def gate(
    record: dict[str, Any],
    registry_sources: dict[str, dict[str, Any]],
    approvals: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    excluded: list[str] = []
    pending: list[str] = []
    provenance = record["provenance"]

    if record["quality"]["status"] != "accepted":
        excluded.append("quality_rejected")
    if record["split"] not in {"train", "validation"}:
        excluded.append(f"split:{record['split']}")

    for source_id in [provenance["code_source_id"], *provenance["metadata_source_ids"]]:
        source = registry_sources.get(source_id)
        if source is None:
            excluded.append(f"source_unregistered:{source_id}")
        elif source["role"] == "eval_only":
            excluded.append(f"source_eval_only:{source_id}")
        elif not source["production_approved"]:
            pending.append(f"source_not_approved:{source_id}")

    approval = approvals.get(provenance["repository_key"])
    if approval is None:
        pending.append("repository_not_reviewed")
    elif approval["decision"] == "blocked":
        excluded.append("repository_blocked")
    else:
        recorded = {f["path"]: f["sha256"] for f in provenance["license"]["files"]}
        if recorded.get(approval["license_file"]) != approval["license_file_sha256"]:
            pending.append("repository_approval_license_mismatch")
        # copyleft, unknown and custom licenses need an explicit written basis, never a plain approval
        if category_of(approval["license_spdx"]) != "permissive" and not approval.get("non_permissive_justification"):
            pending.append("non_permissive_license_requires_justification")

    if excluded:
        return "excluded", excluded + pending
    if pending:
        return "pending_review", pending
    return "training_approved", []
