"""Selected training foundation: load and cross-check `configs/model/foundation.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinel.eval.bakeoff import load_bakeoff
from sentinel.eval.runner import is_immutable_revision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG = PROJECT_ROOT / "configs" / "model" / "foundation.yaml"

REQUIRED_KEYS = {
    "schema_version",
    "foundation_status",
    "model_id",
    "revision",
    "license",
    "architecture",
    "decision",
    "optional_comparison",
}
DECISION_KEYS = {"date", "decided_by", "basis", "rationale", "not_claimed"}
DECISION_BASES = {"expected_capability", "experimental_evidence"}


@dataclass(frozen=True)
class Foundation:
    model_id: str
    revision: str
    license: str
    architecture: str
    basis: str
    bakeoff_config: Path
    baseline_candidate: str


def load_foundation(path: Path = FOUNDATION_CONFIG) -> Foundation:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("foundation config must be a mapping")
    missing = REQUIRED_KEYS - set(document)
    if missing:
        raise ValueError(f"foundation config is missing keys: {sorted(missing)}")
    if document["schema_version"] != "1.0":
        raise ValueError("unsupported foundation config schema_version")
    if document["foundation_status"] != "selected":
        raise ValueError("foundation_status must be 'selected' once a training foundation is chosen")
    if not is_immutable_revision(document["revision"]):
        raise ValueError("foundation revision must be a 40-character commit SHA")

    decision = document["decision"]
    if not isinstance(decision, dict) or DECISION_KEYS - set(decision):
        raise ValueError(f"decision must contain {sorted(DECISION_KEYS)}")
    if decision["basis"] not in DECISION_BASES:
        raise ValueError(f"decision.basis must be one of {sorted(DECISION_BASES)}")
    if not decision["rationale"]:
        raise ValueError("decision.rationale must not be empty")
    if decision["basis"] == "expected_capability" and not decision["not_claimed"]:
        raise ValueError("an expected-capability decision must state what is not claimed")

    comparison: dict[str, Any] = document["optional_comparison"]
    return Foundation(
        model_id=document["model_id"],
        revision=document["revision"],
        license=document["license"],
        architecture=document["architecture"],
        basis=decision["basis"],
        bakeoff_config=PROJECT_ROOT / comparison["bakeoff_config"],
        baseline_candidate=comparison["baseline_candidate"],
    )


def validate_foundation(foundation: Foundation) -> None:
    """The baseline evaluation path must load exactly the selected checkpoint."""
    bakeoff = load_bakeoff(foundation.bakeoff_config)
    candidate = bakeoff.candidates.get(foundation.baseline_candidate)
    if candidate is None:
        raise ValueError(f"baseline candidate {foundation.baseline_candidate!r} is not in {foundation.bakeoff_config.name}")
    if candidate.model_id != foundation.model_id:
        raise ValueError(f"baseline candidate model {candidate.model_id} != selected foundation {foundation.model_id}")
    if candidate.revision != foundation.revision:
        raise ValueError(
            f"baseline candidate revision {candidate.revision} != selected foundation revision {foundation.revision}"
        )
