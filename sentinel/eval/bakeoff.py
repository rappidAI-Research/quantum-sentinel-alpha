"""Bake-off configuration: one shared evaluation setup, per-candidate model identity only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinel.eval.backends import GenerationSettings, RuntimeSettings
from sentinel.eval.io import PROJECT_ROOT
from sentinel.eval.prompt import PROMPT_VERSION
from sentinel.eval.runner import is_immutable_revision

TOP_LEVEL_KEYS = {"bakeoff_id", "cases", "prompt_version", "generation", "runtime", "candidates"}
CANDIDATE_KEYS = {"name", "model_id", "revision", "runtime_overrides", "override_reason"}
OVERRIDABLE_RUNTIME_KEYS = {"device", "dtype", "quantization"}


@dataclass(frozen=True)
class Candidate:
    name: str
    model_id: str
    revision: str | None
    runtime: RuntimeSettings
    override_reason: str | None


@dataclass(frozen=True)
class BakeoffConfig:
    bakeoff_id: str
    cases: Path
    generation: GenerationSettings
    candidates: dict[str, Candidate]
    sha256: str


def load_bakeoff(path: Path) -> BakeoffConfig:
    raw_bytes = path.read_bytes()
    document = yaml.safe_load(raw_bytes)
    if not isinstance(document, dict):
        raise ValueError("bake-off config must be a mapping")
    _require_keys(document, TOP_LEVEL_KEYS, TOP_LEVEL_KEYS, "bake-off config")
    if document["prompt_version"] != PROMPT_VERSION:
        raise ValueError(f"config targets prompt {document['prompt_version']!r}, code provides {PROMPT_VERSION!r}")

    cases = PROJECT_ROOT / document["cases"]
    if not cases.is_file():
        raise ValueError(f"bake-off case file not found: {document['cases']}")
    generation = GenerationSettings(**document["generation"])
    shared_runtime = dict(document["runtime"])
    RuntimeSettings(**shared_runtime)

    candidates: dict[str, Candidate] = {}
    model_ids: set[str] = set()
    raw_candidates = document["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        raise ValueError("a bake-off needs at least two candidates")
    for raw in raw_candidates:
        _require_keys(raw, CANDIDATE_KEYS, {"name", "model_id", "revision"}, "candidate")
        name, model_id, revision = raw["name"], raw["model_id"], raw["revision"]
        if name in candidates:
            raise ValueError(f"duplicate candidate name: {name}")
        if model_id in model_ids:
            raise ValueError(f"duplicate candidate model_id: {model_id}")
        if revision is not None and not is_immutable_revision(revision):
            raise ValueError(f"{name}: revision must be null or a 40-character commit SHA")
        overrides = raw.get("runtime_overrides") or {}
        unknown = set(overrides) - OVERRIDABLE_RUNTIME_KEYS
        if unknown:
            raise ValueError(f"{name}: only {sorted(OVERRIDABLE_RUNTIME_KEYS)} may be overridden, got {sorted(unknown)}")
        reason = raw.get("override_reason")
        if overrides and not reason:
            raise ValueError(f"{name}: runtime_overrides require an override_reason")
        candidates[name] = Candidate(
            name=name,
            model_id=model_id,
            revision=revision,
            runtime=RuntimeSettings(**{**shared_runtime, **overrides}),
            override_reason=reason if overrides else None,
        )
        model_ids.add(model_id)

    return BakeoffConfig(
        bakeoff_id=document["bakeoff_id"],
        cases=cases,
        generation=generation,
        candidates=candidates,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _require_keys(mapping: Any, allowed: set[str], required: set[str], label: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{label} has unsupported keys: {sorted(unknown)}")
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"{label} is missing keys: {sorted(missing)}")
