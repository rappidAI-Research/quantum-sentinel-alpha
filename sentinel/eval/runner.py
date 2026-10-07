"""Shared QuickEval run path: identical prompt, parser and scorer for every backend."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from sentinel.eval.backends import GenerationSettings, ModelBackend, RuntimeSettings
from sentinel.eval.coding_regression import CodingRegressionSuite, not_run
from sentinel.eval.io import PROJECT_ROOT, case_set_sha256, case_set_summary, validate_cases, validate_predictions
from sentinel.eval.metrics import SCORER_VERSION, score
from sentinel.eval.parse import PARSER_VERSION, parse_model_output
from sentinel.eval.prompt import PROMPT_VERSION, build_messages, prompt_sha256

ARTIFACT_VERSION = "1.0"
ARTIFACT_KIND = "quantum-sentinel-alpha.quickeval-run"
RUN_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "eval_run.schema.json"
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
MAX_STORED_OUTPUT_CHARS = 4000


def is_immutable_revision(revision: str | None) -> bool:
    return bool(revision and IMMUTABLE_REVISION.match(revision))


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def reportability_blockers(artifact: dict[str, Any], full_case_set: bool) -> list[str]:
    blockers: list[str] = []
    model = artifact["model"]
    if model["backend"] != "hf":
        blockers.append(f"backend {model['backend']!r} is not a real model")
    if not is_immutable_revision(model["revision"]):
        blockers.append("model revision is not an immutable 40-character commit SHA")
    resolved = artifact["runtime"].get("resolved_revision")
    if model["backend"] == "hf" and not resolved:
        blockers.append("the loaded model revision could not be confirmed")
    elif resolved and model["revision"] and resolved != model["revision"]:
        blockers.append(f"loaded revision {resolved} differs from requested {model['revision']}")
    if model["backend"] == "hf":
        requested, loaded = artifact["runtime"]["dtype"], artifact["runtime"].get("loaded_dtype")
        if loaded != requested:
            blockers.append(f"loaded model dtype {loaded} differs from requested {requested}")
    if not full_case_set:
        blockers.append("run used a subset of the case file (--limit)")
    git = artifact["git"]
    if not git["commit"]:
        blockers.append("git commit of the evaluation code is unknown")
    elif git["dirty"]:
        blockers.append("evaluation code has uncommitted changes")
    return blockers


def validate_run_artifact(artifact: dict[str, Any]) -> None:
    with RUN_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator(schema).validate(artifact)


def run_quickeval(
    cases: list[dict[str, Any]],
    backend: ModelBackend,
    *,
    model_id: str,
    revision: str | None,
    generation: GenerationSettings,
    runtime: RuntimeSettings,
    cases_path: str,
    full_case_set: bool = True,
    bakeoff: dict[str, Any] | None = None,
    coding_suite: CodingRegressionSuite | None = None,
) -> dict[str, Any]:
    validate_cases(cases)
    started_at = datetime.now(timezone.utc)

    load_start = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - load_start

    records: list[dict[str, Any]] = []
    generation_seconds = 0.0
    generated_tokens: int | None = 0
    for case in cases:
        case_start = time.perf_counter()
        result = backend.generate(build_messages(case))
        latency = time.perf_counter() - case_start
        generation_seconds += latency
        if result.generated_tokens is None:
            generated_tokens = None
        elif generated_tokens is not None:
            generated_tokens += result.generated_tokens

        parsed = parse_model_output(result.text, case)
        records.append(
            {
                "case_id": case["case_id"],
                "expected_decision": case["expected_decision"],
                "prediction": parsed.prediction,
                "evidence": parsed.evidence,
                "output": parsed.visible_output[:MAX_STORED_OUTPUT_CHARS],
                "latency_seconds": latency,
                "generated_tokens": result.generated_tokens,
                "prompt_tokens": result.prompt_tokens,
            }
        )

    predictions = [record["prediction"] for record in records]
    validate_predictions(predictions)
    metrics = score(cases, predictions).as_dict()

    coding = not_run() if coding_suite is None else coding_suite.run(backend).as_dict()

    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": ARTIFACT_KIND,
        "created_at": started_at.isoformat(),
        "reportable": False,
        "reportable_blockers": [],
        "git": git_state(),
        "model": {"id": model_id, "revision": revision, "backend": backend.name},
        "runtime": {**runtime.as_dict(), **backend.describe()},
        "generation": generation.as_dict(),
        "contract": {
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_sha256(),
            "parser_version": PARSER_VERSION,
            "scorer_version": SCORER_VERSION,
        },
        "cases": {"path": cases_path, "sha256": case_set_sha256(cases), **case_set_summary(cases)},
        "bakeoff": bakeoff,
        "metrics": metrics,
        "efficiency": {
            "model_load_seconds": load_seconds,
            "total_eval_seconds": generation_seconds,
            "mean_case_latency_seconds": generation_seconds / len(cases) if cases else None,
            "generated_tokens": generated_tokens,
            "tokens_per_second": (
                generated_tokens / generation_seconds if generated_tokens and generation_seconds > 0 else None
            ),
            "peak_vram_gib": backend.peak_vram_gib(),
        },
        "coding_regression": coding,
        "predictions": records,
    }
    blockers = reportability_blockers(artifact, full_case_set)
    artifact["reportable"] = not blockers
    artifact["reportable_blockers"] = blockers
    validate_run_artifact(artifact)
    return artifact


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validate_run_artifact(artifact)
    return artifact
