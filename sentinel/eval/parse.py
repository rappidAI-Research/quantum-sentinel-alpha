"""Strict parser for the QuickEval model output contract.

Only format normalization is applied (reasoning-tag and markdown-fence stripping, locating the
JSON object, CWE spelling, case of the decision). Missing or malformed security fields are never
filled in: such output becomes an ``invalid`` prediction carrying the error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

PARSER_VERSION = "1"

DECISIONS = {"finding", "no_finding", "abstain"}
FINDING_KEYS = {"decision", "cwe", "start_line", "end_line", "confidence", "evidence"}
NON_FINDING_KEYS = {"decision", "evidence"}
MAX_EVIDENCE_CHARS = 1000

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)
_CWE = re.compile(r"^CWE[-_ ]?0*([1-9][0-9]*)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedOutput:
    prediction: dict[str, Any]
    evidence: str | None
    error: str | None
    visible_output: str


class OutputError(ValueError):
    pass


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Remove reasoning so it is neither scored nor stored.

    Returns the visible text and whether an unterminated reasoning block was cut off
    (e.g. generation hit max_new_tokens while still "thinking").
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:  # opening tag was part of the prompt template
        text = text.rsplit("</think>", 1)[1]
    truncated = "<think>" in text
    if truncated:  # everything after an unclosed tag is reasoning
        text = text.split("<think>", 1)[0]
    return text.strip(), truncated


def _extract_object(text: str) -> dict[str, Any]:
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise OutputError("no JSON object in output")
    try:
        value, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise OutputError(f"invalid JSON: {exc.msg}") from exc
    if "{" in text[end:]:
        raise OutputError("more than one JSON object in output")
    if not isinstance(value, dict):
        raise OutputError("output is not a JSON object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _evidence(value: Any, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise OutputError("evidence must be a non-empty string")
    if len(value) > MAX_EVIDENCE_CHARS:
        raise OutputError(f"evidence exceeds {MAX_EVIDENCE_CHARS} characters")
    return value.strip()


def _validate(payload: dict[str, Any], line_count: int) -> tuple[dict[str, Any] | None, str, str | None]:
    decision = payload.get("decision")
    if not isinstance(decision, str) or decision.strip().lower() not in DECISIONS:
        raise OutputError(f"decision must be one of {sorted(DECISIONS)}")
    decision = decision.strip().lower()

    if decision != "finding":
        # Null-valued finding fields are a harmless formatting variant; real values are not.
        payload = {key: value for key, value in payload.items() if value is not None or key == "decision"}
        extra = set(payload) - NON_FINDING_KEYS
        if extra:
            raise OutputError(f"unexpected fields for {decision}: {sorted(extra)}")
        return None, decision, _evidence(payload.get("evidence"), required=False)

    missing = FINDING_KEYS - set(payload)
    if missing:
        raise OutputError(f"finding is missing fields: {sorted(missing)}")
    extra = set(payload) - FINDING_KEYS
    if extra:
        raise OutputError(f"unexpected fields for finding: {sorted(extra)}")

    cwe_raw = payload["cwe"]
    if not isinstance(cwe_raw, str):
        raise OutputError("cwe must be a string")
    if cwe_raw.strip().lower() == "unknown":
        cwe = "unknown"
    else:
        match = _CWE.match(cwe_raw.strip())
        if not match:
            raise OutputError(f"cwe is not a CWE identifier: {cwe_raw!r}")
        cwe = f"CWE-{match.group(1)}"

    start, end = payload["start_line"], payload["end_line"]
    if not (_is_int(start) and _is_int(end)):
        raise OutputError("start_line and end_line must be integers")
    if not 1 <= start <= end <= line_count:
        raise OutputError(f"line span {start}-{end} is outside the excerpt (1-{line_count}) or reversed")

    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.0 <= confidence <= 1.0:
        raise OutputError("confidence must be a number between 0.0 and 1.0")

    finding = {"cwe": cwe, "start_line": start, "end_line": end, "confidence": float(confidence)}
    return finding, decision, _evidence(payload["evidence"], required=True)


def parse_model_output(text: str, case: dict[str, Any]) -> ParsedOutput:
    visible, truncated = strip_reasoning(text)
    base = {"schema_version": "1.0", "case_id": case["case_id"], "patch_verification": None}
    try:
        if truncated:
            raise OutputError("unterminated reasoning block (output truncated before an answer)")
        payload = _extract_object(visible)
        finding, decision, evidence = _validate(payload, len(case["code"].splitlines()))
    except OutputError as exc:
        prediction = {**base, "decision": "invalid", "finding": None, "error": str(exc)}
        return ParsedOutput(prediction=prediction, evidence=None, error=str(exc), visible_output=visible)
    prediction = {**base, "decision": decision, "finding": finding}
    return ParsedOutput(prediction=prediction, evidence=evidence, error=None, visible_output=visible)
