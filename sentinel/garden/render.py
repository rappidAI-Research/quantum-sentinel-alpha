"""Render Garden records in the formats the rest of the project already uses.

- `to_chat_example`: the exact QuickEval prompt (`sentinel.eval.prompt.build_messages`) plus an
  assistant reply in the QuickEval output contract. Training and evaluation therefore share one
  input/output format, and every rendered reply is checked with the strict evaluation parser.
- `to_eval_case`: an `upstream_derived` evaluation case (eval_case.schema.json) for
  holdout_reserved records. Such cases belong in the Git-ignored eval/sentinelbench/private/.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sentinel.eval.parse import parse_model_output
from sentinel.eval.prompt import PROMPT_VERSION, build_messages

# The output contract requires a confidence for findings. Garden labels carry no calibrated
# confidence, so the training format uses one fixed, documented value. It is a formatting policy,
# not a measurement, and can be changed before the first training run without rebuilding data.
FINDING_CONFIDENCE = 0.9


def _case_view(record: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": record["record_id"], **record["input"]}


def assistant_reply(record: dict[str, Any], finding_confidence: float = FINDING_CONFIDENCE) -> str:
    target = record["target"]
    if target["decision"] == "finding":
        payload: dict[str, Any] = {
            "decision": "finding",
            "cwe": target["cwe"],
            "start_line": target["start_line"],
            "end_line": target["end_line"],
            "confidence": finding_confidence,
            "evidence": target["evidence"],
        }
    else:
        payload = {"decision": target["decision"], "evidence": target["evidence"]}
    return json.dumps(payload, ensure_ascii=False)


def to_chat_example(record: dict[str, Any], finding_confidence: float = FINDING_CONFIDENCE) -> dict[str, Any]:
    messages = build_messages(_case_view(record))
    messages.append({"role": "assistant", "content": assistant_reply(record, finding_confidence)})
    return {"record_id": record["record_id"], "prompt_version": PROMPT_VERSION, "messages": messages}


def reply_matches_contract(record: dict[str, Any]) -> bool:
    """The rendered reply must parse under the strict evaluation parser to the same label."""
    parsed = parse_model_output(assistant_reply(record), _case_view(record))
    prediction = parsed.prediction
    target = record["target"]
    if prediction["decision"] != target["decision"]:
        return False
    if target["decision"] == "finding":
        finding = prediction["finding"]
        return (finding["cwe"], finding["start_line"], finding["end_line"]) == (
            target["cwe"],
            target["start_line"],
            target["end_line"],
        )
    return prediction["finding"] is None


def to_eval_case(record: dict[str, Any]) -> dict[str, Any]:
    target, provenance = record["target"], record["provenance"]
    finding = target["decision"] == "finding"
    truth: dict[str, Any] = {
        "cwe": target["cwe"] if finding else None,
        "start_line": target["start_line"] if finding else None,
        "end_line": target["end_line"] if finding else None,
    }
    if finding and target["alternate_cwes"]:
        truth["alternate_cwes"] = list(target["alternate_cwes"])
    case_type = {"finding": "vulnerable", "no_finding": "hard_negative", "abstain": "abstention"}[target["decision"]]
    case: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": record["record_id"],
        "language": record["input"]["language"],
        "source_kind": "private_holdout",
        "task": "review",
        "expected_decision": target["decision"],
        "case_type": case_type,
        "file": record["input"]["file"],
        "context": record["input"]["context"],
        "code": record["input"]["code"],
        "ground_truth": truth,
        "provenance": {
            "origin": "upstream_derived",
            "content_sha256": hashlib.sha256(record["input"]["code"].encode("utf-8")).hexdigest(),
            "upstream": {
                "source_id": provenance["code_source_id"],
                "repository": provenance["repository"],
                "revision": provenance["excerpt_revision"],
                "file": provenance["file_path"],
                "license": provenance["license"]["detected"],
                "license_status": "reviewed" if provenance["license"]["status"] == "reviewed" else "unreviewed",
                "redistribution_status": "not_redistributed",
            },
        },
    }
    if record["pair_id"] and target["decision"] in {"finding", "no_finding"}:
        case["pair_id"] = record["pair_id"]
    return case
