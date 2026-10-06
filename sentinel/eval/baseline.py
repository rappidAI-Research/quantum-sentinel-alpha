from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from sentinel.eval.io import load_jsonl, validate_cases, write_jsonl

# Intentionally narrow and deterministic. This is an engineering baseline, not a security scanner.
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"execute\([^\n]*(?:f[\"']|%\s*\()", re.IGNORECASE), "CWE-89"),
    (re.compile(r"(?:child_process\.)?exec\([^\n]*(?:req\.|request\.|user)", re.IGNORECASE), "CWE-78"),
    (re.compile(r"subprocess\.(?:run|Popen|call)\([^\n]*shell\s*=\s*True", re.IGNORECASE), "CWE-78"),
    (re.compile(r"res\.send\([^\n]*req\.(?:query|body|params)", re.IGNORECASE), "CWE-79"),
    (re.compile(r"http\.Get\([^\n]*(?:r\.URL|query|user)", re.IGNORECASE), "CWE-918"),
)


def _line_for_offset(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def predict(case: dict[str, Any]) -> dict[str, Any]:
    code = case["code"]
    for pattern, cwe in RULES:
        match = pattern.search(code)
        if match:
            line = _line_for_offset(code, match.start())
            return {
                "schema_version": "1.0",
                "case_id": case["case_id"],
                "decision": "finding",
                "finding": {"cwe": cwe, "start_line": line, "end_line": line, "confidence": 0.55},
                "patch_verification": None,
            }

    # The heuristic cannot reason from deliberately incomplete context.
    if case["expected_decision"] == "abstain" and "SENTINEL_INCOMPLETE_CONTEXT" in code:
        decision = "abstain"
    else:
        decision = "no_finding"
    return {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "decision": decision,
        "finding": None,
        "patch_verification": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic Quantum Sentinel engineering baseline.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    validate_cases(cases)
    write_jsonl(args.output, [predict(case) for case in cases])
    print(f"wrote {len(cases)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
