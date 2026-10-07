import json

from sentinel.eval.parse import parse_model_output

CASE = {"case_id": "case-a", "code": "a = 1\nb = run(a)\nc = 3\n"}


def finding(**overrides) -> dict:
    payload = {
        "decision": "finding",
        "cwe": "CWE-78",
        "start_line": 2,
        "end_line": 2,
        "confidence": 0.8,
        "evidence": "User input reaches a shell.",
    }
    payload.update(overrides)
    return payload


def test_valid_finding_is_parsed() -> None:
    parsed = parse_model_output(json.dumps(finding()), CASE)
    assert parsed.error is None
    assert parsed.prediction["decision"] == "finding"
    assert parsed.prediction["finding"] == {"cwe": "CWE-78", "start_line": 2, "end_line": 2, "confidence": 0.8}
    assert parsed.evidence == "User input reaches a shell."


def test_reasoning_and_fences_are_stripped_and_not_stored() -> None:
    text = "<think>secret chain of thought</think>\n```json\n" + json.dumps(finding(cwe="cwe 78")) + "\n```"
    parsed = parse_model_output(text, CASE)
    assert parsed.error is None
    assert parsed.prediction["finding"]["cwe"] == "CWE-78"
    assert "secret" not in parsed.visible_output


def test_malformed_json_is_an_invalid_prediction() -> None:
    parsed = parse_model_output('{"decision": "finding", "cwe": ', CASE)
    assert parsed.prediction["decision"] == "invalid"
    assert parsed.prediction["finding"] is None
    assert "invalid JSON" in parsed.prediction["error"]


def test_missing_security_fields_are_not_invented() -> None:
    payload = finding()
    del payload["start_line"]
    parsed = parse_model_output(json.dumps(payload), CASE)
    assert parsed.prediction["decision"] == "invalid"
    assert "start_line" in parsed.error


def test_out_of_range_span_and_bad_confidence_are_invalid() -> None:
    assert parse_model_output(json.dumps(finding(end_line=9)), CASE).prediction["decision"] == "invalid"
    assert parse_model_output(json.dumps(finding(confidence=85)), CASE).prediction["decision"] == "invalid"
    assert parse_model_output(json.dumps(finding(start_line="2")), CASE).prediction["decision"] == "invalid"


def test_unknown_decision_and_extra_fields_are_invalid() -> None:
    assert parse_model_output('{"decision": "maybe"}', CASE).prediction["decision"] == "invalid"
    assert parse_model_output(json.dumps(finding(reasoning="step 1")), CASE).prediction["decision"] == "invalid"


def test_non_finding_tolerates_nulls_but_not_finding_values() -> None:
    ok = parse_model_output('{"decision": "Abstain", "cwe": null, "evidence": "Sink not shown."}', CASE)
    assert ok.prediction["decision"] == "abstain" and ok.prediction["finding"] is None
    smuggled = parse_model_output('{"decision": "no_finding", "cwe": "CWE-79"}', CASE)
    assert smuggled.prediction["decision"] == "invalid"
