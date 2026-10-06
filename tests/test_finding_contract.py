import pytest
from jsonschema import ValidationError

from sentinel.contracts import validate_finding


def valid_finding() -> dict:
    return {
        "schema_version": "1.0",
        "id": "QSA-example-001",
        "severity": "high",
        "confidence": 0.91,
        "cwe": "CWE-79",
        "file": "src/render.py",
        "start_line": 41,
        "end_line": 44,
        "summary": "Untrusted input reaches an HTML sink without output encoding.",
        "evidence": ["request.args['q'] -> template fragment -> HTML response"],
        "preconditions": ["Attacker controls the q query parameter."],
        "remediation": "Use the framework's escaping path and avoid marking attacker data safe.",
        "patch": None,
        "verification": {"status": "unverified", "tests": []},
    }


def test_valid_finding_is_accepted() -> None:
    validate_finding(valid_finding())


def test_confidence_outside_unit_interval_is_rejected() -> None:
    finding = valid_finding()
    finding["confidence"] = 1.1
    with pytest.raises(ValidationError):
        validate_finding(finding)


def test_free_form_extra_fields_are_rejected() -> None:
    finding = valid_finding()
    finding["chain_of_thought"] = "should never be part of the contract"
    with pytest.raises(ValidationError):
        validate_finding(finding)


def test_reversed_line_span_is_rejected() -> None:
    finding = valid_finding()
    finding["start_line"] = 50
    finding["end_line"] = 40
    with pytest.raises(ValueError, match="end_line"):
        validate_finding(finding)


def test_parent_path_escape_is_rejected() -> None:
    finding = valid_finding()
    finding["file"] = "../outside.py"
    with pytest.raises(ValueError, match="repository boundary"):
        validate_finding(finding)
