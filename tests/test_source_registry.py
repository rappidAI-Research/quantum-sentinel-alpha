from copy import deepcopy

import pytest

from sentinel.data.registry import load_registry, validate_registry


def test_registry_is_valid_and_default_deny() -> None:
    registry = load_registry()
    validate_registry(registry)
    assert registry["sources"]
    assert not any(source["production_approved"] for source in registry["sources"])


def test_eval_source_cannot_be_training_approved() -> None:
    registry = load_registry()
    modified = deepcopy(registry)
    source = next(item for item in modified["sources"] if item["role"] == "eval_only")
    source.update(
        production_approved=True,
        revision="0123456789abcdef0123456789abcdef01234567",
        license_status="reviewed",
        terms_status="approved_for_project",
        redistribution_status="not_redistributed",
        reviewer="test-reviewer",
    )
    with pytest.raises(ValueError, match="eval-only source"):
        validate_registry(modified)
