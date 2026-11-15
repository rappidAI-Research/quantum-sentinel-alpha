from copy import deepcopy

import pytest

from sentinel.data.registry import load_registry, validate_registry


GARDEN_SOURCES = {"ghsa-osv-fix-commits", "osv", "github-advisory-db"}


def test_registry_is_valid_and_default_deny() -> None:
    """Only the three Sentinel Garden sources may be approved (decided 2026-11-15, docs/TRAINING_AUTHORIZATION.md);
    every other source stays unapproved, and every approval names a reviewer and a pinned revision."""
    registry = load_registry()
    validate_registry(registry)
    assert registry["sources"]
    approved = {s["source_id"] for s in registry["sources"] if s["production_approved"]}
    assert approved <= GARDEN_SOURCES
    for source in registry["sources"]:
        if source["production_approved"]:
            assert source["reviewer"] and "claude" not in source["reviewer"].lower()
            assert source["license_status"] == "reviewed" and source["terms_status"] == "approved_for_project"


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
