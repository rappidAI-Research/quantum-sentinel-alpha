from pathlib import Path

import pytest
import yaml

from sentinel.foundation_model import FOUNDATION_CONFIG, load_foundation, validate_foundation

SELECTED = "ornith-ai/Ornith-1.5-9B"
SELECTED_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
PREVIOUS = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")


def write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "foundation.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def raw() -> dict:
    return yaml.safe_load(FOUNDATION_CONFIG.read_text(encoding="utf-8"))


def test_selected_foundation_is_pinned_ornith_15_9b() -> None:
    foundation = load_foundation()
    validate_foundation(foundation)
    assert foundation.model_id == SELECTED
    assert foundation.revision == SELECTED_REVISION
    assert foundation.license == "MIT"
    assert foundation.architecture == "Qwen3_5ForConditionalGeneration"
    # The decision rests on expected capability, not on a measured security advantage.
    assert foundation.basis == "expected_capability"
    assert foundation.bakeoff_config.name == "baseline_ornith15.yaml"
    assert foundation.baseline_candidate == "ornith15-9b"


def test_previous_foundation_stays_documented_but_inactive() -> None:
    document = raw()
    assert [(h["model_id"], h["revision"]) for h in document["history"]] == [PREVIOUS]
    assert (document["model_id"], document["revision"]) != PREVIOUS
    from sentinel.train.config import load_train_config

    assert load_train_config().model_id == SELECTED


def test_unpinned_revision_is_rejected(tmp_path: Path) -> None:
    document = raw()
    document["revision"] = "main"
    with pytest.raises(ValueError, match="40-character"):
        load_foundation(write(tmp_path, document))


def test_expected_capability_decision_must_state_limits(tmp_path: Path) -> None:
    document = raw()
    document["decision"]["not_claimed"] = []
    with pytest.raises(ValueError, match="not claimed"):
        load_foundation(write(tmp_path, document))


def test_baseline_candidate_must_match_selected_revision(tmp_path: Path) -> None:
    document = raw()
    document["revision"] = "0" * 40
    foundation = load_foundation(write(tmp_path, document))
    with pytest.raises(ValueError, match="revision"):
        validate_foundation(foundation)


def test_baseline_candidate_must_exist(tmp_path: Path) -> None:
    document = raw()
    document["optional_comparison"]["baseline_candidate"] = "qwen35-27b"
    with pytest.raises(ValueError, match="not in"):
        validate_foundation(load_foundation(write(tmp_path, document)))
