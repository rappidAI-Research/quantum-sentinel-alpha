from pathlib import Path

import pytest
import yaml

from sentinel.garden.pipeline import DEFAULT_CONFIG_PATH, load_config
from sentinel.garden.split import load_split_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "garden" / "split_policy.yaml"


def test_frozen_split_policy_values_are_pinned() -> None:
    """Changing these re-splits every Garden dataset: it needs a new policy_id and an explicit decision."""
    policy = load_split_policy(POLICY)
    assert policy.policy_id == "sentinel-garden-split-v1"
    assert policy.status == "frozen"
    assert policy.salt == "sentinel-garden-seed-v1-split"
    assert (policy.validation_per_mille, policy.holdout_reserved_per_mille) == (100, 100)
    assert policy.temporal_cutoff == "2026-07-01T00:00:00Z"


def test_dataset_config_uses_the_frozen_policy() -> None:
    assert load_config(DEFAULT_CONFIG_PATH).split == load_split_policy(POLICY)


def test_unfrozen_or_incomplete_policy_is_refused(tmp_path: Path) -> None:
    document = yaml.safe_load(POLICY.read_text())
    document["status"] = "provisional"
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="frozen"):
        load_split_policy(path)
    del document["salt"]
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="exactly"):
        load_split_policy(path)
