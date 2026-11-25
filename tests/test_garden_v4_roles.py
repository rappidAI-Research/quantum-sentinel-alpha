"""Garden v4 evaluation roles: a fresh development set and an untouched final test set, never training data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sentinel.eval.io import case_set_sha256
from sentinel.garden import pipeline
from sentinel.garden.audit import patch_fingerprint, split_leakage
from sentinel.train import data as train_data
from sentinel.train.data import GovernanceError, load_training_data
from tests.train_fixtures import GardenWorkdir, approved_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4 = PROJECT_ROOT / "configs" / "garden" / "seed_v4.yaml"


def _record(repository: str, split: str, status: str = "training_approved") -> dict:
    return {"split": split, "provenance": {"repository_key": repository}, "governance": {"approval_status": status, "blockers": []}}


@pytest.mark.parametrize(
    ("repository", "split", "role"),
    [
        ("a/train", "train", "training"),
        ("a/old-dev", "validation", "retired_development"),
        ("a/new-dev", "validation", "development"),
        ("a/old-holdout", "holdout_reserved", "previous_holdout"),
        ("a/new-holdout", "holdout_reserved", "final_test"),
        ("a/train", "temporal_quarantine", "temporal_quarantine"),
    ],
)
def test_roles_follow_the_split_and_what_earlier_versions_evaluated(repository: str, split: str, role: str) -> None:
    assert pipeline.evaluation_role(_record(repository, split), {"a/old-dev"}, {"a/old-holdout"}) == role


def test_development_records_are_never_training_approved() -> None:
    record = _record("a/new-dev", "validation")
    pipeline.apply_evaluation_role(record, ({"a/old-dev"}, set()))
    assert record["governance"] == {"approval_status": "excluded", "blockers": ["evaluation_role:development"]}
    pending = _record("a/new-dev", "validation", "pending_review")
    pending["governance"]["blockers"] = ["repository_not_reviewed"]
    pipeline.apply_evaluation_role(pending, ({"a/old-dev"}, set()))
    # a later approval of the repository cannot turn it into training data
    assert pending["governance"] == {"approval_status": "excluded", "blockers": ["evaluation_role:development", "repository_not_reviewed"]}


def test_v4_history_is_the_v3_development_set_and_every_exported_holdout() -> None:
    config = pipeline.load_config(V4)
    retired, previous = pipeline.evaluation_history(config)
    rows = [json.loads(line) for line in (PROJECT_ROOT / "data" / "garden" / "seed_v3" / "manifest.jsonl").read_text().splitlines()]
    dev = [r for r in rows if r["split"] == "validation" and r["quality"]["status"] == "accepted" and r["governance"]["approval_status"] == "training_approved"]
    assert len(dev) == 133 and retired == {r["provenance"]["repository_key"] for r in dev}  # the 133 cases checkpoints were selected on
    holdout = {r["provenance"]["repository_key"] for r in rows if r["split"] == "holdout_reserved" and r["quality"]["status"] == "accepted"}
    assert holdout <= previous and not retired & previous
    assert pipeline.evaluation_history(pipeline.load_config(PROJECT_ROOT / "configs" / "garden" / "seed_v3.yaml")) is None


@pytest.fixture
def garden(monkeypatch: pytest.MonkeyPatch):
    workdir = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    yield workdir
    workdir.cleanup()


def test_role_exports_are_a_devset_and_a_sealed_final_test(garden: GardenWorkdir) -> None:
    records = []
    for record in copy.deepcopy(garden.train + garden.validation):
        record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
        record["evaluation_role"] = "development" if record["split"] == "validation" else "final_test"
        records.append(record)
    pipeline.write_role_exports(garden.workdir, records)
    directory = garden.workdir / "development"
    cases = [json.loads(line) for line in (directory / "dev.eval_cases.jsonl").read_text().splitlines()]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert {c["case_id"] for c in cases} == {r["record_id"] for r in garden.validation}
    assert all(c["source_kind"] == "development" for c in cases)
    assert manifest["detection"]["cases_sha256"] == case_set_sha256(cases)  # what select_checkpoint checks
    final = [json.loads(line) for line in (garden.workdir / "final_test.eval_cases.jsonl").read_text().splitlines()]
    assert {c["case_id"] for c in final} == {r["record_id"] for r in garden.train}


def test_training_loader_refuses_evaluation_role_records(garden: GardenWorkdir) -> None:
    for record in garden.validation:
        record["evaluation_role"] = "development"
    garden.write()
    with pytest.raises(GovernanceError, match="evaluation role 'development'"):
        load_training_data(garden.config_path, approvals=garden.approvals)


def test_retired_development_stays_usable_for_the_validation_loss(garden: GardenWorkdir) -> None:
    for record in garden.train:
        record["evaluation_role"] = "training"
    for record in garden.validation:
        record["evaluation_role"] = "retired_development"
    garden.write()
    data = load_training_data(garden.config_path, approvals=garden.approvals)
    assert data.validation


def _leak_records(garden: GardenWorkdir) -> list[dict]:
    records = []
    for record in copy.deepcopy(garden.train + garden.validation):
        record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
        if record["split"] == "validation":  # another advisory, so only the checks under test can connect them
            record["provenance"].update(aliases=["CVE-2025-9999"], fixed_revision="e" * 40, vulnerable_revision="d" * 40)
        records.append(record)
    return records


def test_strict_leakage_catches_forks_and_copied_fixes(garden: GardenWorkdir) -> None:
    records = _leak_records(garden)
    train_vulnerable = next(r for r in records if r["split"] == "train" and r["kind"] == "vulnerable")
    held_vulnerable = next(r for r in records if r["split"] == "validation" and r["kind"] == "vulnerable")
    # the held-out repository is a fork: it shares the training record's vulnerable revision
    held_vulnerable["provenance"]["vulnerable_revision"] = train_vulnerable["provenance"]["vulnerable_revision"]
    assert split_leakage(records).get(train_vulnerable["record_id"]) != "shared_commit_with_validation"  # v1-v3: fixed revision only
    assert split_leakage(records, containment=0.8)[train_vulnerable["record_id"]] == "shared_commit_with_validation"

    records = _leak_records(garden)
    train_vulnerable = next(r for r in records if r["split"] == "train" and r["kind"] == "vulnerable")
    held_vulnerable = next(r for r in records if r["split"] == "validation" and r["kind"] == "vulnerable")
    held_vulnerable["target"]["patch"] = train_vulnerable["target"]["patch"]  # the same fix applied in another repository
    assert patch_fingerprint(held_vulnerable) == patch_fingerprint(train_vulnerable)
    assert split_leakage(records, containment=0.8)[train_vulnerable["record_id"]].startswith("shared_patch_with_validation")
