"""The pilot rights-review package is a proposal: consistent with the dataset, never an approval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from sentinel.data.registry import load_registry
from sentinel.garden.governance import load_approvals

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "garden" / "seed_v1"
PILOT = SEED / "pilot"
COHORT = json.loads((PILOT / "cohort.json").read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_cohort_is_bound_to_the_committed_manifest_and_queue() -> None:
    assert COHORT["manifest_sha256"] == hashlib.sha256((SEED / "manifest.jsonl").read_bytes()).hexdigest(), "rerun scripts/pilot_cohort.py"
    assert COHORT["review_queue_sha256"] == hashlib.sha256((SEED / "review" / "review_queue.jsonl").read_bytes()).hexdigest()
    rows = {r["record_id"]: r for r in _jsonl(SEED / "manifest.jsonl")}
    queue = {e["repository_key"]: e for e in _jsonl(SEED / "review" / "review_queue.jsonl")}
    assert 30 <= len(COHORT["repositories"]) <= 50
    for repo in COHORT["repositories"]:
        entry = queue[repo["repository_key"]]
        assert entry["tier"] == "A" and entry["license_files_identical_across_revisions"]
        assert repo["license_file_sha256"] == entry["license_files_latest"][0]["sha256"]
        for record_id in repo["record_ids"]:
            row = rows[record_id]
            assert row["provenance"]["repository_key"] == repo["repository_key"]
            assert row["quality"]["status"] == "accepted"
            assert row["split"] in {"train", "validation"}  # never holdout or quarantine
            assert row["governance"]["approval_status"] == "pending_review"


def test_nothing_is_approved_by_the_package() -> None:
    assert load_approvals() == {}  # the active approvals file stays empty (default-deny)
    assert not any(s["production_approved"] for s in load_registry()["sources"])
    with pytest.raises(ValidationError):  # proposed entries are templates and cannot be loaded
        load_approvals(PILOT / "proposed_approvals.yaml")
    stats = json.loads((SEED / "stats.json").read_text())
    assert stats["governance"]["training_approved"] == 0


def test_proposed_approvals_cover_exactly_the_recommended_repositories() -> None:
    import yaml

    proposed = yaml.safe_load((PILOT / "proposed_approvals.yaml").read_text())["repositories"]
    recommended = sorted(r["repository_key"] for r in COHORT["repositories"] if r["proposed_decision"] == "recommended_for_review")
    assert sorted(p["repository_key"] for p in proposed) == recommended
    for entry in proposed:
        assert entry["reviewer"].startswith("<REQUIRED") and entry["decision"].startswith("<REQUIRED")


def test_selection_is_reproducible_from_committed_data() -> None:
    spec = importlib.util.spec_from_file_location("pilot_cohort", ROOT / "scripts" / "pilot_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = module.build(inspect=False)
    assert [r["repository_key"] for r in rebuilt["repositories"]] == [r["repository_key"] for r in COHORT["repositories"]]
    assert [r["proposed_decision"] for r in rebuilt["repositories"]] == [r["proposed_decision"] for r in COHORT["repositories"]]
    assert rebuilt["if_recommended_are_approved"] == COHORT["if_recommended_are_approved"]
