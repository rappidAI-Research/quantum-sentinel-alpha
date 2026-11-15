"""Garden v2 training cohort: a proposal bound to the committed dataset, never an approval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from sentinel.data.registry import load_registry
from sentinel.garden import pipeline
from sentinel.garden.governance import load_approvals

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs" / "garden" / "seed_v1.yaml"
V2 = ROOT / "configs" / "garden" / "seed_v2.yaml"
DATA = ROOT / "data" / "garden" / "seed_v2"
COHORT = json.loads((DATA / "cohort" / "training_cohort.json").read_text())
TIER_A = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
APPROVALS = load_approvals()  # schema-validated; empty until an authorized reviewer decides


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _module():
    spec = importlib.util.spec_from_file_location("training_cohort", ROOT / "scripts" / "training_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config", [V1, V2])
def test_committed_garden_versions_pass_the_offline_check(config) -> None:
    result = pipeline.check(pipeline.load_config(config))
    assert result["training_approved"] == 0


def test_garden_v1_is_a_subset_of_v2_with_identical_content_and_splits() -> None:
    v1 = {r["record_id"]: r for r in _jsonl(ROOT / "data" / "garden" / "seed_v1" / "manifest.jsonl")}
    v2 = {r["record_id"]: r for r in _jsonl(DATA / "manifest.jsonl")}
    assert set(v1) <= set(v2)
    for record_id, row in v1.items():
        assert v2[record_id]["provenance"]["content_sha256"] == row["provenance"]["content_sha256"]
        assert v2[record_id]["split"] == row["split"]  # no holdout/validation repository moves


def test_cohort_is_bound_to_the_committed_v2_manifest() -> None:
    assert COHORT["manifest_sha256"] == hashlib.sha256((DATA / "manifest.jsonl").read_bytes()).hexdigest(), "rerun scripts/training_cohort.py"
    assert COHORT["review_queue_sha256"] == hashlib.sha256((DATA / "review" / "review_queue.jsonl").read_bytes()).hexdigest()
    rows = {r["record_id"]: r for r in _jsonl(DATA / "manifest.jsonl")}
    queue = {e["repository_key"]: e for e in _jsonl(DATA / "review" / "review_queue.jsonl")}
    tier_a = {k for k, e in queue.items() if e["tier"] == "A"}
    held_out = {r["provenance"]["repository_key"] for r in rows.values() if r["split"] in {"holdout_reserved", "temporal_quarantine"}}
    # every tier A repository the training loader can accept, nothing else; the others are listed as excluded
    assert {r["repository_key"] for r in COHORT["repositories"]} == tier_a - held_out
    assert {e["repository_key"] for e in COHORT["excluded"]} == tier_a & held_out
    module = _module()
    for repo in COHORT["repositories"]:
        entry = queue[repo["repository_key"]]
        assert entry["license_detected"] in TIER_A and entry["license_files_identical_across_revisions"]
        assert repo["license_file_sha256"] == entry["license_files_latest"][0]["sha256"]
        records = [rows[i] for i in repo["record_ids"]]
        for row in records:
            assert row["provenance"]["repository_key"] == repo["repository_key"]
            assert row["quality"]["status"] == "accepted" and row["split"] in {"train", "validation"}
            # proposals never approve; only an authorized entry in approvals.yaml can
            expected = {"pending_review", "training_approved"} if repo["repository_key"] in APPROVALS else {"pending_review"}
            assert row["governance"]["approval_status"] in expected
        state = module.pilot.source_state(records)
        if repo["proposed_decision"] == "recommended_for_review":
            assert module.pilot.stale_reason(repo["evidence"], state) is None
            assert repo["evidence"]["problems"] == [] and repo["open_questions"] == []
            assert {d["inspection"] for d in repo["evidence"]["revisions"].values()} == {"complete"}
        else:
            assert repo["open_questions"]


def test_nothing_is_approved_by_the_proposal() -> None:
    if not APPROVALS:  # the state this package was prepared in: default-deny
        assert not any(s["production_approved"] for s in load_registry()["sources"])
        assert json.loads((DATA / "stats.json").read_text())["governance"]["training_approved"] == 0
    with pytest.raises(ValidationError):
        load_approvals(DATA / "cohort" / "proposed_approvals.yaml")
    proposed = yaml.safe_load((DATA / "cohort" / "proposed_approvals.yaml").read_text())["repositories"]
    recommended = sorted(r["repository_key"] for r in COHORT["repositories"] if r["proposed_decision"] == "recommended_for_review")
    assert sorted(p["repository_key"] for p in proposed) == recommended
    assert all(p["reviewer"].startswith("<REQUIRED") and p["decision"].startswith("<REQUIRED") for p in proposed)


def test_cohort_rebuilds_offline_from_committed_data() -> None:
    rebuilt = _module().build(V2, inspect=False)
    assert [(r["repository_key"], r["proposed_decision"]) for r in rebuilt["repositories"]] == [
        (r["repository_key"], r["proposed_decision"]) for r in COHORT["repositories"]
    ]
    assert rebuilt["if_recommended_are_approved"] == COHORT["if_recommended_are_approved"]
    assert rebuilt["decision_groups"] == COHORT["decision_groups"]


def test_inspection_reads_the_selected_datasets_own_cache() -> None:
    # review finding: the inspector used Garden v1's cache path for every dataset version
    module = _module()
    module.build(V2, inspect=False)
    assert module.pilot.CACHE == pipeline.load_config(V2).workdir / "cache" / "git"
