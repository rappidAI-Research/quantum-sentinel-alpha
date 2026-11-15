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
    if not load_approvals():  # until an authorized reviewer decides: default-deny
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


def _module():
    spec = importlib.util.spec_from_file_location("pilot_cohort", ROOT / "scripts" / "pilot_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("field", ["revisions", "paths", "source_state"])
def test_stale_evidence_is_not_reused(tmp_path, monkeypatch, field) -> None:
    # review finding: cached evidence must cover exactly the revisions and paths the records use
    module = _module()
    cohort = json.loads(json.dumps(COHORT))
    target = next(r for r in cohort["repositories"] if r["proposed_decision"] == "recommended_for_review")
    if field == "revisions":
        target["evidence"]["revisions"].popitem()
    elif field == "paths":
        target["evidence"]["paths"] = target["evidence"]["paths"][:-1] + ["other/file.py"]
    else:
        target["evidence"]["source_state_sha256"] = "0" * 64
    (tmp_path / "cohort.json").write_text(json.dumps(cohort))
    monkeypatch.setattr(module, "OUT", tmp_path)
    rebuilt = {r["repository_key"]: r for r in module.build(inspect=False)["repositories"]}
    entry = rebuilt[target["repository_key"]]
    assert entry["proposed_decision"] == "needs_clarification"
    assert any("run with --inspect" in q for q in entry["open_questions"])
    assert entry["evidence"] is None  # stale evidence is dropped, never carried forward


def test_inspection_failures_are_open_questions_not_absence() -> None:
    # review finding: a failed nested read must never count as "no other license file"
    module = _module()
    queue = {e["repository_key"]: e for e in _jsonl(SEED / "review" / "review_queue.jsonl")}
    repo = next(r for r in COHORT["repositories"] if r["proposed_decision"] == "recommended_for_review")
    evidence = json.loads(json.dumps(repo["evidence"]))
    assert module.assess(queue[repo["repository_key"]], evidence)[0] == "recommended_for_review"
    evidence["problems"].append("cannot inspect pkg at 0123456789ab: fatal: bad object")
    decision, _, questions = module.assess(queue[repo["repository_key"]], evidence)
    assert decision == "needs_clarification" and any("cannot inspect pkg" in q for q in questions)


def test_committed_evidence_covers_the_current_source_state() -> None:
    module = _module()
    rows = _jsonl(SEED / "manifest.jsonl")
    by_repo: dict[str, list[dict]] = {}
    for row in rows:
        by_repo.setdefault(row["provenance"]["repository_key"], []).append(row)
    for repo in COHORT["repositories"]:
        records = [r for r in by_repo[repo["repository_key"]] if r["record_id"] in set(repo["record_ids"])]
        state = module.source_state(records)
        assert module.stale_reason(repo["evidence"], state) is None, repo["repository_key"]
        if repo["proposed_decision"] == "recommended_for_review":
            assert repo["evidence"]["problems"] == []
            assert {d["inspection"] for d in repo["evidence"]["revisions"].values()} == {"complete"}
