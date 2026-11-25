import copy
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from sentinel.data.registry import load_registry
from sentinel.eval.io import validate_cases
from sentinel.garden import render
from sentinel.garden.build import build_records
from sentinel.garden.governance import gate, load_approvals
from sentinel.garden.pipeline import DEFAULT_CONFIG_PATH, curate_records, load_config
from sentinel.garden.split import SplitSettings, assign_split, check_split_invariants, repository_bucket
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate

SPLIT = SplitSettings("salt", 100, 100, "2026-07-01T00:00:00Z")


def built() -> list[dict]:
    return build_records(
        candidate(), FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}), SETTINGS, "sentinel-garden-test",
        "ghsa-osv-fix-commits", ["osv", "github-advisory-db"],
    )


def curated_record(split: str = "train") -> dict:
    record = copy.deepcopy(built()[0])
    record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
    record["split"] = split
    return record


def unapproved_sources() -> dict:
    """The registry as it was before any source decision (gate logic must not depend on today's decisions)."""
    sources = {s["source_id"]: copy.deepcopy(s) for s in load_registry()["sources"]}
    for source_id in ("ghsa-osv-fix-commits", "osv", "github-advisory-db"):
        sources[source_id]["production_approved"] = False
    return sources


def approved_sources() -> dict:
    sources = {s["source_id"]: copy.deepcopy(s) for s in load_registry()["sources"]}
    for source_id in ("ghsa-osv-fix-commits", "osv", "github-advisory-db"):
        sources[source_id]["production_approved"] = True
    return sources


def approval(record: dict, **overrides) -> dict:
    entry = {
        "repository_key": record["provenance"]["repository_key"],
        "decision": "approved_for_training",
        "license_spdx": "MIT",
        "license_file": "LICENSE",
        "license_file_sha256": record["provenance"]["license"]["files"][0]["sha256"],
        "terms_status": "approved_for_project",
        "redistribution_status": "not_redistributed",
        "reviewer": "test-reviewer",
        "reviewed_at": "2026-11-04",
        "evidence_urls": ["https://example.com/LICENSE"],
        "notes": "",
    }
    entry.update(overrides)
    return entry


AI_REVIEWER = ("claude", "codex", "gpt", "assistant", " ai", "ai ", "<required")


def test_committed_approvals_are_documented_human_decisions() -> None:
    """Default-deny holds for everything not listed; every listed entry is a completed human decision
    bound to the license file the rights review recorded, for a repository a training cohort (tier A, or
    the inspected tier B extension) recommended after a clean inspection."""
    import json

    approvals = load_approvals()  # schema-validated
    root = Path(__file__).resolve().parents[1]
    # every clean inspection that recommended a repository, with that version's queue entry (a repository can
    # be recommended in several versions, and a later version may see it relicensed, e.g. canonical/lxd)
    inspections: dict[str, list[tuple[dict, dict]]] = {}
    for version in ("seed_v2", "seed_v3", "seed_v5", "seed_v6"):
        data = root / "data" / "garden" / version
        queue = {json.loads(line)["repository_key"]: json.loads(line) for line in (data / "review" / "review_queue.jsonl").read_text().splitlines() if line.strip()}
        cohort = json.loads((data / "cohort" / "training_cohort.json").read_text())
        for r in cohort["repositories"] + cohort.get("extension_repositories", []):
            if r["proposed_decision"] == "recommended_for_review":
                inspections.setdefault(r["repository_key"], []).append((r, queue[r["repository_key"]]))
    for key, entry in approvals.items():
        reviewer = f" {entry['reviewer'].lower()} "
        assert not any(marker in reviewer for marker in AI_REVIEWER), f"{key}: an AI assistant cannot be the reviewer"
        assert key in inspections, f"{key}: approved without a clean cohort inspection"
        assert any(
            queue["tier"] in {"A", "B"}
            and queue["license_detected"] == entry["license_spdx"]
            and entry["license_file_sha256"] == recommended["license_file_sha256"]
            and set(entry.get("additional_license_file_sha256s", [])) <= set(recommended.get("additional_license_file_sha256s", []))
            for recommended, queue in inspections[key]
        ), f"{key}: the approval does not match the inspection that recommended it"


def test_unreviewed_repository_is_pending_not_approved() -> None:
    sources = unapproved_sources()
    status, blockers = gate(curated_record(), sources, {})
    assert status == "pending_review"
    assert "repository_not_reviewed" in blockers
    assert "source_not_approved:ghsa-osv-fix-commits" in blockers


def test_fully_reviewed_record_is_training_approved() -> None:
    record = curated_record()
    status, blockers = gate(record, approved_sources(), {record["provenance"]["repository_key"]: approval(record)})
    assert (status, blockers) == ("training_approved", [])


def test_approval_is_bound_to_the_reviewed_license_file() -> None:
    record = curated_record()
    entry = approval(record, license_file_sha256="0" * 64)
    status, blockers = gate(record, approved_sources(), {entry["repository_key"]: entry})
    assert status == "pending_review" and blockers == ["repository_approval_license_mismatch"]


def test_repository_approval_alone_does_not_approve_sources() -> None:
    record = curated_record()
    sources = unapproved_sources()
    status, _ = gate(record, sources, {record["provenance"]["repository_key"]: approval(record)})
    assert status == "pending_review"


@pytest.mark.parametrize("split", ["holdout_reserved", "temporal_quarantine"])
def test_holdout_and_quarantine_never_train(split: str) -> None:
    record = curated_record(split)
    status, blockers = gate(record, approved_sources(), {record["provenance"]["repository_key"]: approval(record)})
    assert status == "excluded" and f"split:{split}" in blockers


def test_blocked_repository_and_eval_only_source_are_excluded() -> None:
    record = curated_record()
    status, _ = gate(record, approved_sources(), {record["provenance"]["repository_key"]: approval(record, decision="blocked")})
    assert status == "excluded"
    eval_record = copy.deepcopy(record)
    eval_record["provenance"]["metadata_source_ids"] = ["quickeval-synthetic-v1"]
    status, blockers = gate(eval_record, approved_sources(), {record["provenance"]["repository_key"]: approval(record)})
    assert status == "excluded" and "source_eval_only:quickeval-synthetic-v1" in blockers


def test_rejected_quality_is_excluded() -> None:
    record = curated_record()
    record["quality"]["status"] = "rejected"
    status, _ = gate(record, approved_sources(), {record["provenance"]["repository_key"]: approval(record)})
    assert status == "excluded"


def test_approvals_file_is_schema_checked(tmp_path: Path) -> None:
    record = curated_record()
    path = tmp_path / "approvals.yaml"
    path.write_text(yaml.safe_dump({"approvals_version": "1.0", "repositories": [approval(record, reviewer="")]}))
    with pytest.raises(ValidationError):
        load_approvals(path)
    path.write_text(yaml.safe_dump({"approvals_version": "1.0", "repositories": [approval(record, terms_status="blocked")]}))
    with pytest.raises(ValueError, match="contradicts"):
        load_approvals(path)


def test_repository_split_is_deterministic_and_temporal() -> None:
    assert repository_bucket("a/b", SPLIT) == repository_bucket("a/b", SPLIT)
    buckets = {repository_bucket(f"owner/repo{i}", SPLIT) for i in range(200)}
    assert buckets == {"train", "validation", "holdout_reserved"}
    record = curated_record()
    record["provenance"]["repository_key"] = next(f"o/r{i}" for i in range(200) if repository_bucket(f"o/r{i}", SPLIT) == "train")
    assert assign_split(record, SPLIT) == "train"
    record["provenance"]["fix_committed_at"] = "2026-08-01T00:00:00+00:00"
    assert assign_split(record, SPLIT) == "temporal_quarantine"


def test_split_invariants_detect_leaks() -> None:
    a, b = curated_record("train"), curated_record("validation")
    assert check_split_invariants([a, b], SPLIT) == [f"{a['provenance']['repository_key']}: repository in several splits ['train', 'validation']"]
    late = curated_record("train")
    late["provenance"]["fix_committed_at"] = "2026-09-01T00:00:00Z"
    assert any("after temporal cutoff" in p for p in check_split_invariants([late], SPLIT))


def test_curation_end_to_end_keeps_everything_out_of_training_by_default() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    records = curate_records(config, built())
    assert {r["quality"]["status"] for r in records} == {"accepted"}
    assert all(r["governance"]["approval_status"] != "training_approved" for r in records)


def test_curation_with_full_approval_produces_training_records(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentinel.garden.pipeline as pipeline

    config = load_config(DEFAULT_CONFIG_PATH)
    raw = built()
    approvals = {raw[0]["provenance"]["repository_key"]: approval(raw[0])}
    original = pipeline.load_registry

    def approved_registry() -> dict:
        registry = original()
        for source in registry["sources"]:
            if source["source_id"] in {"ghsa-osv-fix-commits", "osv", "github-advisory-db"}:
                source.update(
                    production_approved=True,
                    revision="0" * 40,
                    license_status="reviewed",
                    terms_status="approved_for_project",
                    redistribution_status="not_redistributed",
                    reviewer="test-reviewer",
                )
        return registry

    monkeypatch.setattr(pipeline, "load_registry", approved_registry)
    records = curate_records(config, raw, approvals=approvals)
    expected = "training_approved" if records[0]["split"] in {"train", "validation"} else "excluded"
    assert {r["governance"]["approval_status"] for r in records} == {expected}


def test_rendering_matches_eval_prompt_and_contract() -> None:
    for record in built():
        example = render.to_chat_example(record)
        assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
        assert record["input"]["code"].splitlines()[0] in example["messages"][1]["content"]
        assert render.reply_matches_contract(record)
    validate_cases([render.to_eval_case(r) for r in built()])


def test_non_permissive_license_needs_written_justification() -> None:
    record = curated_record()
    entry = approval(record, license_spdx="AGPL-3.0")
    status, blockers = gate(record, approved_sources(), {entry["repository_key"]: entry})
    assert status == "pending_review" and blockers == ["non_permissive_license_requires_justification"]
    entry["non_permissive_justification"] = "Counsel memo 2026-12: training-only use of AGPL code is acceptable."
    assert gate(record, approved_sources(), {entry["repository_key"]: entry})[0] == "training_approved"


def test_exports_are_checksummed_and_hold_no_unapproved_training_records(tmp_path: Path) -> None:
    import hashlib

    from sentinel.garden.pipeline import EXPORT_FILES, external_resources, write_exports

    config = load_config(DEFAULT_CONFIG_PATH)
    records = curate_records(config, built())
    write_exports(tmp_path, records)
    assert (tmp_path / "train.jsonl").read_text() == "" and (tmp_path / "train.chat.jsonl").read_text() == ""
    sums = dict(reversed(line.split("  ")) for line in (tmp_path / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == set(EXPORT_FILES)
    for name, digest in sums.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    resources = external_resources(records)
    assert resources["repositories"][0]["commits"] == sorted({"a" * 40, "b" * 40})

    forged = copy.deepcopy(records)
    forged[0]["governance"]["approval_status"] = "training_approved"
    forged[0]["split"] = "holdout_reserved"
    with pytest.raises(AssertionError, match="gate invariant"):
        write_exports(tmp_path, forged)


def test_additional_reviewed_license_versions_cover_records_at_other_revisions() -> None:
    """A license file that changed between revisions (e.g. a copyright year) is covered only by versions the
    reviewer listed; any other version keeps the record pending."""
    record = curated_record()
    entry = approval(record)  # reviewed the license file as it is at this record's revision
    other = "e" * 64
    record["provenance"]["license"]["files"] = [{"path": "LICENSE", "sha256": other}]  # another revision's version
    assert entry["license_file_sha256"] != other
    status, blockers = gate(record, approved_sources(), {record["provenance"]["repository_key"]: entry})
    assert status == "pending_review" and "repository_approval_license_mismatch" in blockers
    entry["additional_license_file_sha256s"] = [other]
    assert gate(record, approved_sources(), {record["provenance"]["repository_key"]: entry})[0] == "training_approved"


def test_additional_license_versions_must_be_sha256(tmp_path: Path) -> None:
    record = curated_record()
    entry = approval(record, additional_license_file_sha256s=["not-a-hash"])
    path = tmp_path / "approvals.yaml"
    path.write_text(yaml.safe_dump({"approvals_version": "1.0", "repositories": [entry]}))
    with pytest.raises(ValidationError):
        load_approvals(path)


def test_approved_repository_with_quarantine_records_exports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Curate applies the training loader's rule: a repository that also has holdout or temporal-quarantine
    records is never exported for training, even when approved (otherwise training would refuse to start)."""
    import sentinel.garden.pipeline as pipeline
    from tests.garden_fixtures import GO_NEW, GO_OLD

    config = load_config(DEFAULT_CONFIG_PATH)
    repo = "example/shop"  # a train-split repository under the frozen policy
    old = build_records(
        candidate(repo=repo), FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}),
        SETTINGS, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv", "github-advisory-db"],
    )
    late = build_records(
        candidate("Go", ["CWE-532"], repo=repo, advisory="GHSA-9999-8888-7777"), FakeRepo({"auth/token.go": GO_OLD}, {"auth/token.go": GO_NEW}),
        SETTINGS, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv", "github-advisory-db"],
    )
    for record in late:
        record["provenance"]["fix_committed_at"] = "2026-09-01T00:00:00+00:00"  # after the frozen temporal cutoff
    approvals = {old[0]["provenance"]["repository_key"]: approval(old[0])}
    monkeypatch.setattr(pipeline, "load_registry", lambda: {"registry_version": "1.0", "sources": list(approved_sources().values())})
    records = curate_records(config, old + late, approvals=approvals)
    splits = {r["record_id"]: r["split"] for r in records}
    assert "train" in splits.values() and "temporal_quarantine" in splits.values()
    trainable = [r for r in records if r["split"] in {"train", "validation"} and r["quality"]["status"] == "accepted"]
    assert trainable and all(r["governance"]["approval_status"] == "excluded" for r in trainable)
    assert all(r["governance"]["blockers"] == ["repository_has_holdout_or_quarantine_records"] for r in trainable)
