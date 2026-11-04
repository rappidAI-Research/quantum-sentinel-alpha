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


def test_committed_approvals_are_empty_default_deny() -> None:
    assert load_approvals() == {}


def test_unreviewed_repository_is_pending_not_approved() -> None:
    sources = {s["source_id"]: s for s in load_registry()["sources"]}
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
    sources = {s["source_id"]: s for s in load_registry()["sources"]}
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
