"""`materialize`: rebuild only the approved records for a training machine, exactly as the manifest binds them."""

from __future__ import annotations

import copy
import json

import pytest

from sentinel.garden import pipeline
from sentinel.train import data as train_data
from sentinel.train.data import load_training_data
from tests.garden_fixtures import candidate
from tests.train_fixtures import GardenWorkdir, approved_registry

CANDIDATES = [candidate(repo="example/webapp", advisory="GHSA-2222-3333-4444"), candidate(repo="example/other", advisory="GHSA-5555-6666-7777")]


def _raw(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in {"quality", "split", "governance"}} for r in records]


@pytest.fixture
def garden(monkeypatch: pytest.MonkeyPatch):
    workdir = GardenWorkdir()
    workdir.write()
    (workdir.committed / "candidates.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CANDIDATES))
    raw = _raw(workdir.train + workdir.validation)
    built: dict = {"records": raw, "asked": None}

    def fake_build(config, candidates, workers=6, log=print):
        built["asked"] = sorted(c["advisory_id"] for c in candidates)
        return copy.deepcopy(built["records"]), []

    monkeypatch.setattr(pipeline, "build_candidates", fake_build)
    monkeypatch.setattr(pipeline, "load_registry", approved_registry)
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    monkeypatch.setattr(pipeline, "load_approvals", lambda: workdir.approvals)
    for name in pipeline.TRAINING_EXPORT_FILES + ("SHA256SUMS",):
        (workdir.workdir / name).unlink()
    (workdir.workdir / "records.raw.jsonl").write_text("sentinel\n")
    yield workdir, built
    workdir.cleanup()


def _config(garden: GardenWorkdir):
    return pipeline.load_config(garden.config_path)


def test_materialized_exports_load_and_match_the_curated_ones(garden) -> None:
    workdir, built = garden
    report = pipeline.materialize(_config(workdir), log=lambda *a: None)
    assert report["train"] == len(workdir.train) and report["validation"] == len(workdir.validation)
    assert built["asked"] == sorted(c["advisory_id"] for c in CANDIDATES)  # only approved advisories are built
    loaded = load_training_data(workdir.config_path, approvals=workdir.approvals, tasks=("detection", "patch"))
    assert loaded.examples_by_task["train"] == {"detection": len(workdir.train), "patch": 1}
    assert (workdir.workdir / "records.raw.jsonl").read_text() == "sentinel\n"  # curate's input is never touched


def test_rebuilt_code_that_differs_from_the_manifest_is_refused(garden) -> None:
    workdir, built = garden
    built["records"][0]["input"]["code"] += "# drift\n"
    built["records"][0]["provenance"]["content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="content hash"):
        pipeline.materialize(_config(workdir), log=lambda *a: None)


def test_changed_supervision_target_is_refused(garden) -> None:
    workdir, built = garden
    vulnerable = next(r for r in built["records"] if r["kind"] == "vulnerable")
    vulnerable["target"]["evidence"] = "edited explanation"
    with pytest.raises(ValueError, match="target_sha256"):
        pipeline.materialize(_config(workdir), log=lambda *a: None)


def test_missing_upstream_record_is_refused(garden) -> None:
    workdir, built = garden
    built["records"] = built["records"][1:]
    with pytest.raises(ValueError, match="could not be rebuilt"):
        pipeline.materialize(_config(workdir), log=lambda *a: None)


def test_revoked_approval_is_refused(garden, monkeypatch) -> None:
    workdir, _ = garden
    monkeypatch.setattr(pipeline, "load_approvals", lambda: {})
    with pytest.raises(ValueError, match="fails the gate"):
        pipeline.materialize(_config(workdir), log=lambda *a: None)
