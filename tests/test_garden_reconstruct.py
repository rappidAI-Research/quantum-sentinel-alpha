import copy
import hashlib
import json
import shutil
import uuid

import pytest
import yaml

import sentinel.garden.pipeline as pipeline
from sentinel.garden.build import build_records
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate


@pytest.fixture
def garden(monkeypatch: pytest.MonkeyPatch):
    name = f"test-reconstruct-{uuid.uuid4().hex[:8]}"
    root = pipeline.PROJECT_ROOT / "artifacts" / "garden" / name
    root.mkdir(parents=True)
    raw_config = yaml.safe_load(pipeline.DEFAULT_CONFIG_PATH.read_text())
    raw_config["workdir"] = f"artifacts/garden/{name}/work"
    raw_config["committed_dir"] = f"artifacts/garden/{name}/committed"
    config_path = root / "garden.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    config = pipeline.load_config(config_path)
    raw = []
    for repo, table in (("example/one", "orders"), ("example/two", "invoices")):
        old, new = PY_OLD.replace("orders", table), PY_NEW.replace("orders", table)
        raw += build_records(candidate(repo=repo, advisory="GHSA-2222-3333-4444" if repo.endswith("one") else "GHSA-5555-6666-7777"),
                             FakeRepo({"shop/x.py": old}, {"shop/x.py": new}), SETTINGS, config.dataset_id, "ghsa-osv-fix-commits", ["osv", "github-advisory-db"])
    records = pipeline.curate_records(config, raw)
    config.committed_dir.mkdir(parents=True)
    pipeline._write_jsonl(config.committed_dir / "manifest.jsonl", [pipeline.manifest_row(r) for r in records])
    pipeline._write_json(config.committed_dir / "EXTERNAL_RESOURCES.json", pipeline.external_resources(records))
    yield config, raw
    shutil.rmtree(root, ignore_errors=True)


def committed_hashes(config) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(config.committed_dir.iterdir())}


def test_reconstruct_reports_missing_upstream_without_substitution(garden, monkeypatch) -> None:
    config, raw = garden
    available = [r for r in raw if r["provenance"]["repository_key"] == "example/one"]
    monkeypatch.setattr(pipeline, "build", lambda config, workers=6, log=print: (copy.deepcopy(available), []))
    monkeypatch.setattr(pipeline, "_reachable", lambda url: "example/one" in url)
    before = committed_hashes(config)
    report = pipeline.reconstruct(config, log=lambda _: None)
    assert report["ok"] and report["unreachable_repositories"] == ["example/two"]
    assert len(report["missing_records"]) == len(raw) - len(available)
    assert report["missing_because_unreachable"] == ["example/two"]
    exported = [json.loads(line) for line in (config.workdir / "records.jsonl").read_text().splitlines()]
    assert {r["provenance"]["repository_key"] for r in exported} == {"example/one"}  # nothing substituted
    assert committed_hashes(config) == before  # committed metadata untouched
    for line in (config.workdir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((config.workdir / name).read_bytes()).hexdigest() == digest


def test_reconstruct_fails_on_content_mismatch(garden, monkeypatch) -> None:
    config, raw = garden
    tampered = copy.deepcopy(raw)
    tampered[0]["provenance"]["content_sha256"] = "0" * 64
    monkeypatch.setattr(pipeline, "build", lambda config, workers=6, log=print: (tampered, []))
    monkeypatch.setattr(pipeline, "_reachable", lambda url: True)
    report = pipeline.reconstruct(config, log=lambda _: None)
    assert not report["ok"] and report["content_hash_mismatches"] == [raw[0]["record_id"]]


def test_status_drift_without_missing_records_fails(garden, monkeypatch) -> None:
    config, raw = garden
    monkeypatch.setattr(pipeline, "build", lambda config, workers=6, log=print: (copy.deepcopy(raw), []))
    monkeypatch.setattr(pipeline, "_reachable", lambda url: True)
    rows = [json.loads(line) for line in (config.committed_dir / "manifest.jsonl").read_text().splitlines()]
    rows[0]["split"] = "validation" if rows[0]["split"] != "validation" else "train"
    pipeline._write_jsonl(config.committed_dir / "manifest.jsonl", rows)
    report = pipeline.reconstruct(config, log=lambda _: None)
    assert not report["ok"] and report["split_drift"] == [rows[0]["record_id"]]


def test_split_drift_fails_even_when_upstream_records_are_missing(garden, monkeypatch) -> None:
    config, raw = garden
    available = [r for r in raw if r["provenance"]["repository_key"] == "example/one"]
    monkeypatch.setattr(pipeline, "build", lambda config, workers=6, log=print: (copy.deepcopy(available), []))
    monkeypatch.setattr(pipeline, "_reachable", lambda url: "example/one" in url)
    rows = [json.loads(line) for line in (config.committed_dir / "manifest.jsonl").read_text().splitlines()]
    target = next(r for r in rows if r["provenance"]["repository_key"] == "example/one")
    target["split"] = "validation" if target["split"] != "validation" else "train"
    pipeline._write_jsonl(config.committed_dir / "manifest.jsonl", rows)
    report = pipeline.reconstruct(config, log=lambda _: None)
    assert report["missing_records"] and not report["ok"] and report["split_drift"] == [target["record_id"]]
