"""Tier B extension of the training cohort: formal questions are classified, substantive ones stay open."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

from tests.train_fixtures import _records

ROOT = Path(__file__).resolve().parents[1]
MIT = "MIT License\n\nCopyright (c) {year} Example\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n"
APACHE = "                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("training_cohort_ext", ROOT / "scripts" / "training_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "registry, expected",
    [
        ({"npm:x": ["MIT"]}, "agrees"),
        ({"npm:x": None}, "unavailable"),
        ({"Go:a": ["Apache-2.0", "MIT", "Unlicense"], "Go:b": ["MIT", "Unlicense"]}, "lists_root_and_others"),  # bundled code
        ({"npm:x": ["AGPL-3.0-only"]}, "names_other_license"),
        ({"Go:a": ["MIT"], "Go:b": ["Apache-2.0"]}, "names_other_license"),  # one package lacks the root license
    ],
)
def test_registry_findings(module, registry, expected) -> None:
    assert module.registry_finding("MIT", registry) == expected


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def licensed_repo(tmp_path, module, monkeypatch):
    repo = tmp_path / "cache" / "example__webapp"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    revisions = []
    for text in (MIT.format(year=2020), MIT.format(year=2024), APACHE):
        (repo / "LICENSE").write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "x")
        revisions.append((_git(repo, "rev-parse", "HEAD"), hashlib.sha256(text.encode()).hexdigest()))
    monkeypatch.setattr(module.pilot, "CACHE", tmp_path / "cache")
    return revisions


def _entry(revisions):
    return {"license_files_by_revision": {rev: [{"path": "LICENSE", "sha256": sha}] for rev, sha in revisions}}


def test_license_versions_of_the_same_license_are_listed(module, licensed_repo) -> None:
    result = module.license_variants("example/webapp", _entry(licensed_repo[:2]), "LICENSE")
    assert set(result["variants"].values()) == {"MIT"} and len(result["variants"]) == 2


def test_a_license_change_in_substance_is_visible(module, licensed_repo) -> None:
    result = module.license_variants("example/webapp", _entry(licensed_repo), "LICENSE")
    assert set(result["variants"].values()) == {"MIT", "Apache-2.0"}


def test_wrong_recorded_hash_or_missing_file_is_a_problem(module, licensed_repo) -> None:
    rev, _ = licensed_repo[0]
    assert "does not match" in module.license_variants("example/webapp", {"license_files_by_revision": {rev: [{"path": "LICENSE", "sha256": "0" * 64}]}}, "LICENSE")["problem"]
    assert "missing" in module.license_variants("example/webapp", {"license_files_by_revision": {rev: [{"path": "COPYING", "sha256": "0" * 64}]}}, "LICENSE")["problem"]


def test_projected_volume_covers_every_reviewed_license_version(module) -> None:
    from sentinel.garden.pipeline import manifest_row
    from sentinel.garden.supervision import pairs_by_id

    old, new = _records("example/webapp", "GHSA-2222-3333-4444", "train"), _records("example/webapp", "GHSA-5555-6666-7777", "train")
    for record in new:
        record["record_id"] += "-new"
        record["group_id"] += "-new"
        record["pair_id"] = record["pair_id"] + "-new" if record["pair_id"] else None
        record["provenance"]["license"]["files"] = [{"path": "LICENSE", "sha256": "e" * 64}]  # a later version of the file
    rows = [manifest_row(r, pairs_by_id(old + new)) for r in old + new]
    entry = {"repository_key": "example/webapp", "license_spdx_detected": "MIT", "license_file": "LICENSE",
             "license_file_sha256": old[0]["provenance"]["license"]["files"][0]["sha256"]}
    assert module.projected_volume([entry], rows)["records"] == len(old)
    entry["additional_license_file_sha256s"] = ["e" * 64]
    assert module.projected_volume([entry], rows)["records"] == len(old) + len(new)


def test_unlicense_metadata_matches_an_unlicense_root(module) -> None:
    assert module.pilot.normalize_license("unlicense") == "Unlicense"
    assert module.pilot.normalize_license("(MIT AND BSD-3-Clause)") == "(MIT AND BSD-3-Clause)"  # a conjunction stays a question
