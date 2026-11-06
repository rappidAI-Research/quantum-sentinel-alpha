"""Pilot license inspection: evidence is bound to the exact source state, and a failed read is never absence."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIT_TEXT = """MIT License

Copyright (c) 2020 Example

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("pilot_cohort_inspection", ROOT / "scripts" / "pilot_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source(tmp_path, module, monkeypatch):
    """A cached upstream repo with a vulnerable and a fixed commit, MIT at the root, nothing nested."""
    cache = tmp_path / "cache"
    repo = cache / "example__shop"
    (repo / "shop").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "LICENSE").write_text(MIT_TEXT)
    (repo / "pyproject.toml").write_text('[project]\nname = "shop"\nlicense = "MIT"\n')
    (repo / "shop" / "orders.py").write_text("def f(x):\n    return eval(x)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "vulnerable")
    vulnerable = _git(repo, "rev-parse", "HEAD")
    (repo / "shop" / "orders.py").write_text("def f(x):\n    return int(x)\n")
    _git(repo, "commit", "-qam", "fix")
    fixed = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(module, "CACHE", cache)
    records = [
        {"record_id": f"r-{kind}", "label": {"language": "python"},
         "provenance": {"vulnerable_revision": vulnerable, "fixed_revision": fixed, "file_path": "shop/orders.py"}}
        for kind in ("vulnerable", "post_fix")
    ]
    entry = {"license_detected": "MIT", "license_files_latest": [{"path": "LICENSE"}], "license_files_by_revision": {vulnerable: 1, fixed: 1},
             "registry_licenses": ["MIT"]}
    return {"repo": repo, "records": records, "entry": entry, "vulnerable": vulnerable, "fixed": fixed}


def _run(module, source):
    evidence = module.inspect_repository("example/shop", source["records"], "MIT")
    state = module.source_state(source["records"])
    return evidence, module.assess(source["entry"], evidence, state)


def test_successful_inspection_is_complete_and_recommended(module, source) -> None:
    evidence, (decision, confirmed, questions) = _run(module, source)
    assert decision == "recommended_for_review" and questions == []
    assert evidence["problems"] == []
    assert {d["inspection"] for d in evidence["revisions"].values()} == {"complete"}
    assert evidence["revisions"][source["fixed"]]["package_metadata"] == [{"file": "pyproject.toml", "license": "MIT"}]


def test_nested_notice_is_found_and_needs_clarification(module, source) -> None:
    (source["repo"] / "shop" / "NOTICE").write_text("Portions derived from another project under other terms.\n")
    _git(source["repo"], "add", "-A")
    _git(source["repo"], "commit", "-qm", "notice")
    head = _git(source["repo"], "rev-parse", "HEAD")
    for record in source["records"]:
        record["provenance"]["fixed_revision"] = head
    evidence, (decision, _, questions) = _run(module, source)
    assert decision == "needs_clarification" and any("shop/NOTICE" in q for q in questions)
    assert evidence["revisions"][head]["inspection"] == "complete"  # found, not failed


def test_partial_cache_missing_commit_is_incomplete(module, source) -> None:
    for record in source["records"]:
        record["provenance"]["fixed_revision"] = "f" * 40  # not in the cache
    evidence, (decision, _, questions) = _run(module, source)
    assert evidence["revisions"]["f" * 40]["inspection"] == "incomplete"
    assert decision == "needs_clarification" and any("cannot read commit ffffffffffff" in q for q in questions)


def _failing(module, monkeypatch, predicate):
    real = module._git

    def git(repo, *args):
        if predicate(args):
            raise RuntimeError("fatal: could not fetch object (partial clone)")
        return real(repo, *args)

    monkeypatch.setattr(module, "_git", git)


def test_inaccessible_nested_tree_is_recorded_not_absence(module, source, monkeypatch) -> None:
    fixed = source["fixed"]
    _failing(module, monkeypatch, lambda a: a[0] == "ls-tree" and a[1] == f"{fixed}:shop")
    evidence, (decision, _, questions) = _run(module, source)
    assert evidence["revisions"][fixed]["inspection"] == "incomplete"
    assert evidence["revisions"][source["vulnerable"]]["inspection"] == "complete"
    assert any(f"cannot inspect shop at {fixed}" in p for p in evidence["problems"])
    assert decision == "needs_clarification"


def test_missing_license_blob_is_recorded_not_absence(module, source, monkeypatch) -> None:
    fixed = source["fixed"]
    _failing(module, monkeypatch, lambda a: a[0] == "show" and a[1] == f"{fixed}:LICENSE")
    evidence, (decision, _, questions) = _run(module, source)
    assert evidence["revisions"][fixed]["root_notice_and_license_files"]["LICENSE"]["read_error"] is True
    assert any(f"cannot read LICENSE at {fixed}" in q for q in questions) and decision == "needs_clarification"


def test_unreadable_package_metadata_is_recorded(module, source, monkeypatch) -> None:
    fixed = source["fixed"]
    _failing(module, monkeypatch, lambda a: a[0] == "show" and a[1] == f"{fixed}:pyproject.toml")
    evidence, (decision, _, _) = _run(module, source)
    assert any(f"cannot read pyproject.toml at {fixed}" in p for p in evidence["problems"]) and decision == "needs_clarification"


def test_valid_unchanged_cached_evidence_is_reused(module, source) -> None:
    evidence, _ = _run(module, source)
    cached = json.loads(json.dumps(evidence))  # as round-tripped through cohort.json
    state = module.source_state(source["records"])
    assert module.stale_reason(cached, state) is None
    assert module.assess(source["entry"], cached, state)[0] == "recommended_for_review"


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r, s: r[0]["provenance"].update(fixed_revision="a" * 40), "other records, revisions or paths"),  # changed commit
        (lambda r, s: [x["provenance"].update(file_path="shop/other.py") for x in r], "other records, revisions or paths"),  # changed path
        (lambda r, s: r.append({**copy.deepcopy(r[0]), "record_id": "r-new"}), "other records, revisions or paths"),  # changed record set
    ],
)
def test_changed_source_state_invalidates_cached_evidence(module, source, change, message) -> None:
    cached, _ = _run(module, source)
    change(source["records"], source)
    state = module.source_state(source["records"])
    assert message in module.stale_reason(cached, state)
    decision, _, questions = module.assess(source["entry"], cached, state)
    assert decision == "needs_clarification" and questions == [module.stale_reason(cached, state)]


@pytest.mark.parametrize(
    "damage",
    [
        lambda e: e.pop("source_state_sha256"),  # evidence from before the binding existed
        lambda e: next(iter(e["revisions"].values())).pop("inspection"),
        lambda e: next(iter(e["revisions"].values())).pop("package_metadata"),
        lambda e: e["revisions"].popitem(),
        lambda e: e.pop("problems"),
    ],
)
def test_incomplete_cached_evidence_is_not_reused(module, source, damage) -> None:
    cached, _ = _run(module, source)
    damage(cached)
    state = module.source_state(source["records"])
    assert module.stale_reason(cached, state)
    assert module.assess(source["entry"], cached, state)[0] == "needs_clarification"


@pytest.mark.parametrize("name", ["LICENSE.GPL", "LICENSE-MIT", "COPYING.LESSER", "licence_apache"])
def test_suffixed_nested_license_files_are_inspected(module, source, name) -> None:
    # review finding: a nested vendor/LICENSE.GPL under a root MIT license must become review evidence
    repo = source["repo"]
    (repo / "shop" / name).write_text("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "nested license")
    head = _git(repo, "rev-parse", "HEAD")
    for record in source["records"]:
        record["provenance"]["fixed_revision"] = head
    evidence, (decision, _, questions) = _run(module, source)
    assert f"shop/{name}" in evidence["revisions"][head]["license_files_on_record_paths"]
    assert decision == "needs_clarification" and any(f"shop/{name}" in q for q in questions)


def test_suffixed_copy_of_the_root_license_is_not_a_question(module, source) -> None:
    repo = source["repo"]
    (repo / "shop" / "LICENSE-MIT").write_text(MIT_TEXT)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "copy")
    head = _git(repo, "rev-parse", "HEAD")
    for record in source["records"]:
        record["provenance"]["fixed_revision"] = head
    evidence, (decision, _, _) = _run(module, source)
    assert evidence["revisions"][head]["license_files_on_record_paths"]["shop/LICENSE-MIT"]["detected"] == "MIT"
    assert decision == "recommended_for_review"
