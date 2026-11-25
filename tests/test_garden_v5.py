"""Garden v5: pull-request fixes, source-approval coverage, defended siblings, labelled-site evidence,
caller notes and fix-comment removal. Every refinement is config-gated (v1-v4 build as before)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sentinel.garden import pipeline
from sentinel.garden.build import (
    Rejected,
    build_records,
    fix_guards,
    primary_site_evidence,
)
from sentinel.garden.osv import merged_commit, parse_candidate
from sentinel.garden.pulls import PullRequestResolver
from sentinel.garden.quality import check_pairs, check_record, load_record_schema
from tests.garden_fixtures import candidate
from tests.test_garden_discovery import OPTIONS, advisory
from tests.test_garden_v3_multicommit import BASE, A, MultiRepo
from tests.test_garden_v4_extraction import NEW, OLD, V4

V5 = dataclasses.replace(
    V4, multi_function_primary_evidence=True, context_caller_notes=True, defended_siblings=True, strip_fix_security_comments=True
)


def _single(cwes: list[str] | None = None, summary: str | None = None) -> dict:
    cand = candidate(cwes=cwes)
    cand["fix_commit"] = A
    if summary is not None:
        cand["summary"] = summary
    return cand


def _build(old: str, new: str, cand: dict, settings=V5) -> dict:
    repo = MultiRepo({BASE: {"shop/orders.py": old}}, {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": new})})
    return {r["kind"]: r for r in build_records(cand, repo, settings, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv"])}


# --- pull requests -------------------------------------------------------------------------------


def test_merge_and_squash_commits_resolve_a_pull_request() -> None:
    merge = [{"sha": "a" * 40, "message": "Merge pull request #12 from x/fix\n\nFix it"}, {"sha": "b" * 40, "message": "Refs #12 later"}]
    assert merged_commit(12, merge) == {"status": "resolved", "commit": "a" * 40, "kind": "merge_commit", "first_line": "Merge pull request #12 from x/fix"}
    squash = [{"sha": "c" * 40, "message": "Limit email length (#9855)"}]
    assert merged_commit(9855, squash)["kind"] == "squash_or_rebase"
    assert merged_commit(98, squash)["status"] == "not_found"  # #9855 is not #98
    assert merged_commit(5, [{"sha": "d" * 40, "message": "A (#5)"}, {"sha": "e" * 40, "message": "B (#5)"}])["status"] == "ambiguous"


def test_an_advisory_with_one_pull_request_and_no_commit_becomes_a_pull_request_candidate() -> None:
    record = advisory(references=[{"type": "WEB", "url": "https://github.com/go-chi/chi/pull/77"}])
    assert parse_candidate(record, "Go", **OPTIONS)[1] == "no_fix_commit"  # v1-v4
    found, reason = parse_candidate(record, "Go", **OPTIONS, allow_pull_requests=True)
    assert reason == "ok" and found["fix_commit"] is None and found["fix_pull_request"] == {"number": 77}
    two = advisory(references=[{"type": "WEB", "url": "https://github.com/go-chi/chi/pull/77"}, {"type": "WEB", "url": "https://github.com/go-chi/chi/pull/78"}])
    assert parse_candidate(two, "Go", **OPTIONS, allow_pull_requests=True)[1] == "multiple_pull_requests"


def test_resolutions_are_frozen_and_unsearchable_repositories_are_not_retried(tmp_path: Path) -> None:
    calls = []

    def search(repository: str, number: int) -> list[dict]:
        calls.append((repository, number))
        if repository == "gone/repo":
            raise LookupError("HTTP 422")
        return [{"sha": "f" * 40, "message": f"Fix (#{number})"}]

    path = tmp_path / "pull_request_resolutions.jsonl"
    resolver = PullRequestResolver(path, search=search, interval=0, log=lambda _: None)
    assert resolver.resolve("example/webapp", 3)["commit"] == "f" * 40
    assert resolver.resolve("gone/repo", 4)["status"] == "repository_not_searchable"
    again = PullRequestResolver(path, search=lambda *a: pytest.fail("frozen resolutions must not be searched again"), interval=0)
    assert again.resolve("example/webapp", 3)["commit"] == "f" * 40 and len(calls) == 2


# --- governance ----------------------------------------------------------------------------------


def _record(advisory_id: str, kind: str = "vulnerable", status: str = "training_approved") -> dict:
    return {"kind": kind, "provenance": {"advisory_id": advisory_id}, "governance": {"approval_status": status, "blockers": []}}


def test_candidates_outside_the_approved_list_wait_for_the_owner() -> None:
    inside, outside = _record("GHSA-2222-3333-4444"), _record("GHSA-5555-6666-7777")
    for record in (inside, outside):
        pipeline.apply_source_coverage(record, {"GHSA-2222-3333-4444"})
    assert inside["governance"]["approval_status"] == "training_approved"
    assert outside["governance"] == {"approval_status": "pending_review", "blockers": ["source_approval_not_extended"]}


def test_defended_siblings_need_a_validated_kind() -> None:
    sibling = _record("GHSA-2222-3333-4444", kind="defended_sibling")
    pipeline.apply_negative_validation(sibling, {"defended_sibling"})
    assert sibling["governance"] == {"approval_status": "pending_review", "blockers": ["negative_kind_not_validated:defended_sibling"]}
    validated = _record("GHSA-2222-3333-4444", kind="defended_sibling")
    pipeline.apply_negative_validation(validated, set())
    assert validated["governance"]["approval_status"] == "training_approved"


def test_v5_config_keeps_v3_approval_scope() -> None:
    config = pipeline.load_config(Path(__file__).resolve().parents[1] / "configs" / "garden" / "seed_v5.yaml")
    covered = pipeline.source_approved_advisories(config)
    assert covered is not None and len(covered) == 7751
    assert pipeline.unvalidated_negative_kinds(config) == {"defended_sibling"}


# --- labelled site of a combined excerpt ----------------------------------------------------------


def test_a_combined_excerpt_without_evidence_for_one_site_is_rejected() -> None:
    different = NEW.replace("return conn.execute(query, (status,)).fetchone()[0]", "rows = conn.execute(query, (status,))\n    return rows.fetchone()[0]")
    with pytest.raises(Rejected, match="multi_function_site_unevidenced"):
        _build(OLD, different, _single())
    records = _build(OLD, different, _single(summary="SQL injection in count_orders"))
    assert records["vulnerable"]["provenance"]["excerpt_function"] == "count_orders"
    assert records["vulnerable"]["localization"]["primary_evidence"] == "advisory_names_function"


def test_the_same_edit_at_every_site_is_evidence() -> None:
    old = OLD.replace("SELECT count(*) FROM orders WHERE status = '%s'\" % status\n    return conn.execute(query).fetchone()[0]",
                      "SELECT id FROM orders WHERE status = '%s'\" % status\n    rows = conn.execute(query)\n    return len(rows)")
    new = old.replace('''    query = "SELECT id FROM orders WHERE status = '%s'" % status
    rows = conn.execute(query)''', '''    query = "SELECT id FROM orders WHERE status = ?"
    rows = conn.execute(query, (status,))''')
    records = _build(old, new, _single())
    assert records["vulnerable"]["localization"]["primary_evidence"] == "identical_change_at_every_site"


# --- negatives, notes, comments ------------------------------------------------------------------

PATH_OLD = '''import os

from werkzeug.utils import secure_filename


def save_avatar(folder, upload):
    path = os.path.join(folder, upload.filename)
    upload.save(path)
    return path


def save_document(folder, upload):
    name = secure_filename(upload.filename)
    path = os.path.join(folder, name)
    upload.save(path)
    return path


def handler(folder, request):
    return save_avatar(folder, request.files["a"])
'''
PATH_NEW = PATH_OLD.replace('''def save_avatar(folder, upload):
    path = os.path.join(folder, upload.filename)''', '''def save_avatar(folder, upload):
    # prevent path traversal (GHSA-2222-3333-4444)
    path = os.path.join(folder, secure_filename(upload.filename))''')


def test_an_unchanged_function_that_already_applies_the_fix_is_a_defended_sibling() -> None:
    records = _build(PATH_OLD, PATH_NEW, _single(cwes=["CWE-22"]))
    sibling = records["defended_sibling"]
    assert sibling["provenance"]["excerpt_function"] == "save_document" and sibling["target"]["decision"] == "no_finding"
    assert sibling["localization"]["defence_call"] == "secure_filename"
    assert "applies `secure_filename`" in sibling["target"]["label_scope"]
    assert sibling["provenance"]["excerpt_revision"] == sibling["provenance"]["fixed_revision"]
    assert check_record(sibling, Draft202012Validator(load_record_schema()), set(), set(), 1000, 3)[0] == []
    assert "defended_sibling" not in _build(PATH_OLD, PATH_NEW, _single(cwes=["CWE-22"]), V4)


def test_fix_comments_that_name_the_weakness_are_left_out_and_callers_are_named() -> None:
    records = _build(PATH_OLD, PATH_NEW, _single(cwes=["CWE-22"]))
    fixed, vulnerable = records["post_fix"], records["vulnerable"]
    assert "prevent path traversal" not in fixed["input"]["code"] and "prevent path traversal" not in vulnerable["target"]["patch"]
    assert vulnerable["localization"]["diff_consistent"] and vulnerable["localization"]["fix_comment_lines_left_out"] == 1
    assert not check_pairs([vulnerable, fixed])
    assert "Called in this file by: `handler`." in vulnerable["input"]["context"]
    v4 = _build(PATH_OLD, PATH_NEW, _single(cwes=["CWE-22"]), V4)
    assert "prevent path traversal" in v4["post_fix"]["input"]["code"]


def test_guards_must_name_a_defence() -> None:
    assert fix_guards("def f(x):\n    return open(x)\n", ["    x = secure_filename(x)", "    raise ValueError(x)", "    log_attempt(x)"], "CWE-22") == ["secure_filename"]
    assert fix_guards("def f(x):\n    return g(x)\n", ["    user = getUser(x)"], "CWE-862") == []


def test_primary_site_evidence_needs_one_named_function() -> None:
    from types import SimpleNamespace

    def change(name: str, removed: str) -> SimpleNamespace:
        hunk = SimpleNamespace(removed=[(1, removed)], added=[(1, removed + "!")])
        return SimpleNamespace(old_function=SimpleNamespace(name=name, start=1), hunks=[hunk])

    a, b = change("load", "x"), change("store", "y")
    assert primary_site_evidence([a, b], "flaw in load and store", "") is None
    assert primary_site_evidence([a, b], "flaw in store", "")[1] == "advisory_names_function"
    assert json.dumps(primary_site_evidence([a, b], "", "fix load")[1]) == '"commit_message_names_function"'


def test_a_name_shared_by_two_changed_functions_is_no_evidence() -> None:
    from types import SimpleNamespace

    def change(start: int, removed: str) -> SimpleNamespace:
        hunk = SimpleNamespace(removed=[(1, removed)], added=[(1, removed + "!")])
        return SimpleNamespace(old_function=SimpleNamespace(name="ParseBody", start=start), hunks=[hunk])

    assert primary_site_evidence([change(1, "a"), change(20, "b")], "bug in ParseBody", "") is None
    assert primary_site_evidence([change(1, "a"), change(20, "a")], "bug in ParseBody", "")[1] == "identical_change_at_every_site"


def test_block_boundaries_survive_left_out_fix_comments() -> None:
    from sentinel.garden.audit import excerpt_blocks

    new = NEW.replace('''    query = "SELECT count(*) FROM orders WHERE status = ?"''', '''    # prevent SQL injection
    query = "SELECT count(*) FROM orders WHERE status = ?"''')
    fixed = _build(OLD, new, _single(summary="SQL injection in count_orders"))["post_fix"]
    assert "prevent SQL injection" not in fixed["input"]["code"]
    blocks = excerpt_blocks(fixed)
    assert len(blocks) == 2 and blocks[1].startswith("def count_orders")
    assert fixed["localization"]["excerpt_block_lines"] == [4, 3]
