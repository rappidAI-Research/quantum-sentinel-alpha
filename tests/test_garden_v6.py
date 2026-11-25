"""Garden v6: MoreFixes discovery index, CodeQL query-test records, external benchmarks, tool-test split."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sentinel.garden import morefixes, querytests
from sentinel.garden.pipeline import (
    apply_external_benchmark,
    load_config,
    unvalidated_negative_kinds,
)
from sentinel.garden.split import SplitSettings, assign_split

ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


# ---------------------------------------------------------------------------------------------
# MoreFixes dump parsing
# ---------------------------------------------------------------------------------------------


def test_copy_rows_reads_wanted_columns_and_unescapes() -> None:
    lines = [
        "CREATE TABLE public.fixes (...);\n",
        "COPY public.fixes (cve_id, hash, repo_url, rel_type, score, extraction_status) FROM stdin;\n",
        "CVE-2024-0001\t" + SHA_A + "\thttps://github.com/o/r\tdirect\t1337\tCOMPLETED\n",
        "\\.\n",
        "COPY public.commits (hash, repo_url, author, committer, msg, parents, merge, committer_date) FROM stdin;\n",
        SHA_A + "\thttps://github.com/o/r\tAlice\tAlice\tfix: escape\\n\\ttab \\\\ back\t{x}\tf\t2024-01-01 00:00:00+00\n",
        "\\.\n",
        "COPY public.users (id) FROM stdin;\n",
        "secret\n",
        "\\.\n",
    ]
    rows = list(morefixes.copy_rows(iter(lines)))
    assert rows[0] == ("fixes", {"cve_id": "CVE-2024-0001", "hash": SHA_A, "repo_url": "https://github.com/o/r", "rel_type": "direct", "score": "1337"})
    table, commit = rows[1]
    assert table == "commits" and commit["msg"] == "fix: escape\n\ttab \\ back" and "author" not in commit
    assert len(rows) == 2  # unwanted tables (users) are skipped entirely


def test_null_and_reference_helpers() -> None:
    assert morefixes._unescape("\\N") is None
    urls = [f"https://github.com/Owner/Repo/commit/{SHA_A[:10]}", "https://github.com/advisories/GHSA-2234-5678-9cfg"]
    assert morefixes.referenced_commit(urls, "owner/repo", SHA_A)
    assert not morefixes.referenced_commit(urls, "owner/other", SHA_A)
    assert not morefixes.referenced_commit(urls, "owner/repo", SHA_B)
    assert morefixes.ghsa_aliases(urls) == ["GHSA-2234-5678-9cfg"]
    assert morefixes.iso_time("2021-05-11 15:15:08.123+00") == "2021-05-11T15:15:08Z"
    blurb = "Acme is a framework for building apps. A stored XSS vulnerability in the title field allows script injection. Upgrade to 2.0."
    assert morefixes.summary_of(blurb) == "A stored XSS vulnerability in the title field allows script injection."
    vuldb = "A vulnerability has been found in acme up to 0.3 and classified as problematic. This affects the file main.js. The manipulation leads to cross site scripting."
    assert morefixes.summary_of(vuldb) == "The manipulation leads to cross site scripting."


def _index(fixes: list[dict[str, Any]], cves: list[dict[str, Any]], cwes: dict[str, list[str]], changes: dict[str, list[str]]) -> dict[str, Any]:
    index: dict[str, Any] = {"cves": {c["cve_id"]: c for c in cves}, "cwes": cwes, "fixes": {}, "commits": {}, "changes": {}, "repositories": {}}
    for row in fixes:
        index["fixes"].setdefault(row["cve_id"], []).append(row)
        key = morefixes.repository_key(row["repo_url"])
        index["commits"][(key, row["hash"])] = {"hash": row["hash"], "repo_url": row["repo_url"], "msg": "m", "parents": "{}", "merge": "f"}
    for sha, paths in changes.items():
        index["changes"][sha] = [{"hash": sha, "new_path": p, "old_path": p} for p in paths]
    return index


def _cve(cve_id: str, urls: list[str]) -> dict[str, Any]:
    return {
        "cve_id": cve_id,
        "published_date": "2022-01-01T00:00:00",
        "last_modified_date": "2022-02-01T00:00:00",
        "description": "Example allows SQL injection via the name parameter of the search endpoint before 2.0. More text.",
        "reference_json": json.dumps([{"url": u} for u in urls]),
    }


SETTINGS = {"published_before": "2026-11-01T00:00:00Z", "max_candidates_per_repository": 40, "max_fix_commits": 3}


def test_discover_candidates_requires_nvd_reference_and_never_duplicates() -> None:
    repo = "https://github.com/acme/app"
    frozen = [{"advisory_id": "GHSA-2234-5678-9cfg", "aliases": ["CVE-2022-0001"], "repository_key": "acme/known", "fix_commit": SHA_C, "packages": []}]
    fixes = [
        {"cve_id": "CVE-2022-0001", "hash": SHA_A, "repo_url": repo},  # alias of a frozen advisory
        {"cve_id": "CVE-2022-0002", "hash": SHA_A, "repo_url": repo},  # referenced by NVD -> candidate
        {"cve_id": "CVE-2022-0003", "hash": SHA_B, "repo_url": repo},  # not referenced: heuristic only
        {"cve_id": "CVE-2022-0004", "hash": SHA_C, "repo_url": "https://github.com/acme/known"},  # frozen fix commit
        {"cve_id": "CVE-2022-0005", "hash": SHA_B, "repo_url": repo},  # rejected CWE
    ]
    cves = [
        _cve("CVE-2022-0001", [f"{repo}/commit/{SHA_A}"]),
        _cve("CVE-2022-0002", [f"{repo}/commit/{SHA_A}"]),
        _cve("CVE-2022-0003", ["https://example.com/advisory"]),
        _cve("CVE-2022-0004", [f"https://github.com/acme/known/commit/{SHA_C}"]),
        _cve("CVE-2022-0005", [f"{repo}/commit/{SHA_B}"]),
    ]
    cwes = {c["cve_id"]: ["CWE-89"] for c in cves} | {"CVE-2022-0005": ["CWE-284"]}
    index = _index(fixes, cves, cwes, {SHA_A: ["src/app.py"], SHA_B: ["src/app.py"], SHA_C: ["lib/x.js"]})
    added, outcomes, records, corroborations = morefixes.discover_candidates(index, frozen, SETTINGS, {"CWE-284"}, 2, "salt")
    assert [c["advisory_id"] for c in added] == ["CVE-2022-0002"]
    candidate = added[0]
    assert candidate["fix_commit"] == SHA_A and candidate["ecosystem"] == "GIT" and candidate["fix_evidence"] == "nvd_reference"
    assert candidate["summary"] == "Example allows SQL injection via the name parameter of the search endpoint before 2.0."
    assert outcomes["fix_commit_heuristic_only"] == 1 and outcomes["rejected_cwe"] == 1
    assert {(c["cve_id"], c["kind"]) for c in corroborations} == {("CVE-2022-0001", "same_vulnerability"), ("CVE-2022-0004", "same_fix_commit")}
    assert set(records) == {"CVE-2022-0002"}


def test_discover_candidates_skips_non_target_language_and_test_only_changes() -> None:
    repo = "https://github.com/acme/app"
    fixes = [{"cve_id": "CVE-2022-0010", "hash": SHA_A, "repo_url": repo}, {"cve_id": "CVE-2022-0011", "hash": SHA_B, "repo_url": repo}]
    cves = [_cve("CVE-2022-0010", [f"{repo}/commit/{SHA_A}"]), _cve("CVE-2022-0011", [f"{repo}/commit/{SHA_B}"])]
    index = _index(fixes, cves, {c["cve_id"]: ["CWE-79"] for c in cves}, {SHA_A: ["src/Main.java"], SHA_B: ["tests/test_app.py"]})
    added, outcomes, _, _ = morefixes.discover_candidates(index, [], SETTINGS, set(), 2, "salt")
    assert added == [] and outcomes == {"no_target_language_change": 2}


def test_repository_vulns_lists_every_cve_with_a_fix_commit_reference() -> None:
    repo = "https://github.com/acme/app"
    fixes = [{"cve_id": "CVE-2022-0020", "hash": SHA_A, "repo_url": repo}, {"cve_id": "CVE-2022-0021", "hash": SHA_B, "repo_url": "https://github.com/other/x"}]
    index = _index(fixes, [_cve("CVE-2022-0020", [])], {}, {})
    vulns = morefixes.repository_vulns(index, "acme/app")
    assert [v["id"] for v in vulns] == ["CVE-2022-0020"]
    assert {"type": "FIX", "url": f"https://github.com/acme/app/commit/{SHA_A}"} in vulns[0]["references"]


def test_extract_index_keeps_target_language_commits_only(tmp_path: Path) -> None:
    dump = "\n".join(
        [
            "COPY public.commits (hash, repo_url, msg, parents, merge, committer_date) FROM stdin;",
            f"{SHA_A}\thttps://github.com/o/r\tfix\t{{}}\tf\t2024-01-01",
            f"{SHA_B}\thttps://github.com/o/r\tfix\t{{}}\tf\t2024-01-01",
            "\\.",
            "COPY public.file_change (file_change_id, hash, old_path, new_path, change_type, num_lines_added, num_lines_deleted, code_before) FROM stdin;",
            f"1\t{SHA_A}\tsrc/a.py\tsrc/a.py\tModify\t1\t1\tBIG CODE",
            f"2\t{SHA_B}\tsrc/a.c\tsrc/a.c\tModify\t1\t1\tBIG CODE",
            "\\.",
            "",
        ]
    )
    path = tmp_path / "dump.sql.gz"
    with gzip.open(path, "wt") as f:
        f.write(dump)
    summary = morefixes.extract_index(path, tmp_path / "index", log=lambda *a: None)
    assert summary["target_language_commits"] == 1
    assert [r["hash"] for r in morefixes.read_table(tmp_path / "index", "commits")] == [SHA_A]
    assert all("code_before" not in r for r in morefixes.read_table(tmp_path / "index", "file_change"))


# ---------------------------------------------------------------------------------------------
# CodeQL query tests
# ---------------------------------------------------------------------------------------------


QUERY = """/**
 * @name SQL query built from user-controlled sources
 * @kind path-problem
 * @precision high
 * @id py/sql-injection
 * @tags security
 *       external/cwe/cwe-089
 */
"""

TEST_PY = '''from flask import request
import db


def show(name):  # $ Source
    """Docstring that says this is BAD."""
    q = "SELECT * FROM t WHERE n = '%s'" % name
    db.execute(q)  # $ Alert


def lookup(name):  # $ Source
    # GOOD: parameterized
    db.execute("SELECT * FROM t WHERE n = %s", (name,))
    return 1


def count(name):
    db.execute("SELECT count(*) FROM t WHERE n = '%s'" % name)  # $ MISSING: Alert
    return 2


def bad_query(name):  # $ Source
    db.execute("SELECT %s" % name)  # $ Alert
'''

EXPECTED = """#select
| test.py:8:16:8:16 | ControlFlowNode for q | test.py:5:10:5:13 | ControlFlowNode for name | test.py:8:16:8:16 | ControlFlowNode for q | This SQL query depends on a $@. | test.py:5:10:5:13 | ControlFlowNode for name | user-provided value |
| test.py:23:16:23:30 | ControlFlowNode for BinaryExpr | test.py:22:15:22:18 | ControlFlowNode for name | test.py:23:16:23:30 | ControlFlowNode for BinaryExpr | This SQL query depends on a $@. | test.py:22:15:22:18 | ControlFlowNode for name | user-provided value |
edges
"""


def _codeql_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "codeql"
    test_dir = root / "python/ql/test/query-tests/Security/CWE-089-SqlInjection"
    test_dir.mkdir(parents=True)
    (root / "python/ql/src/Security/CWE-089").mkdir(parents=True)
    (root / "python/ql/src/Security/CWE-089/SqlInjection.ql").write_text(QUERY)
    (test_dir / "SqlInjection.qlref").write_text("query: Security/CWE-089/SqlInjection.ql\npostprocess: utils/test/InlineExpectationsTestQuery.ql\n")
    (test_dir / "SqlInjection.expected").write_text(EXPECTED)
    (test_dir / "test.py").write_text(TEST_PY)
    (root / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "t"], check=True)
    return root


def test_parse_expected_resolves_placeholders_and_sources() -> None:
    results = querytests.parse_expected(EXPECTED, "path-problem")
    assert [(r.file, r.line, r.source_line) for r in results] == [("test.py", 8, 5), ("test.py", 23, 22)]
    assert results[0].message == "This SQL query depends on a user-provided value."


def test_markers_respect_query_tags() -> None:
    assert querytests._markers("x  # $ Alert[py/sql-injection]", "py/sql-injection") == {"Alert"}
    assert querytests._markers("x  # $ Alert[py/other]", "py/sql-injection") == set()
    assert querytests._markers("x  # $ SPURIOUS: Alert", "py/sql-injection") == {"SPURIOUS Alert"}


def test_query_test_records_are_scoped_and_markers_removed(tmp_path: Path) -> None:
    root = _codeql_checkout(tmp_path)
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    config = load_config(ROOT / "configs/garden/seed_v6.yaml")
    records, log = querytests.build_records(root, revision, config.extraction, "sentinel-garden-v6", set(), 2, log=lambda *a: None)
    by_function = {r["provenance"]["excerpt_function"]: r for r in records}
    # `count` has a MISSING marker (a known false negative) and `bad_query` names its label: both left out
    assert set(by_function) == {"show", "lookup"}
    finding, negative = by_function["show"], by_function["lookup"]
    assert finding["kind"] == "query_test_finding" and finding["target"]["cwe"] == "CWE-89"
    assert finding["target"]["evidence"] == "This SQL query depends on a user-provided value."
    lines = finding["input"]["code"].splitlines()
    assert lines[finding["target"]["start_line"] - 1].strip() == "db.execute(q)"
    assert negative["kind"] == "query_test_negative" and negative["provenance"]["negative_basis"] == "source_reaches_no_sink"
    assert "py/sql-injection" in negative["target"]["label_scope"]
    for record in records:
        text = record["input"]["code"] + record["input"]["context"]
        assert "$" not in text and "GOOD" not in text and "BAD" not in text and "Docstring" not in text
        assert record["input"]["file"] in querytests.NEUTRAL_FILES["python"]
        assert record["provenance"]["file_path"].endswith("/test.py")
        assert record["provenance"]["source_kind"] == "tool_test"


# ---------------------------------------------------------------------------------------------
# Split, governance, external benchmarks
# ---------------------------------------------------------------------------------------------


def _record(source_kind: str | None, committed: str) -> dict[str, Any]:
    provenance: dict[str, Any] = {"repository_key": "github/codeql", "fix_committed_at": committed}
    if source_kind:
        provenance["source_kind"] = source_kind
    return {"provenance": provenance}


def test_tool_tests_are_exempt_from_the_temporal_cutoff_only() -> None:
    settings = SplitSettings(salt="sentinel-garden-seed-v1-split", validation_per_mille=100, holdout_reserved_per_mille=100, temporal_cutoff="2026-07-01T00:00:00Z")
    assert assign_split(_record("tool_test", "2026-11-25T00:00:00Z"), settings) == "train"
    assert assign_split(_record(None, "2026-11-25T00:00:00Z"), settings) == "temporal_quarantine"


def test_query_test_negatives_stay_pending_until_validated() -> None:
    config = load_config(ROOT / "configs/garden/seed_v6.yaml")
    assert unvalidated_negative_kinds(config) == {"defended_sibling", "query_test_negative"}


def test_external_benchmark_records_are_excluded() -> None:
    benchmarks = {"ids": {"CVE-2020-1": "ossf-cve-benchmark:CVE-2020-1"}, "commits": {("o/r", SHA_B): "ossf-cve-benchmark:CVE-2020-2"}}

    def record(advisory: str, aliases: list[str], revision: str) -> dict[str, Any]:
        return {
            "provenance": {"advisory_id": advisory, "aliases": aliases, "repository_key": "o/r", "vulnerable_revision": revision, "fixed_revision": SHA_C},
            "governance": {"approval_status": "training_approved", "blockers": []},
            "evaluation_role": "training",
        }

    by_alias = record("GHSA-2234-5678-9cfg", ["CVE-2020-1"], SHA_A)
    apply_external_benchmark(by_alias, benchmarks)
    assert by_alias["evaluation_role"] == "external_benchmark"
    assert by_alias["governance"] == {"approval_status": "excluded", "blockers": ["external_benchmark:ossf-cve-benchmark:CVE-2020-1"]}
    by_commit = record("CVE-2021-9", [], SHA_B)
    apply_external_benchmark(by_commit, benchmarks)
    assert by_commit["evaluation_role"] == "external_benchmark"
    unrelated = record("CVE-2021-8", [], SHA_A)
    apply_external_benchmark(unrelated, benchmarks)
    assert unrelated["evaluation_role"] == "training"


def test_v6_config_keeps_the_v5_list_as_its_frozen_prefix() -> None:
    config = load_config(ROOT / "configs/garden/seed_v6.yaml")
    d = config.raw["discovery"]
    assert d["frozen_candidates"] == "data/garden/seed_v5/candidates.jsonl"
    assert d["frozen_candidates_sha256"] == config.raw["governance"]["source_approval_covers"]["sha256"]
    assert "extension" not in d  # the v5 extension is already part of the frozen list
    v6 = ROOT / "data/garden/seed_v6/candidates.jsonl"
    if v6.is_file():
        v5 = (ROOT / d["frozen_candidates"]).read_bytes()
        assert v6.read_bytes().startswith(v5)


@pytest.mark.parametrize("name", ["bad1", "goodHandler", "unsafe_query", "test_ok", "vulnerable"])
def test_revealing_names(name: str) -> None:
    assert querytests._REVEALING_NAME.search(name)


def test_tool_test_records_reach_the_review_sample_and_their_own_package(tmp_path: Path) -> None:
    from sentinel.garden import human_review

    root = _codeql_checkout(tmp_path)
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    config = load_config(ROOT / "configs/garden/seed_v6.yaml")
    records, _ = querytests.build_records(root, revision, config.extraction, "sentinel-garden-v6", set(), 2, log=lambda *a: None)
    for record in records:
        record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
        record["split"] = "train"
    sample = human_review.draw_sample(records)
    assert {item["item"] for item in sample} == {"tool_finding", "tool_negative"}
    by_id = {r["record_id"]: r for r in records}
    for item in sample:
        text = human_review.build_package(item, by_id, None)
        assert "CodeQL rule `py/sql-injection`" in text and "rule_" in text and "input_reveals_label" in text


def test_commit_message_naming_the_cve_is_the_weakest_accepted_grade() -> None:
    repo = "https://github.com/acme/app"
    fixes = [{"cve_id": "CVE-2022-0030", "hash": SHA_A, "repo_url": repo, "rel_type": "NVD_GIT_REPOBASED"}]
    index = _index(fixes, [_cve("CVE-2022-0030", ["https://example.com/report"])], {"CVE-2022-0030": ["CWE-79"]}, {SHA_A: ["src/app.js"]})
    index["commits"][("acme/app", SHA_A)]["msg"] = "Escape the title (fixes CVE-2022-0030)"
    added, outcomes, _, _ = morefixes.discover_candidates(index, [], SETTINGS, set(), 2, "salt")
    assert [c["fix_evidence"] for c in added] == ["commit_message_reference"] and outcomes == {"candidate_commit_message_reference": 1}
    assert not morefixes.names_advisory("see CVE-2022-00301", ["CVE-2022-0030"])
