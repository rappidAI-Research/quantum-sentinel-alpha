"""Garden v4 extraction: combined multi-function excerpts, contiguous commit chains, context notes.

Every refinement is config-gated: with the v3 settings the same inputs are rejected or built exactly as
before (tests/test_garden_v3_multicommit.py, tests/test_garden_v2_extraction.py)."""

from __future__ import annotations

import dataclasses

import pytest
from jsonschema import Draft202012Validator

from sentinel.garden import audit
from sentinel.garden.build import CommitChain, Rejected, build_records, commit_chain
from sentinel.garden.gitsource import CommitInfo
from sentinel.garden.quality import check_pairs, check_record, load_record_schema
from tests.garden_fixtures import PY_NEW, PY_OLD, candidate
from tests.test_garden_v3_multicommit import BASE, V3, A, B, C, MultiRepo, multi

V4 = dataclasses.replace(
    V3, multi_function_fixes=True, max_multi_functions=3, max_multi_changed_lines=60, commit_chains=True, context_callee_notes=True
)

OLD = '''import sqlite3

from flask import request


def list_orders(conn, status):
    query = "SELECT id FROM orders WHERE status = '%s'" % status
    rows = conn.execute(query)
    return [row[0] for row in rows]


def count_orders(conn, status):
    query = "SELECT count(*) FROM orders WHERE status = '%s'" % status
    return conn.execute(query).fetchone()[0]


def orders_view(conn):
    status = request.args.get("status", "open")
    return {"orders": list_orders(conn, status), "count": count_orders(conn, status)}
'''
NEW = OLD.replace(
    '''    query = "SELECT id FROM orders WHERE status = '%s'" % status
    rows = conn.execute(query)''',
    '''    query = "SELECT id FROM orders WHERE status = ?"
    rows = conn.execute(query, (status,))''',
).replace(
    '''    query = "SELECT count(*) FROM orders WHERE status = '%s'" % status
    return conn.execute(query).fetchone()[0]''',
    '''    query = "SELECT count(*) FROM orders WHERE status = ?"
    return conn.execute(query, (status,)).fetchone()[0]''',
)


def _repo(old: str, new: str) -> MultiRepo:
    return MultiRepo({BASE: {"shop/orders.py": old}}, {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": new})})


def _single(sha: str = A) -> dict:
    cand = candidate()
    cand["fix_commit"] = sha
    return cand


def _build(repo, cand, settings=V4):
    return {r["kind"]: r for r in build_records(cand, repo, settings, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv"])}


def test_two_changed_functions_in_one_file_become_one_combined_excerpt() -> None:
    records = _build(_repo(OLD, NEW), _single())
    vulnerable, fixed = records["vulnerable"], records["post_fix"]
    assert vulnerable["provenance"]["excerpt_functions"] == ["list_orders", "count_orders"]
    lines = vulnerable["input"]["code"].splitlines()
    # the two functions in file order, one blank line between them
    assert lines[0].startswith("def list_orders") and lines[5].startswith("def count_orders") and lines[4] == ""
    # the label span is a changed site; every site is recorded in excerpt lines
    sites = vulnerable["localization"]["sites"]
    assert len(sites) == 2 and [vulnerable["target"]["start_line"], vulnerable["target"]["end_line"]] in sites
    for start, end in sites:
        assert "%" in "\n".join(lines[start - 1 : end])
    assert vulnerable["provenance"]["excerpt_file_lines"] == [[6, 9], [12, 14]]
    assert fixed["provenance"]["excerpt_functions"] == ["list_orders", "count_orders"]
    assert "caller_abstention" not in records  # one withheld function per abstention record only
    assert vulnerable["input"]["context"].startswith("Functions `list_orders`, `count_orders` from shop/orders.py.")
    # the combined pair passes the record and pair checks (the patch turns one excerpt into the other)
    validator = Draft202012Validator(load_record_schema())
    for record in (vulnerable, fixed):
        assert check_record(record, validator, set(), set(), 1000, 3)[0] == []
    assert not check_pairs([vulnerable, fixed])
    assert audit.excerpt_blocks(fixed) == [b + "\n" for b in fixed["input"]["code"].rstrip("\n").split("\n\n")]


def test_combined_excerpts_need_the_v4_settings() -> None:
    with pytest.raises(Rejected, match="multiple_functions_changed"):
        _build(_repo(OLD, NEW), _single(), V3)


def test_changes_in_two_files_or_too_many_functions_stay_rejected() -> None:
    other_old = PY_OLD.replace("orders", "items")
    other_new = PY_NEW.replace("orders", "items")
    repo = MultiRepo(
        {BASE: {"shop/orders.py": PY_OLD, "shop/items.py": other_old}},
        {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": PY_NEW, "shop/items.py": other_new})},
    )
    with pytest.raises(Rejected, match="multiple_functions_changed"):
        _build(repo, _single())
    with pytest.raises(Rejected, match="multiple_functions_changed"):
        _build(_repo(OLD, NEW), _single(), dataclasses.replace(V4, max_multi_functions=1))


def test_primary_span_prefers_replaced_code_over_an_insertion() -> None:
    new = OLD.replace(
        '''def count_orders(conn, status):
''',
        '''def count_orders(conn, status):
    if status not in ("open", "closed"):
        raise ValueError(status)
''',
    ).replace(
        '''    query = "SELECT id FROM orders WHERE status = '%s'" % status
    rows = conn.execute(query)''',
        '''    query = "SELECT id FROM orders WHERE status = ?"
    rows = conn.execute(query, (status,))''',
    )
    vulnerable = _build(_repo(OLD, new), _single())["vulnerable"]
    assert vulnerable["provenance"]["excerpt_function"] == "list_orders"
    assert vulnerable["localization"]["span_kind"] == "changed_lines"
    assert vulnerable["localization"]["primary_function_lines"] == 4


def test_contiguous_commit_chain_is_one_fix() -> None:
    step1 = OLD.replace(
        '''    query = "SELECT id FROM orders WHERE status = '%s'" % status
    rows = conn.execute(query)''',
        '''    query = "SELECT id FROM orders WHERE status = ?"
    rows = conn.execute(query, (status,))''',
    )
    repo = MultiRepo(
        {BASE: {"shop/orders.py": OLD}},
        {
            A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": step1}),
            B: (A, "2025-03-02T10:00:00+00:00", {"shop/orders.py": NEW}),
        },
    )
    vulnerable = _build(repo, multi(A, B))["vulnerable"]
    provenance = vulnerable["provenance"]
    assert provenance["vulnerable_revision"] == BASE and provenance["fixed_revision"] == B
    assert provenance["fix_commit_chain"] == [A, B]
    assert provenance["fix_committed_at"] == "2025-03-02T10:00:00+00:00"  # the fix was complete with the last commit
    with pytest.raises(Rejected, match="multi_commit_distinct_changes"):
        _build(repo, multi(A, B), dataclasses.replace(V4, commit_chains=False))


def test_commits_that_are_not_one_contiguous_chain_are_rejected() -> None:
    def info(sha: str, *parents: str) -> CommitInfo:
        return CommitInfo(sha, parents, "2025-03-01T10:00:00+00:00", "2025-03-01T10:00:00+00:00")

    chain = commit_chain([info(B, A), info(A, BASE), info(C, B)])
    assert isinstance(chain, CommitChain) and chain.commits == (A, B, C) and chain.parents == (BASE,)
    assert commit_chain([info(A, BASE), info(C, B)]) is None  # B (unlisted) lies in between
    assert commit_chain([info(A, BASE), info(B, BASE)]) is None  # two branches from one parent
    assert commit_chain([info(A, BASE), info(B, A, "f" * 40)]) is None  # a merge commit
    assert commit_chain([info(A, BASE)]) is None


def test_every_record_kind_names_the_callees_it_does_not_show() -> None:
    records = _build(_repo(PY_OLD, PY_NEW), _single())
    abstention = records["caller_abstention"]
    note = "Defined elsewhere in shop/orders.py and not shown: `def list_orders(conn, status)`."
    assert note in abstention["input"]["context"]
    assert "its body is not included" not in abstention["input"]["context"]
    # the vulnerable function calls no same-file function: no note, and none is invented
    assert "Defined elsewhere" not in records["vulnerable"]["input"]["context"]
    v3 = _build(_repo(PY_OLD, PY_NEW), _single(), V3)
    assert "`list_orders` is defined elsewhere in shop/orders.py; its body is not included." in v3["caller_abstention"]["input"]["context"]


def test_callee_notes_appear_on_findings_too() -> None:
    old = PY_OLD.replace(
        '''def list_orders(conn, status):
    query = "SELECT id FROM orders WHERE status = '%s'" % status''',
        '''def _table():
    return "orders"


def list_orders(conn, status):
    query = "SELECT id FROM " + _table() + " WHERE status = '%s'" % status''',
    )
    new = old.replace(
        '''    query = "SELECT id FROM " + _table() + " WHERE status = '%s'" % status
    rows = conn.execute(query)''',
        '''    query = "SELECT id FROM " + _table() + " WHERE status = ?"
    rows = conn.execute(query, (status,))''',
    )
    records = _build(_repo(old, new), _single())
    for kind in ("vulnerable", "post_fix"):
        assert "and not shown: `def _table()`." in records[kind]["input"]["context"]


def test_combined_excerpt_span_precision_uses_the_primary_function() -> None:
    vulnerable = _build(_repo(OLD, NEW), _single())["vulnerable"]
    precision = audit.span_precision(vulnerable)
    assert precision["function_lines"] == vulnerable["localization"]["primary_function_lines"]
