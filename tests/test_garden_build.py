import pytest
from jsonschema import Draft202012Validator

from sentinel.garden.build import Rejected, build_records
from sentinel.garden.quality import apply_unified_patch, check_pairs, check_record, load_record_schema
from tests.garden_fixtures import FIX, GO_NEW, GO_OLD, PARENT, PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate

VALIDATOR = Draft202012Validator(load_record_schema())


def build(repo: FakeRepo, cand: dict | None = None) -> list[dict]:
    return build_records(cand or candidate(), repo, SETTINGS, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv", "github-advisory-db"])


def checks(record: dict) -> list[str]:
    failed, _ = check_record(record, VALIDATOR, {"CWE-1395"}, {"CWE-20"}, 1000, 3)
    return failed


def test_python_fix_yields_vulnerable_fixed_and_abstention_records() -> None:
    records = build(FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}))
    by_kind = {r["kind"]: r for r in records}
    assert set(by_kind) == {"vulnerable", "post_fix", "caller_abstention"}

    vulnerable = by_kind["vulnerable"]
    assert vulnerable["target"]["decision"] == "finding"
    assert vulnerable["target"]["cwe"] == "CWE-89"
    lines = vulnerable["input"]["code"].splitlines()
    assert lines[0] == "def list_orders(conn, status):"
    span = lines[vulnerable["target"]["start_line"] - 1 : vulnerable["target"]["end_line"]]
    assert span == ["    query = \"SELECT id FROM orders WHERE status = '%s'\" % status", "    rows = conn.execute(query)"]
    assert vulnerable["provenance"]["excerpt_revision"] == PARENT
    assert vulnerable["provenance"]["license"] == {
        "detected": "MIT",
        "category": "permissive",
        "files": [{"path": "LICENSE", "sha256": vulnerable["provenance"]["license"]["files"][0]["sha256"]}],
        "status": "unreviewed",
    }
    assert "from flask import request" in vulnerable["input"]["context"]

    fixed = by_kind["post_fix"]
    assert fixed["target"]["decision"] == "no_finding" and fixed["target"]["cwe"] is None
    assert fixed["provenance"]["excerpt_revision"] == FIX
    assert fixed["pair_id"] == vulnerable["pair_id"]
    assert apply_unified_patch(vulnerable["input"]["code"], vulnerable["target"]["patch"]) == fixed["input"]["code"]

    abstention = by_kind["caller_abstention"]
    assert abstention["target"]["decision"] == "abstain"
    assert abstention["input"]["code"].startswith("def orders_view(conn):")
    assert "`list_orders` is defined elsewhere" in abstention["input"]["context"]
    assert abstention["pair_id"] is None and abstention["group_id"] == vulnerable["group_id"]

    for record in records:
        assert checks(record) == [], record["kind"]
    assert check_pairs(records) == {}


def test_go_fix_with_new_helper_includes_helper_in_fixed_excerpt() -> None:
    records = build(FakeRepo({"auth/token.go": GO_OLD}, {"auth/token.go": GO_NEW}), candidate("Go", ["CWE-532"]))
    by_kind = {r["kind"]: r for r in records}
    vulnerable, fixed = by_kind["vulnerable"], by_kind["post_fix"]
    span = vulnerable["input"]["code"].splitlines()[vulnerable["target"]["start_line"] - 1]
    assert span.strip() == 'slog.String("token", token))'
    assert "func redactToken(token string) string {" in fixed["input"]["code"]
    assert vulnerable["localization"]["fix_added_top_level_lines"] > 0
    assert vulnerable["localization"]["diff_consistent"]
    assert len(fixed["provenance"]["excerpt_file_lines"]) == 2
    for record in records:
        assert checks(record) == []
    assert check_pairs(records) == {}


def test_record_ids_are_deterministic() -> None:
    first = build(FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}))
    second = build(FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}))
    assert first == second


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        # two functions changed -> function-level label is ambiguous
        (
            {"a.py": "def one(x):\n    return x\n\n\ndef two(y):\n    return y\n"},
            {"a.py": "def one(x):\n    return int(x)\n\n\ndef two(y):\n    return int(y)\n"},
            "multiple_functions_changed",
        ),
        # module-level logic change
        ({"a.py": "LIMIT = 10\n\n\ndef f(x):\n    return x\n"}, {"a.py": "LIMIT = 1\n\n\ndef f(x):\n    return x\n"}, "change_outside_function"),
        # only tests changed
        ({"tests/test_a.py": "def test_a():\n    assert 1\n"}, {"tests/test_a.py": "def test_a():\n    assert 2\n"}, "no_source_change_in_language"),
        # whitespace/comment-only change inside a function
        ({"a.py": "def f(x):\n    # old\n    return x\n"}, {"a.py": "def f(x):\n    # new\n    return x\n"}, "no_logic_change"),
        # fix introduces a new source file
        ({"a.py": "def f(x):\n    return x\n"}, {"a.py": "def f(x):\n    return g(x)\n", "b.py": "def g(x):\n    return x\n"}, "source_file_added_or_deleted"),
    ],
)
def test_ambiguous_fixes_are_rejected(old: dict, new: dict, reason: str) -> None:
    with pytest.raises(Rejected) as excinfo:
        build(FakeRepo(old, new))
    assert excinfo.value.reason == reason


def test_cosmetic_fix_is_rejected() -> None:
    old = {"a.py": "def f(x):\n    y = x+1\n    return y\n"}
    new = {"a.py": "def f(x):\n    y = x + 1\n    return y\n"}
    with pytest.raises(Rejected) as excinfo:
        build(FakeRepo(old, new))
    assert excinfo.value.reason == "cosmetic_fix"


def test_language_must_match_ecosystem() -> None:
    with pytest.raises(Rejected) as excinfo:
        build(FakeRepo({"auth/token.go": GO_OLD}, {"auth/token.go": GO_NEW}), candidate("PyPI"))
    assert excinfo.value.reason == "no_source_change_in_language"


def test_unresolvable_and_root_commits_are_rejected() -> None:
    with pytest.raises(Rejected) as excinfo:
        build(FakeRepo({}, {}), {**candidate(), "fix_commit": "c" * 40})
    assert excinfo.value.reason == "unresolvable_commit"
    with pytest.raises(Rejected) as excinfo:
        build(FakeRepo({}, {}, parents=()))
    assert excinfo.value.reason == "root_commit"


def test_missing_license_is_recorded_not_guessed() -> None:
    records = build(FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}, license_text=None))
    assert records[0]["provenance"]["license"]["detected"] == "none"
    assert records[0]["provenance"]["license"]["files"] == []


def test_docstring_edits_neither_widen_the_span_nor_count_as_logic_changes() -> None:
    old = (
        'def check(opts):\n    """Check options."""\n    for o in opts:\n        if o == "--upload-pack":\n            raise ValueError(o)\n\n\n'
        'def other(x):\n    """Old text."""\n    return x\n'
    )
    new = (
        'def check(opts):\n    """Check options, including abbreviations."""\n    for o in opts:\n        if "--upload-pack".startswith(o):\n            raise ValueError(o)\n\n\n'
        'def other(x):\n    """New text."""\n    return x\n'
    )
    records = build(FakeRepo({"git/cmd.py": old}, {"git/cmd.py": new}), candidate(cwes=["CWE-88"]))
    vulnerable = next(r for r in records if r["kind"] == "vulnerable")
    assert vulnerable["provenance"]["excerpt_function"] == "check"
    assert (vulnerable["target"]["start_line"], vulnerable["target"]["end_line"]) == (4, 4)
    assert checks(vulnerable) == []
