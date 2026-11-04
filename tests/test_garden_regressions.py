"""Regression tests for review findings on PR #4 and adjacent line-classification defects."""

import copy

import pytest
from jsonschema import Draft202012Validator

from sentinel.garden import code
from sentinel.garden.build import Rejected, build_records
from sentinel.garden.osv import parse_candidate
from sentinel.garden.quality import (
    EvalReference,
    QualitySettings,
    check_record,
    contamination,
    load_record_schema,
    repository_key,
)
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate

QUALITY = QualitySettings(0.85, 0.92, 0.5, 5)


def build(old: dict, new: dict, cand: dict | None = None) -> list[dict]:
    return build_records(cand or candidate(), FakeRepo(old, new), SETTINGS, "t", "ghsa-osv-fix-commits", ["osv"])


# --- Finding 1: repository names with dots ------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://github.com/example/webapp.js", "example/webapp.js"),
        ("https://github.com/Example/WebApp.js.git", "example/webapp.js"),
        ("https://github.com/socketio/socket.io/tree/main/lib", "socketio/socket.io"),
        ("git@github.com:owner/repo.git", "owner/repo"),
    ],
)
def test_repository_key_keeps_dots(url: str, key: str) -> None:
    assert repository_key(url) == key


def test_repository_overlap_detected_for_dotted_repository_names() -> None:
    cand = candidate(repo="example/webapp.js")
    record = build({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}, cand)[0]
    reference = EvalReference("holdout-1", "python", "x = 1\n", "https://github.com/example/webapp.js")
    assert contamination([record], [reference], QUALITY)[record["record_id"]].startswith("eval_repository_overlap")


def test_discovery_keeps_dotted_repository_names() -> None:
    record = {
        "id": "GHSA-2222-3333-4444",
        "published": "2025-01-01T00:00:00Z",
        "summary": "x",
        "affected": [{"package": {"name": "socket.io", "ecosystem": "npm"}}],
        "references": [{"type": "WEB", "url": f"https://github.com/socketio/socket.io/commit/{'a' * 40}"}],
        "database_specific": {"cwe_ids": ["CWE-79"], "github_reviewed": True},
    }
    cand, _ = parse_candidate(record, "npm", published_before="2026-01-01", require_reviewed=True, rejected_cwes=set(), max_cwes=2)
    assert cand["repository_key"] == "socketio/socket.io"


# --- Finding 2: Python callers mentioned only in strings / docstrings ---------------------------

HELP = 'def usage_help():\n    text = "Call list_orders(conn, status) to get orders"\n    return text\n'
DOC = 'def describe():\n    """Wraps list_orders(conn, status)."""\n    return None\n'
VIEW = 'def orders_view(conn):\n    status = request.args.get("status", "open")\n    return {"orders": list_orders(conn, status)}\n'


@pytest.mark.parametrize("helper", [HELP, DOC])
def test_string_or_docstring_mention_is_not_a_caller(helper: str) -> None:
    old, new = PY_OLD.replace(VIEW, helper), PY_NEW.replace(VIEW, helper)
    records = build({"a.py": old}, {"a.py": new})
    assert not [r for r in records if r["kind"] == "caller_abstention"]


def test_real_caller_after_string_mention_is_still_found() -> None:
    old, new = PY_OLD.replace(VIEW, HELP + "\n\n" + VIEW), PY_NEW.replace(VIEW, HELP + "\n\n" + VIEW)
    abstention = [r for r in build({"a.py": old}, {"a.py": new}) if r["kind"] == "caller_abstention"]
    assert [r["provenance"]["excerpt_function"] for r in abstention] == ["orders_view"]


def test_quality_check_rejects_abstention_whose_call_is_only_in_a_string() -> None:
    abstention = next(r for r in build({"a.py": PY_OLD}, {"a.py": PY_NEW}) if r["kind"] == "caller_abstention")
    forged = copy.deepcopy(abstention)
    forged["input"]["code"] = HELP
    forged["provenance"]["content_sha256"] = __import__("hashlib").sha256(HELP.encode()).hexdigest()
    forged["provenance"]["normalized_sha256"] = code.normalized_sha256("python", HELP)
    failed, _ = check_record(forged, Draft202012Validator(load_record_schema()), set(), set(), 1000, 3)
    assert "abstention_calls_withheld_callee" in failed


def test_python_call_lines_ignore_strings_comments_and_definitions() -> None:
    source = 'def run(x):\n    # run(x)\n    s = "run(x)"\n    return obj.run(x) + run(1)\n'
    assert code.call_lines("python", source, "run") == [4]


# --- Finding 3: Go pointer assignments are code, not comments ----------------------------------

GO_OLD = "package a\n\nfunc copyIn(out *string, in string) {\n\tcheck(in)\n\t*out = in\n}\n"
GO_NEW = "package a\n\nfunc copyIn(out *string, in string) {\n\tcheck(in)\n\t*out = sanitize(in)\n}\n"


def test_go_pointer_assignment_fix_is_extracted() -> None:
    records = build({"a.go": GO_OLD}, {"a.go": GO_NEW}, candidate("Go", ["CWE-79"]))
    vulnerable = next(r for r in records if r["kind"] == "vulnerable")
    line = vulnerable["input"]["code"].splitlines()[vulnerable["target"]["start_line"] - 1]
    assert line.strip() == "*out = in"


def test_go_pointer_change_counts_towards_the_one_function_rule() -> None:
    old = GO_OLD + "\nfunc other(p *int) {\n\tuse(p)\n\t*p = 1\n}\n"
    new = GO_NEW + "\nfunc other(p *int) {\n\tuse(p)\n\t*p = 2\n}\n"
    with pytest.raises(Rejected) as excinfo:
        build({"a.go": old}, {"a.go": new}, candidate("Go", ["CWE-79"]))
    assert excinfo.value.reason == "multiple_functions_changed"


# --- Adjacent: structural line classification ---------------------------------------------------


def test_block_comment_continuation_lines_are_trivial_but_code_is_not() -> None:
    source = "/*\n * comment\n */\nfunc f(p *int) {\n\t*p = 1\n}\n"
    assert {1, 2, 3} <= code.trivial_lines("go", source)
    assert 5 not in code.trivial_lines("go", source)
    js = "function f(a, b) {\n  return a\n    * b;\n}\n"
    assert 3 not in code.trivial_lines("javascript", js)


def test_python_lines_inside_strings_are_code() -> None:
    source = 'def q(owner):\n    sql = """\n    SELECT 1\n    # WHERE owner = %s\n    from users import name\n    """\n    return sql\n'
    trivial = code.trivial_lines("python", source)
    assert not {4, 5} & trivial


def test_go_string_literal_line_outside_import_block_is_code() -> None:
    source = 'package a\n\nimport (\n\t"fmt"\n)\n\nvar roles = []string{\n\t"admin",\n\t"user"}\n'
    trivial = code.trivial_lines("go", source)
    assert {3, 4, 5} <= trivial and not {8, 9} & trivial


def test_python_multiline_import_and_docstring_are_trivial() -> None:
    source = 'from x import (\n    a,\n    b,\n)\n\n\ndef f():\n    """Doc."""\n    return a\n'
    assert {1, 2, 3, 4, 8} <= code.trivial_lines("python", source)
    assert 9 not in code.trivial_lines("python", source)


def test_js_multiline_import_is_trivial() -> None:
    source = "import {\n  a,\n  b,\n} from 'x';\nconst c = require('c');\nrun(a);\n"
    assert code.trivial_lines("javascript", source) == {1, 2, 3, 4, 5}


def test_identifiers_starting_with_import_are_code() -> None:
    source = "importantValue = load();\nimport x from 'x';\nconst y = import ('y');\n"
    assert code.trivial_lines("javascript", source) == {2}


def test_typescript_import_require_without_semicolon_ends_the_statement() -> None:
    source = "import url = require('url')\nimport http = require('http')\n\nexport class C {\n  run() {\n    strip(headers)\n  }\n}\n"
    assert code.trivial_lines("typescript", source) == {1, 2, 3, 7, 8}


def test_lines_inside_multiline_strings_are_code() -> None:
    js = "function page(name) {\n  return `\n    <h1>${name}</h1>\n    plain text\n  `;\n}\n"
    assert not {3, 4} & code.trivial_lines("javascript", js)
    go = "package a\n\nfunc q() string {\n\treturn `\nSELECT *\nFROM t\n`\n}\n"
    assert not {5, 6} & code.trivial_lines("go", go)


def test_strip_comments_preserves_line_numbers() -> None:
    source = "a();\n/* one\n two\n*/ b();\nc(); // x\n"
    assert code.strip_comments("javascript", source).count("\n") == source.count("\n")


def test_semicolon_free_call_is_not_a_method_definition() -> None:
    source = (
        "function mergeSource(state, source) {\n"
        "  for (const element of source) {\n"
        "    mergeKeys(state, element)\n"
        "  }\n"
        "  if (x) {\n"
        "    y()\n"
        "  } else {\n"
        "    z()\n"
        "  }\n"
        "}\n"
    )
    names = [f.name for f in code.find_functions("typescript", source)]
    assert names == ["mergeSource"]
