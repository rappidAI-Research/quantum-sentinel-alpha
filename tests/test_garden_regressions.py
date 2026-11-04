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


# --- Second review round ------------------------------------------------------------------------


def test_python_reindentation_fix_is_not_cosmetic() -> None:
    old = "def get_secret(user, resource):\n    if user.is_admin:\n        log(user)\n    return resource.secret\n"
    new = "def get_secret(user, resource):\n    if user.is_admin:\n        log(user)\n        return resource.secret\n"
    assert code.normalized_sha256("python", old) != code.normalized_sha256("python", new)
    records = build({"a.py": old}, {"a.py": new}, candidate(cwes=["CWE-862"]))
    vulnerable = next(r for r in records if r["kind"] == "vulnerable")
    assert vulnerable["input"]["code"].splitlines()[vulnerable["target"]["start_line"] - 1].strip() == "return resource.secret"


def test_python_normalization_still_ignores_comments_and_spacing() -> None:
    a = "def f(x):\n    # note\n    return x + 1\n"
    b = "def f(x):\n    return x+1  # other\n"
    assert code.normalized_sha256("python", a) == code.normalized_sha256("python", b)
    method = "    def m(self):\n        return 1\n"
    assert code.normalized_sha256("python", method) == code.normalized_sha256("python", "def m(self):\n    return 1\n")


@pytest.mark.parametrize(
    "signature",
    [
        "function render(name: string): { html: string } {",
        "function render(name: string): Promise<{ html: string }> {",
        "function render(name: string): { a: string; b: number } | null {",
        "function render(name: string): Record<string, (x: number) => void> {",
        "export const render = (name: string): { html: string } => {",
    ],
)
def test_typescript_return_type_braces_are_skipped(signature: str) -> None:
    source = f"{signature}\n  const html = escape(name);\n  return {{ html }};\n}}\n"
    functions = [(f.name, f.start, f.end) for f in code.find_functions("typescript", source)]
    assert ("render", 1, 4) in functions


def test_typescript_return_type_fix_is_extracted() -> None:
    old = "function render(name: string): { html: string } {\n  const html = '<b>' + name + '</b>';\n  return { html };\n}\n"
    new = "function render(name: string): { html: string } {\n  const html = '<b>' + escapeHtml(name) + '</b>';\n  return { html };\n}\n"
    records = build({"src/render.ts": old}, {"src/render.ts": new}, candidate("npm", ["CWE-79"]))
    vulnerable = next(r for r in records if r["kind"] == "vulnerable")
    assert vulnerable["provenance"]["excerpt_function"] == "render"
    assert vulnerable["target"]["start_line"] == 2


# --- Third review round -------------------------------------------------------------------------

REGEX_CASES = [
    ("a.py", "PyPI", 'import re\n\ndef check(name):\n    pattern = r"^[a-z ]+$"\n    return re.match(pattern, name)\n', 'import re\n\ndef check(name):\n    pattern = r"^[a-z]+$"\n    return re.match(pattern, name)\n'),
    ("a.js", "npm", "function check(name) {\n  const ok = /^[a-z ]+$/.test(name);\n  return ok;\n}\n", "function check(name) {\n  const ok = /^[a-z]+$/.test(name);\n  return ok;\n}\n"),
    ("a.go", "Go", 'package a\n\nfunc check(name string) bool {\n\tre := regexp.MustCompile("^[a-z ]+$")\n\treturn re.MatchString(name)\n}\n', 'package a\n\nfunc check(name string) bool {\n\tre := regexp.MustCompile("^[a-z]+$")\n\treturn re.MatchString(name)\n}\n'),
]


@pytest.mark.parametrize(("path", "ecosystem", "old", "new"), REGEX_CASES)
def test_whitespace_inside_literals_is_significant(path: str, ecosystem: str, old: str, new: str) -> None:
    language = code.language_of(path)
    assert code.normalized_sha256(language, old) != code.normalized_sha256(language, new)
    records = build({path: old}, {path: new}, candidate(ecosystem, ["CWE-20"]))
    vulnerable = next(r for r in records if r["kind"] == "vulnerable")
    assert "[a-z ]" in vulnerable["input"]["code"].splitlines()[vulnerable["target"]["start_line"] - 1]


def test_whitespace_outside_literals_is_still_cosmetic() -> None:
    assert code.normalized_sha256("javascript", "f(a,b);  // x\n") == code.normalized_sha256("javascript", "f( a, b );\n")
    assert code.normalized_sha256("javascript", "f('a b');\n") != code.normalized_sha256("javascript", "f('ab');\n")


def test_moved_closing_brace_is_a_logic_change_and_localized() -> None:
    old = "function get(user, resource) {\n  if (user.isAdmin) {\n    log(user);\n  }\n  return resource.secret;\n}\n"
    new = "function get(user, resource) {\n  if (user.isAdmin) {\n    log(user);\n  return resource.secret;\n  }\n}\n"
    records = build({"a.js": old}, {"a.js": new}, candidate("npm", ["CWE-862"]))
    vulnerable = next(r for r in records if r["kind"] == "vulnerable")
    lines = vulnerable["input"]["code"].splitlines()
    span = lines[vulnerable["target"]["start_line"] - 1 : vulnerable["target"]["end_line"]]
    assert "  return resource.secret;" in span


def test_brace_change_in_second_function_counts_for_the_one_function_rule() -> None:
    other_old = "function other(x) {\n  if (x) {\n    a();\n  }\n  b();\n}\n"
    other_new = "function other(x) {\n  if (x) {\n    a();\n  b();\n  }\n}\n"
    old = "function get(name) {\n  run(name);\n  return 1;\n}\n\n" + other_old
    new = "function get(name) {\n  run(escape(name));\n  return 1;\n}\n\n" + other_new
    with pytest.raises(Rejected) as excinfo:
        build({"a.js": old}, {"a.js": new}, candidate("npm", ["CWE-79"]))
    assert excinfo.value.reason == "multiple_functions_changed"


def test_noise_and_trivial_line_sets() -> None:
    source = "function f(x) {\n  // c\n  if (x) {\n    g();\n  }\n}\n"
    assert code.noise_lines("javascript", source) == {2}
    assert code.trivial_lines("javascript", source) == {2, 5, 6}


def test_presence_check_never_touches_git_objects(tmp_path) -> None:
    """has_commit must not look up objects: in a partial clone that lazily fetches whole histories."""
    from sentinel.garden.gitsource import FETCHED_MARKER, UpstreamRepo

    repo = UpstreamRepo(tmp_path, "https://example.invalid/none.git", "example/none")
    sha = "a" * 40
    assert not repo.has_commit(sha)  # no network, no git object access
    (repo.path / FETCHED_MARKER).write_text(sha + "\n")
    assert repo.has_commit(sha)


def _local_upstream(tmp_path):
    """A real git repository A -> B -> C served over file:// with partial-clone support."""
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(origin), *args], capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "uploadpack.allowfilter", "true")
    git("config", "uploadpack.allowanysha1inwant", "true")
    shas = []
    for name in ("a", "b", "c"):
        (origin / "f.py").write_text(f"def f():\n    return '{name}'\n")
        git("add", "f.py")
        git("commit", "-q", "-m", name)
        shas.append(git("rev-parse", "HEAD"))
    return f"file://{origin}", shas


def test_fix_commit_that_is_another_fix_parent_keeps_its_parent(tmp_path) -> None:
    from sentinel.garden.gitsource import UpstreamRepo

    url, (a, b, c) = _local_upstream(tmp_path)
    cache = tmp_path / "cache"
    for _ in range(2):  # a rebuild must see exactly what the first build saw
        repo = UpstreamRepo(cache, url, "example/local")
        assert repo.ensure_commit(c).parents == (b,)
        assert repo.ensure_commit(b).parents == (a,)  # b was the shallow boundary of c's fetch
        assert repo.ensure_commit(c).parents == (b,)
