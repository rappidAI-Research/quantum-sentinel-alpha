from pathlib import Path

import pytest

from sentinel.garden import code
from sentinel.garden.diff import parse_unified_diff
from sentinel.garden.licenses import detect, detect_license_text
from tests.garden_fixtures import MIT_TEXT, git_diff

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "language", "role"),
    [
        ("app/views.py", "python", "source"),
        ("tests/test_views.py", "python", "test"),
        ("pkg/server/handler_test.go", "go", "test"),
        ("vendor/github.com/x/y.go", "go", "excluded"),
        ("src/index.ts", "typescript", "source"),
        ("src/types.d.ts", None, "source"),
        ("lib/util.spec.js", "javascript", "test"),
        ("node_modules/a/index.js", "javascript", "excluded"),
        ("dist/bundle.min.js", None, "excluded"),
        ("docs/conf.py", "python", "excluded"),
    ],
)
def test_path_classification(path: str, language: str | None, role: str) -> None:
    assert code.language_of(path) == language
    assert code.path_role(path) == role


def test_python_functions_include_decorators_and_nesting() -> None:
    source = "import x\n\n@app.get('/a')\ndef outer(a):\n    def inner(b):\n        return b\n    return inner(a)\n"
    functions = code.find_functions("python", source)
    assert [(f.name, f.start, f.end) for f in functions] == [("outer", 3, 7), ("inner", 5, 6)]
    assert code.innermost(functions, 6, 6).name == "inner"


def test_python2_code_falls_back_to_indentation() -> None:
    source = "def legacy(x):\n    print 'value', x\n    return x\n\ndef other():\n    pass\n"
    functions = code.find_functions("python", source)
    assert [(f.name, f.start, f.end) for f in functions] == [("legacy", 1, 3), ("other", 5, 6)]


def test_go_functions_follow_gofmt_layout() -> None:
    source = "package a\n\nfunc (s *Server) Run() map[string]struct{} {\n\tx := struct{}{}\n\treturn nil\n}\n\nfunc One() int { return 1 }\n"
    functions = code.find_functions("go", source)
    assert [(f.name, f.start, f.end) for f in functions] == [("Run", 3, 6), ("One", 8, 8)]


def test_js_functions_methods_arrows_and_route_handlers() -> None:
    source = (
        "const re = /[{]/g;\n"  # regex literal with a brace must not confuse brace matching
        "function render(name) {\n"
        "  return `<b>${name.replace(re, '')}</b>`;\n"
        "}\n"
        "class Store {\n"
        "  async load(id: string): Promise<{ id: string }> {\n"
        "    return db.get(id);\n"
        "  }\n"
        "}\n"
        "export const save = async (item) => {\n"
        "  // { not a brace\n"
        "  return db.put(item);\n"
        "};\n"
        "app.post('/login', (req, res) => {\n"
        "  res.send('}');\n"
        "});\n"
    )
    functions = {(f.name, f.start, f.end) for f in code.find_functions("typescript", source)}
    assert ("render", 2, 4) in functions
    assert ("save", 10, 13) in functions
    assert ("<anonymous:post-handler>", 14, 16) in functions
    assert any(name == "load" and start == 6 for name, start, _ in functions)


def test_masking_keeps_line_numbers_and_hides_strings() -> None:
    source = 'a = "{"; // }\nb = `x ${y + "}"} z`;\n/* {\n */ c = 1;\n'
    masked = code.mask_c_like(source, "javascript")
    assert masked.count("\n") == source.count("\n")
    assert masked.count("{") - masked.count("}") == 0


def test_import_lines_and_trivial_changes() -> None:
    assert code.import_lines("python", "import os\nfrom a import b\nfrom typing import (\n    X,\n)\nx = 1\n") == ["import os", "from a import b"]
    assert code.import_lines("go", 'package a\n\nimport (\n\t"os"\n\tlog "log/slog"\n)\n') == ['"os"', 'log "log/slog"']
    assert code.is_import_or_trivial("python", "# comment")
    assert code.is_import_or_trivial("javascript", "const fs = require('fs');")
    assert not code.is_import_or_trivial("go", "\treturn nil")


def test_normalized_hash_ignores_comments_and_layout() -> None:
    a = "def f(x):\n    # comment\n    return x + 1\n"
    b = "def f(x):\n    return x+1  # other\n"
    assert code.normalized_sha256("python", a) == code.normalized_sha256("python", b)
    js_a = "function f(x) { // c\n  return '//' + x; }\n"
    js_b = "function f(x) {\n  /* c */ return '//' + x;\n}\n"
    assert code.normalized_sha256("javascript", js_a) == code.normalized_sha256("javascript", js_b)


def test_shingles_are_stable_across_processes() -> None:
    import subprocess
    import sys

    program = "from sentinel.garden import code; print(sorted(code.shingles(['a','b','c','d','e','f']))[0])"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, env={"PYTHONHASHSEED": seed}, cwd=ROOT, check=True
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1


def test_excerpt_well_formedness() -> None:
    assert code.excerpt_is_well_formed("python", "    def m(self):\n        return 1\n")
    assert code.excerpt_is_well_formed("go", "func a() {\n\tif x {\n\t}\n}\n")
    assert not code.excerpt_is_well_formed("go", "func a() {\n\tif x {\n}\n")


def test_diff_parser_reports_old_and_new_lines() -> None:
    hunks = parse_unified_diff(git_diff("a\nb\nc\n", "a\nB\nc\nd\n", "f.py"))
    assert [(h.old_start, h.old_count, h.new_start, h.new_count) for h in hunks] == [(2, 1, 2, 1), (3, 0, 4, 1)]
    assert hunks[0].removed == [(2, "b")] and hunks[0].added == [(2, "B")]
    assert hunks[1].is_insertion and hunks[1].added == [(4, "d")]


def test_diff_parser_rejects_inconsistent_hunks() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_unified_diff("@@ -1,2 +1,1 @@\n-a\n+b\n")


def test_license_detection_is_evidence_only() -> None:
    assert detect_license_text(MIT_TEXT) == "MIT"
    assert detect_license_text("Apache License\nVersion 2.0, January 2004") == "Apache-2.0"
    assert detect_license_text("GNU GENERAL PUBLIC LICENSE Version 3") == "GPL-3.0"
    assert detect({}).spdx == "none"
    assert detect({"LICENSE": MIT_TEXT, "NOTICE.txt": "random"}).category == "permissive"
    dual = detect({"LICENSE-MIT": MIT_TEXT, "LICENSE-APACHE": "Apache License Version 2.0"})
    assert dual.spdx.startswith("multiple:") and dual.category == "unknown"
