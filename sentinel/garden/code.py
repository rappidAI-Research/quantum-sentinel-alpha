"""Language-aware helpers: path classification, function boundaries, imports and normalization.

Function boundaries:
- Python: the `ast` module (decorators included); an indentation scan is the fallback for files
  that do not parse under the running interpreter (for example Python 2 code).
- Go, JavaScript and TypeScript: the tree-sitter parse tree (`syntax.py`): function and method
  declarations, block-bodied function expressions and arrows (named after the variable, property or
  member they are assigned to), anonymous callbacks; strings, template and regex literals, comments,
  tokens, calls, imports and local names are read from the same tree.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
import warnings
from dataclasses import dataclass
from pathlib import PurePosixPath

from sentinel.garden import syntax

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
}
ECOSYSTEM_LANGUAGES = {
    "PyPI": {"python"},
    "npm": {"javascript", "typescript"},
    "Go": {"go"},
}

_TEST_DIRS = {"test", "tests", "__tests__", "testing", "testdata", "spec", "specs", "e2e", "fixtures", "__mocks__", "mocks"}
_EXCLUDED_DIRS = {
    "vendor",
    "node_modules",
    "third_party",
    "third-party",
    "dist",
    "build",
    "docs",
    "doc",
    "examples",
    "example",
    "benchmark",
    "benchmarks",
    "bench",
    ".github",
}
_TEST_FILE = re.compile(
    r"(^test_.*\.py$|.*_test\.py$|^conftest\.py$|.*_test\.go$|.*\.(test|spec)\.[cm]?[jt]sx?$)", re.IGNORECASE
)


def language_of(path: str) -> str | None:
    pure = PurePosixPath(path)
    if pure.name.endswith(".d.ts") or pure.name.endswith(".min.js"):
        return None
    return LANGUAGE_EXTENSIONS.get(pure.suffix.lower())


def path_role(path: str) -> str:
    """'source', 'test' or 'excluded' (vendored, generated, docs or examples)."""
    pure = PurePosixPath(path)
    parts = [p.lower() for p in pure.parts[:-1]]
    if any(p in _EXCLUDED_DIRS for p in parts):
        return "excluded"
    if any(p in _TEST_DIRS for p in parts) or _TEST_FILE.match(pure.name):
        return "test"
    return "source"


# --------------------------------------------------------------------------------------------
# Masking of strings and comments (C-like languages)
# --------------------------------------------------------------------------------------------

# a `)` closing the head of one of these statements ends an expression context: `if (x) /re/.test(y)`


def mask_c_like(source: str, language: str, literal_spans: list[tuple[int, int]] | None = None) -> str:
    """Replace the contents of strings, comments, template literals and regex literals by spaces.

    Read from the tree-sitter parse tree. Newlines are kept so line numbers stay valid; quote and
    comment delimiters are kept. When `literal_spans` is given, the (start, end) offsets of every
    blanked string/template/regex body are appended to it in source order (comments are not literals).
    """
    src = syntax.parse(language, source)
    out = list(source)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, len(out))):
            if out[k] != "\n":
                out[k] = " "

    spans: list[tuple[int, int]] = []
    for node in syntax.literals(src):
        spans.extend(syntax.literal_bodies(src, node))
    for start, end in spans:
        blank(start, end)
    for node in syntax.comments(src):
        start, end = src.span(node)
        text = source[start:end]
        closing = 2 if text.startswith("/*") and text.endswith("*/") and len(text) >= 4 else 0
        blank(start + 2, end - closing)
    if literal_spans is not None:
        literal_spans.extend(sorted(spans))
    return "".join(out)


# --------------------------------------------------------------------------------------------
# Function boundaries
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Function:
    name: str
    start: int  # 1-based, inclusive (decorators / leading signature line)
    end: int  # 1-based, inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def find_functions(language: str, source: str) -> list[Function]:
    if language == "python":
        return _python_functions(source)
    if language in {"go", "javascript", "typescript"}:
        return [Function(f.name, f.start, f.end) for f in syntax.functions(syntax.parse(language, source))]
    raise ValueError(f"unsupported language: {language}")


def innermost(functions: list[Function], first: int, last: int) -> Function | None:
    containing = [f for f in functions if f.start <= first and last <= f.end]
    if not containing:
        return None
    return min(containing, key=lambda f: (f.length, -f.start))


def _parse_python(source: str) -> ast.Module:
    with warnings.catch_warnings():  # upstream code may contain e.g. invalid escape sequences
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def _python_functions(source: str) -> list[Function]:
    try:
        tree = _parse_python(source)
    except (SyntaxError, ValueError):
        return _python_functions_by_indent(source)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            functions.append(Function(node.name, start, node.end_lineno or node.lineno))
    return sorted(functions, key=lambda f: (f.start, f.end))


def python_docstring_lines(source: str) -> set[int]:
    """Line numbers of module, class and function docstrings (empty if the file does not parse)."""
    try:
        tree = _parse_python(source)
    except (SyntaxError, ValueError):
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


_PY_DEF = re.compile(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")


def _python_functions_by_indent(source: str) -> list[Function]:
    lines = source.splitlines()
    functions = []
    for index, line in enumerate(lines):
        match = _PY_DEF.match(line)
        if not match:
            continue
        indent = len(match.group(1).expandtabs())
        start = index
        while start > 0 and lines[start - 1].strip().startswith("@"):
            start -= 1
        end = index
        for k in range(index + 1, len(lines)):
            stripped = lines[k].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(lines[k]) - len(lines[k].lstrip()) <= indent and not stripped.startswith((")", "]", "}")):
                break
            end = k
        functions.append(Function(match.group(2), start + 1, end + 1))
    return functions










# --------------------------------------------------------------------------------------------
# Imports, comments and normalization
# --------------------------------------------------------------------------------------------

_BRACKETS_ONLY = re.compile(r"^[\s(){}\[\];,]*$")
# Top-level only (column 0): a `require` inside a function decides which implementation it calls.


def _python_line_classes(source: str) -> tuple[set[int], set[int]] | None:
    """(code lines, import-statement lines) from the tokenizer; None if the file does not tokenize."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    code_lines: set[int] = set()
    imports: set[int] = set()
    statement: list[tokenize.TokenInfo] = []
    for token in tokens:
        if token.type not in ignored:
            code_lines.update(range(token.start[0], token.end[0] + 1))
            statement.append(token)
        elif token.type in (tokenize.NEWLINE, tokenize.ENDMARKER) and statement:
            first = statement[0]
            if first.type == tokenize.NAME and first.string in {"import", "from"} and any(
                t.type == tokenize.NAME and t.string == "import" for t in statement
            ):
                imports.update(range(first.start[0], statement[-1].end[0] + 1))
            statement = []
    # Only module-scope imports are noise (also inside module-level `try:`/`if` blocks; the same
    # scope `_module_scope_imports` checks for re-binding). Imports in a function or class body are
    # code: they decide which implementation that code calls (pickle vs json).
    scopes = _python_scope_ranges(source)
    imports = {n for n in imports if not any(start < n <= end for start, end in scopes)}
    return code_lines, imports


def import_statement_lines(language: str, source: str) -> set[int]:
    """Line numbers that belong to module-level import statements (structural, from the parse tree)."""
    if language == "python":
        classes = _python_line_classes(source)
        return classes[1] if classes else set()
    return syntax.import_lines(syntax.parse(language, source))


def _line_classes(language: str, source: str) -> tuple[set[int], set[int]]:
    """(noise lines, bracket-only lines), classified on the whole source."""
    lines = source.splitlines()
    noise: set[int] = set()
    brackets: set[int] = set()
    if language == "python":
        classes = _python_line_classes(source)
        if classes is None:  # untokenizable: conservative textual fallback (blank / '#' lines only)
            noise = {n for n, line in enumerate(lines, start=1) if not line.strip() or line.strip().startswith("#")}
            return noise, set()
        code_lines, imports = classes
        for number, line in enumerate(lines, start=1):
            if number not in code_lines:
                noise.add(number)
            elif _BRACKETS_ONLY.match(line):
                brackets.add(number)
        return noise | imports | python_docstring_lines(source), brackets
    # From the parse tree: every line touching a string, template or regex literal is code (even a
    # whitespace-only line inside it); blank and comment-only lines are noise; a line whose code is
    # only brackets/separators is bracket-only.
    src = syntax.parse(language, source)
    code_lines, bracket_lines = syntax.code_lines(src)
    noise = {n for n in range(1, len(lines) + 1) if n not in code_lines}
    return noise | syntax.import_lines(src), bracket_lines


def noise_lines(language: str, source: str) -> set[int]:
    """Lines whose change cannot change behaviour: blank, comment-only, docstring or import statement.

    Classification uses the whole source, so a `*` dereference in Go, a `#` inside a Python SQL
    string or a string literal line are code, while block-comment continuation lines are not.
    Bracket-only lines are NOT noise: moving a closing brace changes control flow.
    """
    return _line_classes(language, source)[0]


def trivial_lines(language: str, source: str) -> set[int]:
    """Lines a localization span must not consist of: noise lines plus bracket-only lines."""
    noise, brackets = _line_classes(language, source)
    return noise | brackets




def module_imports(language: str, source: str) -> set[tuple[str, str]]:
    """Module-level imported bindings as (module, binding) pairs, for detecting non-additive changes."""
    pairs: set[tuple[str, str]] = set()
    if language == "python":
        try:
            tree = _parse_python(source)
        except (SyntaxError, ValueError):
            lines = source.splitlines()
            return {("<line>", re.sub(r"\s+", "", lines[n - 1])) for n in import_statement_lines(language, source)}
        for node in _module_scope_imports(tree):
            if isinstance(node, ast.Import):
                pairs.update((alias.name, alias.asname or alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                pairs.update((module, f"{alias.name} as {alias.asname or alias.name}") for alias in node.names)
        return pairs
    return syntax.imported_bindings(syntax.parse(language, source))


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _module_scope_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Imports executed in module scope, including inside module-level try/if/with/for blocks,
    but not inside function or class bodies."""
    found: list[ast.Import | ast.ImportFrom] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _python_scope_ranges(source: str) -> list[tuple[int, int]]:
    """Line ranges of function and class bodies (imports there are not module-level context)."""
    try:
        tree = _parse_python(source)
    except (SyntaxError, ValueError):
        return [(f.start, f.end) for f in _python_functions_by_indent(source)]
    return [
        (min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]), node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _python_bindings(source: str, path: str) -> dict[str, set[str]]:
    """Module-level import bindings -> fully qualified origins (relative imports resolved)."""
    try:
        tree = _parse_python(source)
    except (SyntaxError, ValueError):
        return {}
    parts = list(PurePosixPath(path).with_suffix("").parts)
    package = parts if parts and parts[-1] == "__init__" else parts[:-1]
    package = [p for p in package if p != "__init__"]
    result: dict[str, set[str]] = {}
    for node in _module_scope_imports(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    result.setdefault(alias.asname, set()).add(alias.name)
                else:  # `import a.b` binds the top-level package `a`
                    top = alias.name.split(".")[0]
                    result.setdefault(top, set()).add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            for alias in node.names:
                if alias.name != "*":
                    result.setdefault(alias.asname or alias.name, set()).add(f"{module}.{alias.name}")
    return result


def _same_origin(a: str, b: str) -> bool:
    # `src/pkg/x.py` resolves relative imports to `src.pkg...`, absolute imports say `pkg...`
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def rebound_imports(language: str, old_source: str, new_source: str, path: str = "") -> list[str]:
    """Local names imported at module level in both versions but from disjoint origins.

    Such a change alters the behaviour of unchanged code that uses the name (e.g. `loads` taken from
    `pickle` and then from `json`, or Go's `template` from text/template and then html/template),
    invisibly to a function-level diff. Added, removed or additionally sourced bindings are fine.
    """

    def bindings(source: str) -> dict[str, set[str]]:
        if language == "python":
            return _python_bindings(source, path)
        result: dict[str, set[str]] = {}
        for module, item in module_imports(language, source):
            if module in {"<line>", "<statement>"}:
                continue
            if language == "go":
                name, origin = item or module.rsplit("/", 1)[-1], module
            else:
                imported, _, alias = item.partition(" as ")
                name, origin = (alias or imported).strip(), f"{module}:{imported.strip()}"
            result.setdefault(name, set()).add(origin)
        return result

    before, after = bindings(old_source), bindings(new_source)
    rebound = []
    for name in before.keys() & after.keys():
        if not any(_same_origin(x, y) for x in before[name] for y in after[name]):
            rebound.append(name)
    return sorted(rebound)


def import_lines(language: str, source: str, limit: int = 15) -> list[str]:
    """Complete, single-line top-level import statements, in file order (used as context)."""
    imports = import_statement_lines(language, source)
    found: list[str] = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if number not in imports or not stripped or stripped in {"(", ")", "import ("}:
            continue
        if stripped.endswith(("(", "{", ",")) or stripped.startswith(("}", ")")):
            continue  # part of a multi-line import
        if language == "go" and stripped.startswith("//"):
            continue
        found.append(stripped)
        if len(found) >= limit:
            break
    return found


def call_lines(language: str, source: str, name: str) -> list[int]:
    """Lines where `name(...)` is called. Strings, comments, docstrings and definitions do not count."""
    if language == "python":
        try:
            tokens = [
                t
                for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            ]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return []
        found = []
        for i, token in enumerate(tokens[:-1]):
            if token.type == tokenize.NAME and token.string == name and tokens[i + 1].string == "(":
                previous = tokens[i - 1].string if i else ""
                if previous not in {"def", "class"}:
                    found.append(token.start[0])
        return sorted(set(found))
    return sorted({call.line for call in syntax.calls(syntax.parse(language, source)) if call.name == name})


def _python_local_bindings(function_source: str) -> set[str] | None:
    """Every name bound anywhere inside a Python function excerpt (parameters, assignments, loop and
    `with` targets, imports, nested definitions, `except ... as`, match captures, `global`/`nonlocal`).
    None when the excerpt does not parse."""
    import textwrap

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(textwrap.dedent(function_source))
    except (SyntaxError, ValueError):
        return None
    roots: list[ast.AST] = list(tree.body)
    if len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        # the function's own name and decorators belong to the enclosing scope
        roots = [tree.body[0].args, *tree.body[0].body]
    names: set[str] = set()
    for node in (n for root in roots for n in ast.walk(root)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def method_receiver(language: str, source: str, fn: Function) -> str | None:
    """None when `fn` is a plain function; otherwise how a method is called on its own object:
    "method" (Python `self.`/`cls.`, JS/TS `this.`, including `X.prototype.f = function` and
    object-literal / class-field functions), "obj:<path>" for a JS function assigned to an object
    (`internals.f = ...` is called as `internals.f(...)`) or, for Go, "go:<receiver type>"."""
    if language == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == fn.name and fn.start <= item.lineno <= fn.end:
                        return "method"
        return None
    found = syntax.function_at(syntax.parse(language, source), fn.start, fn.end)
    if found is None:
        return None
    if found.kind == "go_method":
        return f"go:{found.owner}"
    if found.kind in {"method", "object"}:
        return "method"
    if found.kind == "assigned" and found.owner is not None:
        if found.owner == "prototype" or found.owner.endswith(".prototype"):
            return "method"
        return f"obj:{found.owner}"
    return None


def owner_class(language: str, source: str, fn: Function) -> str | None:
    """The class a Python or JS/TS method belongs to (`X.prototype.f = ...` belongs to `X`), as
    `name@line` so two classes with the same name stay distinct. None when it has no class."""
    if language == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.start <= item.lineno <= fn.end:
                        return f"{node.name}@{node.lineno}"
        return None
    if language not in {"javascript", "typescript"}:
        return None
    src = syntax.parse(language, source)
    found = syntax.function_at(src, fn.start, fn.end)
    if found is None:
        return None
    if found.kind == "assigned" and found.owner is not None and (found.owner == "prototype" or found.owner.endswith(".prototype")):
        return found.owner[: -len(".prototype")] if found.owner.endswith(".prototype") else found.owner
    if found.kind in {"method", "object"}:
        return syntax.enclosing_class(src, found.node)
    return None


def enclosing_scope(language: str, source: str, fn: Function) -> tuple[int, int] | None:
    """Line range of the function `fn` is defined in, or None for a module-level function. A nested
    function is only visible inside that range, so only callers inside it can reach it by name."""
    if language == "python":
        try:
            tree = _parse_python(source)
        except (SyntaxError, ValueError):
            return None
        defs = (ast.FunctionDef, ast.AsyncFunctionDef)

        def visit(node: ast.AST, scope: ast.AST | None) -> tuple[int, int] | None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, defs) and child.name == fn.name and fn.start <= child.lineno <= fn.end:
                    if scope is None:
                        return None
                    first = min([scope.lineno, *(d.lineno for d in getattr(scope, "decorator_list", []))])
                    return (first, scope.end_lineno or scope.lineno)
                found = visit(child, child if isinstance(child, (*defs, ast.Lambda)) else scope)
                if found is not None:
                    return found
            return None

        return visit(tree, None)
    src = syntax.parse(language, source)
    found = syntax.function_at(src, fn.start, fn.end)
    if found is None:
        return None
    outer = syntax.enclosing_function(src, found.node)
    if outer is None:
        return None
    return (src.first_line(outer), src.last_line(outer))


def is_nested_function(language: str, source: str, fn: Function) -> bool:
    return enclosing_scope(language, source, fn) is not None


def calls_outer_function(language: str, function_source: str, name: str, receiver: str | None = None) -> bool:
    """True when the function excerpt really calls the outer function `name`.

    `receiver` says how the callee is reached (see `method_receiver`): a plain function only through a
    bare call `name(...)`; a method only through its own object (`self.`/`cls.`, `this.`, the same
    object path, or the caller's Go receiver of the same type; "go:*" and "obj:*" accept any owner).
    Calls on other objects (`repo.name(...)`) may reach a different function and do not count.

    A bare call does not count when the name is bound inside the excerpt (a local variable,
    parameter, nested definition or import shadows the outer function). Python bindings come from the
    AST. For Go/JS/TS (parse tree) every occurrence of the name as a variable must be the callee of a
    call; any other occurrence (declaration, parameter, destructuring, assignment) may bind it, so the
    excerpt is conservatively not treated as a caller.
    """
    if language == "python":
        try:
            tokens = [
                t
                for t in tokenize.generate_tokens(io.StringIO(function_source).readline)
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            ]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return False
        bare = member = False
        for i, token in enumerate(tokens[:-1]):
            if token.type != tokenize.NAME or token.string != name or tokens[i + 1].string != "(":
                continue
            before = tokens[i - 1].string if i else ""
            if before == ".":
                member = member or (i >= 2 and tokens[i - 2].string in {"self", "cls"} and (i < 3 or tokens[i - 3].string != "."))
            elif before not in {"def", "class"}:
                bare = True
        if receiver is not None:
            return member
        bindings = _python_local_bindings(function_source)
        return bare and bindings is not None and name not in bindings
    src = syntax.parse(language, function_source)
    found = [call for call in syntax.calls(src) if call.name == name]
    if receiver is not None:
        if receiver.startswith("go:"):
            fn = next(iter(syntax.functions(src)), None)
            if fn is None or fn.kind != "go_method":
                return False
            owner = syntax._go_receiver_type(src, fn.node)
            if owner is None or not owner[0] or receiver[3:] not in {owner[1], "*"}:
                return False
            return any(call.receiver == owner[0] for call in found)
        if receiver == "obj:*":
            return any(call.receiver is not None for call in found)
        if receiver.startswith("obj:"):
            return any(call.receiver == receiver[4:] for call in found)
        return any(call.receiver == "this" for call in found)
    bare_callees = {call.callee.id for call in found if call.receiver is None}
    if not bare_callees:
        return False
    return all(node.id in bare_callees for node in syntax.identifiers(src, name))


def strip_comments(language: str, source: str) -> str:
    if language == "python":
        lines = source.splitlines(keepends=True)
        try:
            comments = [
                t.start for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT
            ]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return "".join(line.split("#", 1)[0] + ("\n" if line.endswith("\n") else "") for line in lines)
        for row, col in comments:  # a comment runs to the end of its physical line
            line = lines[row - 1]
            lines[row - 1] = line[:col] + ("\n" if line.endswith("\n") else "")
        return "".join(lines)
    # a removed comment becomes one space (it still separates tokens: `a+/**/++b` is not `a+++b`)
    # and keeps its line breaks, so line numbers stay valid
    src = syntax.parse(language, source)
    result = []
    position = 0
    for node in syntax.comments(src):
        start, end = src.span(node)
        result.append(source[position:start])
        result.append(" " + "\n" * source.count("\n", start, end))
        position = end
    result.append(source[position:])
    return "".join(result)


# identifiers, numbers, operator runs (so `a + ++b` and `a++ + b` differ), any other single character
_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d[\w.]*|[-+*/%=<>!&|^~?:.]+|\S")


def _python_tokens(source: str) -> list[str] | None:
    """Tokens with block structure: Python indentation is significant, so INDENT/DEDENT and
    statement ends are kept as tokens (moving a statement into or out of a block is not cosmetic).

    The source is tokenized as is: a method excerpt's outer indentation shows up as one leading INDENT
    and one final DEDENT, which are dropped, so a method normalizes like the same top-level function.
    (Dedenting the text first would also rewrite whitespace inside multi-line string literals.)"""
    try:
        stream = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    significant = [t for t in stream if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER)]
    if significant and significant[0].type == tokenize.INDENT:
        closing = max(i for i, t in enumerate(significant) if t.type == tokenize.DEDENT)
        stream = significant[1:closing] + significant[closing + 1 :]
    out: list[str] = []
    for token in stream:
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER):
            continue
        if token.type == tokenize.INDENT:
            out.append("<INDENT>")
        elif token.type == tokenize.DEDENT:
            out.append("<DEDENT>")
        elif token.type == tokenize.NEWLINE:
            out.append("<NEWLINE>")
        else:  # string literals stay one token, so whitespace inside them is significant
            out.append(token.string)
    return out


def normalized_tokens(language: str, source: str) -> list[str]:
    """Comment- and layout-insensitive tokens; Python keeps its significant block structure.

    Go/JS/TS tokens come from the parse tree: operators are the grammar's own tokens, every string,
    regex literal and template-literal text piece is one exact token, and a line break between two
    tokens is kept as `<NL>` (runs collapse): with automatic semicolon insertion `return\n  !x` and
    `return !x` behave differently, so a line break is never treated as cosmetic.
    """
    if language == "python":
        tokens = _python_tokens(source)
        if tokens is not None:
            return tokens
        # untokenizable (e.g. mixed tabs/spaces): keep each line's indentation width as a token
        out: list[str] = []
        for line in strip_comments(language, source).splitlines():
            if line.strip():
                out.append(f"<I{len(line) - len(line.lstrip())}>")
                out.extend(_TOKEN.findall(line))
        return out
    result: list[str] = []
    last_line = None
    for text, first, last in syntax.tokens(syntax.parse(language, source), _TOKEN.findall):
        if last_line is not None and first > last_line:
            result.append("<NL>")
        result.append(text)
        last_line = last
    return result




def normalized_sha256(language: str, source: str) -> str:
    return hashlib.sha256(" ".join(normalized_tokens(language, source)).encode("utf-8")).hexdigest()


def _stable_hash(parts: list[str]) -> int:
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def shingles(tokens: list[str], size: int = 5) -> set[int]:
    """Token n-gram fingerprints with a process-independent hash (deterministic across runs)."""
    if len(tokens) < size:
        return {_stable_hash(tokens)} if tokens else set()
    return {_stable_hash(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def excerpt_is_well_formed(language: str, excerpt: str) -> bool:
    """Cheap syntax sanity check of a function excerpt."""
    if language == "python":
        import textwrap

        try:
            _parse_python(textwrap.dedent(excerpt))
            return True
        except SyntaxError:
            # Python 2 code or a method using syntax the running interpreter rejects:
            # fall back to a bracket balance check
            pass
        except ValueError:
            return False
        text = excerpt
    else:
        text = mask_c_like(excerpt, language)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for c in text if language != "python" else _strip_py_strings(text):
        if c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


def _strip_py_strings(text: str) -> str:
    return re.sub(r"(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|#[^\n]*)", "", text)
