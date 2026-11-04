"""Language-aware helpers: path classification, function boundaries, imports and normalization.

Function boundaries:
- Python: the `ast` module (decorators included); an indentation scan is the fallback for files
  that do not parse under the running interpreter (for example Python 2 code).
- Go: gofmt layout - a top-level `func` starts at column 0 and ends at the next line that is `}`.
- JavaScript/TypeScript: strings, comments, template literals and regex literals are masked, then
  named functions, methods, block-bodied arrow functions and anonymous callbacks are matched to
  their closing brace.
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

_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = ("return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do")


def mask_c_like(source: str, language: str, literal_spans: list[tuple[int, int]] | None = None) -> str:
    """Replace the contents of strings, comments, template literals and regex literals by spaces.

    Newlines are kept so line numbers stay valid; quote and delimiter characters are kept. When
    `literal_spans` is given, the (start, end) offsets of every blanked string/template/regex body are
    appended to it (comments are not literals).
    """
    out = list(source)
    n = len(source)
    i = 0
    is_js = language in {"javascript", "typescript"}
    template_stack: list[int] = []  # brace depth at which each `${` interpolation started
    brace_depth = 0

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":
                out[k] = " "

    def last_code_char(pos: int) -> tuple[str, str]:
        k = pos - 1
        while k >= 0 and out[k] in " \t\r\n":
            k -= 1
        if k < 0:
            return "", ""
        end = k + 1
        while k >= 0 and (out[k].isalnum() or out[k] in "_$"):
            k -= 1
        return out[end - 1], "".join(out[k + 1 : end])

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":  # keep the delimiter so comments stay identifiable
            end = source.find("\n", i)
            end = n if end < 0 else end
            blank(i + 2, end)
            i = end
            continue
        if ch == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            blank(i + 2, end - 2)
            i = end
            continue
        if ch in "\"'" or (ch == "`" and not is_js):
            j = i + 1
            while j < n and source[j] != ch:
                if source[j] == "\\" and ch != "`":
                    j += 1
                elif source[j] == "\n" and ch != "`":
                    break
                j += 1
            blank(i + 1, j)
            if literal_spans is not None:
                literal_spans.append((i + 1, j))
            i = j + 1
            continue
        if ch == "`" and is_js:
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == "`":
                    break
                if source[j] == "$" and j + 1 < n and source[j + 1] == "{":
                    break
                j += 1
            blank(i + 1, j)
            if literal_spans is not None:
                literal_spans.append((i + 1, j))
            if j < n and source[j] == "$":
                template_stack.append(brace_depth)
                brace_depth += 1
                i = j + 2
            else:
                i = j + 1
            continue
        if is_js and ch == "/":
            prev, word = last_code_char(i)
            if prev == "" or prev in _REGEX_PRECEDERS or word in _REGEX_KEYWORDS:
                j = i + 1
                in_class = False
                while j < n and source[j] != "\n":
                    c = source[j]
                    if c == "\\":
                        j += 2
                        continue
                    if c == "[":
                        in_class = True
                    elif c == "]":
                        in_class = False
                    elif c == "/" and not in_class:
                        break
                    j += 1
                if j < n and source[j] == "/":
                    blank(i + 1, j)
                    if literal_spans is not None:
                        literal_spans.append((i + 1, j))
                    i = j + 1
                    continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if template_stack and brace_depth == template_stack[-1]:
                # end of a `${...}` interpolation: continue inside the template literal
                template_stack.pop()
                j = i + 1
                while j < n:
                    if source[j] == "\\":
                        j += 2
                        continue
                    if source[j] == "`":
                        break
                    if source[j] == "$" and j + 1 < n and source[j + 1] == "{":
                        break
                    j += 1
                blank(i + 1, j)
                if literal_spans is not None:
                    literal_spans.append((i + 1, j))
                if j < n and source[j] == "$":
                    template_stack.append(brace_depth)
                    brace_depth += 1
                    i = j + 2
                else:
                    i = j + 1
                continue
        i += 1
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
    if language == "go":
        return _go_functions(source)
    if language in {"javascript", "typescript"}:
        return _js_functions(source, language)
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


_GO_FUNC = re.compile(r"^func\s*(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*[\[(]")


def _go_functions(source: str) -> list[Function]:
    # Scan the string/comment mask (same line coordinates): function-shaped text inside a raw string
    # or block comment is not a function, and a `}` inside a raw string does not end one.
    lines = mask_c_like(source, "go").splitlines()
    functions = []
    for index, line in enumerate(lines):
        match = _GO_FUNC.match(line)
        if not match:
            continue
        if line.rstrip().endswith("}") and line.count("{") == line.count("}"):
            functions.append(Function(match.group(1), index + 1, index + 1))  # one-line function
            continue
        for k in range(index + 1, len(lines)):
            if lines[k].rstrip() == "}":
                functions.append(Function(match.group(1), index + 1, k + 1))
                break
            if _GO_FUNC.match(lines[k]):
                break  # not gofmt layout; skip rather than guess
    return functions


_JS_IDENT = r"[A-Za-z_$][\w$]*"
_JS_NOT_FUNCTIONS = {
    "if", "for", "while", "switch", "catch", "function", "return", "with", "else", "do", "typeof",
    "new", "await", "super", "import", "export", "throw", "case", "delete", "void", "yield",
}
_JS_PATTERNS = [
    # function declarations / expressions with a name
    re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*({_JS_IDENT})\s*(?:<[^>]*>)?\s*\("),
    # const name = function(...) / const name = async (...) => / name = function
    re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_JS_IDENT})\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b[^(]*)?\("),
    re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_JS_IDENT})\s*(?::[^=]+)?=\s*(?:async\s+)?({_JS_IDENT})\s*=>"),
    re.compile(rf"^\s*((?:{_JS_IDENT}\.)*{_JS_IDENT})\s*=\s*(?:async\s+)?(?:function\b[^(]*)?\("),
    # object property `name: function(` / `name: (...) =>`
    re.compile(rf"^\s*['\"]?({_JS_IDENT})['\"]?\s*:\s*(?:async\s+)?(?:function\b[^(]*)?\("),
    # class / object method shorthand
    re.compile(
        rf"^\s*(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*\*?\s*(#?{_JS_IDENT})\s*(?:<[^>]*>)?\s*\("
    ),
]
_JS_ARROW_OR_FUNCTION = re.compile(r"=>\s*\{|\bfunction\b\s*\*?\s*[A-Za-z_$]*\s*\(")
_JS_ROUTE = re.compile(r"\.(get|post|put|delete|patch|use|all|options|head)\s*\(")


def _js_functions(source: str, language: str) -> list[Function]:
    masked = mask_c_like(source, language)
    line_offsets = [0]
    for line in masked.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    masked_lines = masked.splitlines()

    def line_of(offset: int) -> int:
        lo, hi = 0, len(line_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_offsets[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def matching_brace(open_at: int) -> int | None:
        depth = 0
        for k in range(open_at, len(masked)):
            c = masked[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return k
        return None

    def body_after_params(paren_at: int) -> int | None:
        """Offset of the `{` that opens the body after a parameter list starting at `paren_at`."""
        depth = 0
        k = paren_at
        while k < len(masked):
            c = masked[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        else:
            return None
        k += 1
        limit = min(len(masked), k + 400)
        between: list[str] = []  # characters outside any return-type nesting
        depth = 0  # nesting inside a TypeScript return-type annotation: (), [], <>, {}
        in_type = False
        last = ""  # last significant character at depth 0
        while k < limit:
            c = masked[k]
            if depth > 0:
                if c in "([{<":
                    depth += 1
                elif c in ")]}" or (c == ">" and masked[k - 1] != "="):
                    depth -= 1
                    if depth == 0:
                        last = c
                k += 1
                continue
            if c == ":" and not "".join(between).strip():
                in_type = True
            elif in_type and (c in "([<" or (c == "{" and last in {":", "|", "&", ",", "("})):
                depth = 1  # generic arguments, tuple/function types or an object-literal type
                k += 1
                continue
            elif c == "{":
                text = "".join(between).strip()
                # Only a signature continuation may separate `)` and the body: nothing, `=>`, or a
                # return-type annotation (`: T` / `: T =>`). Anything else - e.g. `return` in
                # semicolon-free code after a call - means the parentheses were a call.
                if text and text != "=>" and not in_type:
                    return None
                if ";" in text or re.search(r"(?<![=!<>])=(?!>)", text):
                    return None
                return k
            elif c in ";,)}":  # statement ended (also in semicolon-free code) before any body
                return None
            between.append(c)
            if not c.isspace():
                last = c
            k += 1
        return None

    functions: dict[tuple[int, int], Function] = {}
    for index, line in enumerate(masked_lines):
        for pattern in _JS_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1)
            if name in _JS_NOT_FUNCTIONS:
                continue
            base = line_offsets[index]
            if match.re.pattern.endswith("=>"):
                arrow = masked.find("=>", base + match.end() - 2)
                brace = masked.find("{", arrow)
                if brace < 0 or masked[arrow + 2 : brace].strip():
                    continue
            else:
                brace = body_after_params(base + match.end() - 1)
            if brace is None:
                continue
            close = matching_brace(brace)
            if close is None:
                continue
            start, end = index + 1, line_of(close)
            functions.setdefault((start, end), Function(name.split(".")[-1], start, end))
            break
    # anonymous callbacks: `(...) => {` or `function (...) {` not already covered
    for match in _JS_ARROW_OR_FUNCTION.finditer(masked):
        if match.group(0).startswith("=>"):
            brace = match.end() - 1
        else:
            brace = body_after_params(match.end() - 1)
            if brace is None:
                continue
        close = matching_brace(brace)
        if close is None:
            continue
        start = line_of(match.start())
        end = line_of(close)
        if any(f.end == end and f.start <= start for f in functions.values()):
            continue
        route = _JS_ROUTE.search(masked_lines[start - 1])
        name = f"<anonymous:{route.group(1)}-handler>" if route else "<anonymous>"
        functions.setdefault((start, end), Function(name, start, end))
    return sorted(functions.values(), key=lambda f: (f.start, f.end))


# --------------------------------------------------------------------------------------------
# Imports, comments and normalization
# --------------------------------------------------------------------------------------------

_BRACKETS_ONLY = re.compile(r"^[\s(){}\[\];,]*$")
# Top-level only (column 0): a `require` inside a function decides which implementation it calls.
_JS_IMPORT_START = re.compile(r"^(?:import(?=[\s{*'\"])(?!\s*\()|export\s*(?:type\s+)?(?:\*|\{[^}]*\}?)\s*(?:as\s+\w+\s*)?from\b)")
_JS_REQUIRE = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+[^=]+=\s*require\(\s*['\"][^'\"]*['\"]\s*\)\s*[;,]?\s*$")


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
    """Line numbers that belong to import statements (structural, never inside strings or comments)."""
    if language == "python":
        classes = _python_line_classes(source)
        return classes[1] if classes else set()
    masked_lines = mask_c_like(source, language).splitlines()
    lines: set[int] = set()
    if language == "go":
        in_block = False
        for number, line in enumerate(masked_lines, start=1):
            stripped = line.strip()
            if in_block:
                lines.add(number)
                if stripped.startswith(")"):
                    in_block = False
            elif re.match(r"^import\s*\(", line):
                lines.add(number)
                in_block = not stripped.endswith(")")
            elif re.match(r"^import\s+(?:[\w.]+\s+)?\"", line):
                lines.add(number)
        return lines
    open_braces = 0  # > 0 while inside a multi-line `import { a, b } from 'x'` list
    for number, line in enumerate(masked_lines, start=1):
        if open_braces > 0:
            lines.add(number)
            open_braces += line.count("{") - line.count("}")
        elif _JS_IMPORT_START.match(line) or _JS_REQUIRE.match(line):
            lines.add(number)
            open_braces = max(0, line.count("{") - line.count("}"))
    return lines


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
    # Comments removed, string contents kept. Every line that touches a string, template, regex or
    # raw-string literal is code (even if it is whitespace-only inside the literal); a block-comment
    # continuation line is not.
    text = strip_comments(language, source)
    spans: list[tuple[int, int]] = []
    mask_c_like(text, language, spans)
    in_literal: set[int] = set()
    for start, end in spans:
        first = text.count("\n", 0, start) + 1
        in_literal.update(range(first, first + text.count("\n", start, end) + 1))
    for number, line in enumerate(text.split("\n"), start=1):
        if number > len(lines):
            break
        if number in in_literal:
            continue
        if not line.strip():
            noise.add(number)
        elif _BRACKETS_ONLY.match(line):
            brackets.add(number)
    return noise | import_statement_lines(language, source), brackets


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


_JS_IMPORT_FROM = re.compile(r"^(?:import|export)\s+(?:type\s+)?(.*?)\s*from\s*(['\"])(.+?)\2", re.S)
_JS_SIDE_EFFECT = re.compile(r"^import\s*(['\"])(.+?)\1")
_JS_REQUIRE_PARTS = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(.+?)\s*=\s*require\(\s*(['\"])(.+?)\2")
_GO_IMPORT_SPEC = re.compile(r'^\s*(?:import\s*\(?\s*)?([\w.]+\s+)?"([^"]+)"')


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
    lines = source.splitlines()
    numbers = sorted(import_statement_lines(language, source))
    if language == "go":
        for n in numbers:
            match = _GO_IMPORT_SPEC.match(lines[n - 1])
            if match:
                pairs.add((match.group(2), (match.group(1) or "").strip()))
        return pairs
    statements: list[str] = []
    previous = -2
    for n in numbers:
        if n == previous + 1 and lines[n - 1][:1].isspace() or (statements and n == previous + 1 and lines[n - 1].lstrip().startswith("}")):
            statements[-1] += "\n" + lines[n - 1]
        else:
            statements.append(lines[n - 1])
        previous = n
    for statement in statements:
        text = statement.strip()
        if match := _JS_IMPORT_FROM.match(text):
            clause, module = match.group(1), match.group(3)
            for part in re.split(r"[{},]", clause):
                part = " ".join(part.split())
                if part:
                    pairs.add((module, part))
        elif match := _JS_SIDE_EFFECT.match(text):
            pairs.add((match.group(2), "<side-effect>"))
        elif match := _JS_REQUIRE_PARTS.match(text):
            pairs.add((match.group(3), " ".join(match.group(1).split())))
        else:
            pairs.add(("<statement>", re.sub(r"\s+", "", text)))
    return pairs


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
    call = re.compile(rf"(?<![\w$]){re.escape(name)}\s*\(")
    definition = re.compile(rf"(?:\bfunction\s*\*?\s*|\bfunc\s*(?:\([^)]*\)\s*)?){re.escape(name)}\s*[\[(<]")
    found = []
    for number, line in enumerate(mask_c_like(source, language).splitlines(), start=1):
        if call.search(line) and not definition.search(line):
            found.append(number)
    return found


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
    masked = mask_c_like(source, language)
    # The mask blanks string contents but keeps comment delimiters, so a `//` or `/*` that is
    # still visible in the mask starts a real comment. String contents are kept in the result.
    result = []
    in_comment = False
    i = 0
    while i < len(source):
        if not in_comment and masked.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
            continue
        if not in_comment and masked.startswith("/*", i):
            # a removed block comment still separates tokens: `a+/**/++b` must not become `a+++b`
            result.append(" ")
            in_comment = True
            i += 2
            continue
        if in_comment:
            if source.startswith("*/", i):
                in_comment = False
                i += 2
            else:
                if source[i] == "\n":  # keep line structure inside block comments
                    result.append("\n")
                i += 1
            continue
        result.append(source[i])
        i += 1
    return "".join(result)


# identifiers, numbers, operator runs (so `a + ++b` and `a++ + b` differ), any other single character
_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d[\w.]*|[-+*/%=<>!&|^~?:.]+|\S")


def _python_tokens(source: str) -> list[str] | None:
    """Tokens with block structure: Python indentation is significant, so INDENT/DEDENT and
    statement ends are kept as tokens (moving a statement into or out of a block is not cosmetic)."""
    import textwrap

    try:
        stream = list(tokenize.generate_tokens(io.StringIO(textwrap.dedent(source)).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
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
    """Comment- and layout-insensitive tokens; Python keeps its significant block structure."""
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
    return _c_like_tokens(language, strip_comments(language, source))


def _c_like_tokens(language: str, text: str) -> list[str]:
    """Tokens of comment-free code; string, template and regex literal bodies stay one exact token.

    Line breaks outside literals are kept as a `<NL>` token (runs collapse, so blank lines do not
    matter): with automatic semicolon insertion `return\n  !x` and `return !x` behave differently,
    so a line break is never treated as cosmetic.
    """
    spans: list[tuple[int, int]] = []
    mask_c_like(text, language, spans)
    out: list[str] = []

    def code(segment: str) -> None:
        for index, part in enumerate(segment.split("\n")):
            if index and out and out[-1] != "<NL>":
                out.append("<NL>")
            out.extend(_TOKEN.findall(part))

    position = 0
    for start, end in sorted(spans):
        if start < position:
            continue
        code(text[position:start])
        out.append("\x00" + text[start:end])
        position = end
    code(text[position:])
    while out and out[-1] == "<NL>":
        out.pop()
    return out


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
