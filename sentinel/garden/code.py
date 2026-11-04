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


def mask_c_like(source: str, language: str) -> str:
    """Replace the contents of strings, comments, template literals and regex literals by spaces.

    Newlines are kept so line numbers stay valid; quote and delimiter characters are kept.
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
    lines = source.splitlines()
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
        limit = min(len(masked), k + 300)
        between = []
        while k < limit:
            c = masked[k]
            if c == "{":
                text = "".join(between)
                # allow return-type annotations and `=>`, but not a statement end or assignment
                if ";" in text or re.search(r"(?<![=!<>])=(?!>)", text):
                    return None
                return k
            if c in ";,)":
                return None
            between.append(c)
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

_IMPORT_LINE = {
    "python": re.compile(r"^\s*(?:import\s+\S|from\s+\S+\s+import\s)"),
    "javascript": re.compile(r"^\s*(?:import\s.+from\s|import\s+['\"]|(?:const|let|var)\s+.+=\s*require\()"),
    "typescript": re.compile(r"^\s*(?:import\s.+from\s|import\s+['\"]|(?:const|let|var)\s+.+=\s*require\()"),
    "go": re.compile(r"^\s*(?:import\s+(?:\w+\s+)?\"[^\"]+\"|(?:\w+\s+)?\"[^\"]+\"\s*$)"),
}


def import_lines(language: str, source: str, limit: int = 15) -> list[str]:
    pattern = _IMPORT_LINE[language]
    found: list[str] = []
    in_go_block = False
    for line in source.splitlines():
        stripped = line.strip()
        if language == "go":
            if stripped.startswith("import ("):
                in_go_block = True
                continue
            if in_go_block and stripped == ")":
                in_go_block = False
                continue
            if not in_go_block and not stripped.startswith("import "):
                continue
        if pattern.match(line) and not stripped.endswith(("(", "{", ",")):  # skip incomplete multi-line imports
            found.append(stripped)
            if len(found) >= limit:
                break
    return found


def is_import_or_trivial(language: str, line: str) -> bool:
    """A changed line that carries no logic: blank, comment, import or a lone bracket."""
    stripped = line.strip()
    if not stripped or stripped in {"(", ")", "{", "}", "};", "});", "]"}:
        return True
    if is_comment_line(language, stripped):
        return True
    return bool(_IMPORT_LINE[language].match(line)) or (language == "go" and stripped.startswith("import"))


def is_comment_line(language: str, stripped: str) -> bool:
    if language == "python":
        return stripped.startswith("#")
    return stripped.startswith(("//", "/*", "*", "*/"))


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
            in_comment = True
            i += 2
            continue
        if in_comment:
            if source.startswith("*/", i):
                in_comment = False
                i += 2
            else:
                i += 1
            continue
        result.append(source[i])
        i += 1
    return "".join(result)


_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d+|\S")


def normalized_tokens(language: str, source: str) -> list[str]:
    return _TOKEN.findall(strip_comments(language, source))


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
