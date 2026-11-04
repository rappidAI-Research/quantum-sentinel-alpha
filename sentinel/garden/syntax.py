"""Parse trees for Go, JavaScript and TypeScript (tree-sitter with pinned grammars).

Everything structural about these languages - where strings, template literals, regular
expressions and comments are, which tokens a function consists of, where functions start and end,
which calls, imports and local names a function has - is read from a real parse tree. Earlier
versions re-derived this with hand-written scanners and regular expressions, which kept missing
language corner cases (a regex after `if (...)`, `===!` vs `=== !`, braces inside raw strings).

The grammar versions are part of the dataset's reproducibility: they are pinned in
`pyproject.toml` and recorded in `stats.json` (`syntax_versions`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Iterator

import tree_sitter
import tree_sitter_go
import tree_sitter_javascript
import tree_sitter_typescript

GRAMMAR_PACKAGES = ("tree-sitter", "tree-sitter-go", "tree-sitter-javascript", "tree-sitter-typescript")

_LANGUAGES = {
    "go": tree_sitter_go.language,
    "javascript": tree_sitter_javascript.language,
    "typescript": tree_sitter_typescript.language_typescript,
    "tsx": tree_sitter_typescript.language_tsx,
}

COMMENTS = {"comment", "html_comment", "hash_bang_line"}
STRING_LITERALS = {"string", "interpreted_string_literal", "raw_string_literal", "rune_literal"}
TEMPLATE = "template_string"
REGEX = "regex"
FUNCTION_NODES = {
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "function",
    "generator_function",
    "arrow_function",
    "method_definition",
}
GO_FUNCTION_NODES = {"function_declaration", "method_declaration"}
ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "use", "all", "options", "head"}
BRACKETS = {"(", ")", "{", "}", "[", "]", ";", ","}


def versions() -> dict[str, str]:
    """Installed versions of the parser and grammars (recorded with the dataset)."""
    found = {}
    for package in GRAMMAR_PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = "missing"
    return found


_local = threading.local()


def _parser(grammar: str) -> tree_sitter.Parser:
    parsers = getattr(_local, "parsers", None)
    if parsers is None:
        parsers = _local.parsers = {}
    if grammar not in parsers:
        parsers[grammar] = tree_sitter.Parser(tree_sitter.Language(_LANGUAGES[grammar]()))
    return parsers[grammar]


def _error_count(node: tree_sitter.Node) -> int:
    if not node.has_error:
        return 0
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_error or current.is_missing:
            count += 1
        if current.has_error:
            stack.extend(current.children)
    return count


class Source:
    """A parsed source text with character offsets and 1-based line numbers."""

    def __init__(self, language: str, text: str) -> None:
        if language not in {"go", "javascript", "typescript"}:
            raise ValueError(f"no tree-sitter grammar for {language}")
        self.language = language
        self.text = text
        self.data = text.encode("utf-8")
        grammars = ["typescript", "tsx"] if language == "typescript" else [language]
        best = None
        for grammar in grammars:  # .ts and .tsx share the language key; keep the cleaner parse
            tree = _parser(grammar).parse(self.data)
            errors = _error_count(tree.root_node)
            if best is None or errors < best[0]:
                best = (errors, tree)
            if errors == 0:
                break
        assert best is not None
        self.tree = best[1]
        self.root = self.tree.root_node
        self._chars: list[int] | None = None
        if len(self.data) != len(text):
            chars = []
            for index, char in enumerate(text):
                chars.extend([index] * len(char.encode("utf-8")))
            chars.append(len(text))
            self._chars = chars

    def char(self, byte: int) -> int:
        return byte if self._chars is None else self._chars[byte]

    def span(self, node: tree_sitter.Node) -> tuple[int, int]:
        return self.char(node.start_byte), self.char(node.end_byte)

    def node_text(self, node: tree_sitter.Node) -> str:
        return self.data[node.start_byte : node.end_byte].decode("utf-8", "replace")

    @staticmethod
    def first_line(node: tree_sitter.Node) -> int:
        return node.start_point[0] + 1

    @staticmethod
    def last_line(node: tree_sitter.Node) -> int:
        row, column = node.end_point
        return row if column == 0 and row > node.start_point[0] else row + 1

    def walk(self, node: tree_sitter.Node | None = None) -> Iterator[tree_sitter.Node]:
        """All nodes in source order (pre-order)."""
        stack = [node or self.root]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))


@lru_cache(maxsize=512)
def parse(language: str, text: str) -> Source:
    return Source(language, text)


# --------------------------------------------------------------------------------------------
# Literals, comments, tokens
# --------------------------------------------------------------------------------------------


def literal_bodies(src: Source, node: tree_sitter.Node) -> list[tuple[int, int]]:
    """Character spans of a literal's contents (delimiters excluded; `${...}` code excluded)."""
    start, end = src.span(node)
    if node.type == REGEX:
        pattern = next((c for c in node.children if c.type == "regex_pattern"), None)
        return [src.span(pattern)] if pattern is not None else []
    if node.type == TEMPLATE:
        spans = []
        cursor = start + 1
        for child in node.children:
            if child.type == "template_substitution":
                s, e = src.span(child)
                spans.append((cursor, s))
                cursor = e
        closing = end - 1 if node.children and node.children[-1].type == "`" and not node.children[-1].is_missing else end
        spans.append((cursor, closing))
        return [(s, e) for s, e in spans if e >= s]
    closing_missing = bool(node.children) and node.children[-1].is_missing
    return [(start + 1, end if closing_missing else end - 1)] if end - start >= 2 else []


def literals(src: Source) -> Iterator[tree_sitter.Node]:
    for node in src.walk():
        if node.type in STRING_LITERALS or node.type in (TEMPLATE, REGEX):
            yield node


def comments(src: Source) -> Iterator[tree_sitter.Node]:
    for node in src.walk():
        if node.type in COMMENTS:
            yield node


def jsx_text_value(text: str) -> str:
    """The string a JSX text child renders as (Babel's `cleanJSXElementLiteralChild`): lines are
    trimmed except at the outer edges, whitespace-only lines dropped, the rest joined by one space;
    spaces inside a line are kept, so `<pre>a + b</pre>` and `<pre>a+b</pre>` differ."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    last_non_empty = max((i for i, line in enumerate(lines) if line.strip(" \t")), default=-1)
    result = ""
    for i, line in enumerate(lines):
        trimmed = line.replace("\t", " ")
        if i > 0:
            trimmed = trimmed.lstrip(" ")
        if i < len(lines) - 1:
            trimmed = trimmed.rstrip(" ")
        if trimmed:
            if i != last_non_empty:
                trimmed += " "
            result += trimmed
    return result


def _jsx_context(node: tree_sitter.Node, inherited: bool) -> bool:
    """Whether `node` is JSX markup (tags, attributes, text) rather than JavaScript code."""
    if node.type == "jsx_expression":
        return False  # `{ ... }` inside JSX is code again
    if node.type.startswith("jsx_"):
        return True
    return inherited


def tokens(src: Source, split_text) -> list[tuple[str, int, int, bool]]:
    """Significant tokens as (text, first line, last line, is JSX markup).

    Comments are dropped; every string and regex literal is one exact token; template-literal text
    pieces are exact while their `${...}` code is tokenized; operators are the grammar's own tokens
    (`===!` is `===` `!`); JSX text is one token with the exact string React renders.
    """
    out: list[tuple[str, int, int, bool]] = []
    stack: list = [(src.root, False)]
    while stack:
        item, markup = stack.pop()
        if isinstance(item, tuple):  # a template-literal text piece
            out.append((*item, markup))
            continue
        current = item
        markup = _jsx_context(current, markup)
        if current.type in COMMENTS or current.is_missing:
            continue
        if current.type in STRING_LITERALS or current.type == REGEX:
            out.append((src.node_text(current), src.first_line(current), src.last_line(current), markup))
            continue
        if current.type == TEMPLATE:
            parts: list = []
            cursor = current.start_byte
            cursor_line = current.start_point[0] + 1
            for child in current.children:
                if child.type == "template_substitution":
                    text = src.data[cursor:child.start_byte].decode("utf-8", "replace")
                    parts.append(((text, cursor_line, child.start_point[0] + 1), markup))
                    parts.append((child, markup))
                    cursor = child.end_byte
                    cursor_line = child.end_point[0] + 1
            text = src.data[cursor:current.end_byte].decode("utf-8", "replace")
            parts.append(((text, cursor_line, src.last_line(current)), markup))
            stack.extend(reversed(parts))
            continue
        if current.type == "jsx_text":
            rendered = jsx_text_value(src.node_text(current))
            if rendered:  # JSX text is output: kept exactly as React/Babel render it
                out.append(("jsx:" + rendered, src.first_line(current), current.end_point[0] + 1, True))
            continue
        if current.child_count == 0:
            text = src.node_text(current)
            if not text.strip():
                continue  # automatic semicolon / newline tokens carry no text of their own
            first, last = src.first_line(current), src.last_line(current)
            if any(c.isspace() for c in text):  # e.g. an ERROR token spanning code
                out.extend((part, first, last, markup) for part in split_text(text))
            else:
                out.append((text, first, last, markup))
            continue
        stack.extend((child, markup) for child in reversed(current.children))
    return out


def code_lines(src: Source) -> tuple[set[int], set[int]]:
    """(lines with code, lines whose code is brackets/separators only). Lines inside a multi-line
    literal are code; comment-only and blank lines are neither."""
    code: set[int] = set()
    non_bracket: set[int] = set()
    stack = [src.root]
    while stack:
        node = stack.pop()
        if node.type in COMMENTS or node.is_missing:
            continue
        if node.type in STRING_LITERALS or node.type in (TEMPLATE, REGEX):
            lines = range(src.first_line(node), src.last_line(node) + 1)
            code.update(lines)
            non_bracket.update(lines)
            if node.type == TEMPLATE:
                stack.extend(c for c in node.children if c.type == "template_substitution")
            continue
        if node.child_count == 0:
            text = src.node_text(node)
            if not text.strip():
                continue
            lines = range(src.first_line(node), src.last_line(node) + 1)
            code.update(lines)
            if text not in BRACKETS:
                non_bracket.update(lines)
            continue
        stack.extend(node.children)
    return code, code - non_bracket


# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionNode:
    name: str
    start: int
    end: int
    node: tree_sitter.Node
    kind: str  # declaration | method | assigned | object | variable | anonymous | go_function | go_method
    owner: str | None  # object path for `owner.name = function`, receiver type for Go methods


def _strip_quotes(text: str) -> str:
    return text[1:-1] if len(text) >= 2 and text[0] in "'\"" and text[-1] == text[0] else text


def _statement_start(node: tree_sitter.Node) -> tree_sitter.Node:
    """The node a declaration's excerpt starts at (`export` included)."""
    parent = node.parent
    if parent is not None and parent.type == "export_statement":
        return parent
    return node


def _js_function(src: Source, node: tree_sitter.Node) -> FunctionNode | None:
    body = node.child_by_field_name("body")
    if body is None or body.type != "statement_block":
        return None
    parent = node.parent
    start = node
    name = None
    kind = "anonymous"
    owner = None
    if node.type in {"function_declaration", "generator_function_declaration"}:
        name_node = node.child_by_field_name("name")
        name = src.node_text(name_node) if name_node is not None else None
        kind = "declaration"
        start = _statement_start(node)
    elif node.type == "method_definition":
        name_node = node.child_by_field_name("name")
        if name_node is not None and name_node.type in {"property_identifier", "private_property_identifier", "string", "number"}:
            name = _strip_quotes(src.node_text(name_node))
        kind = "method"
    elif parent is not None and parent.type == "variable_declarator" and parent.child_by_field_name("value") == node:
        target = parent.child_by_field_name("name")
        if target is not None and target.type == "identifier":
            name = src.node_text(target)
        kind = "variable"
        declaration = parent.parent
        if declaration is not None and declaration.type in {"lexical_declaration", "variable_declaration"}:
            first = next((c for c in declaration.children if c.type == "variable_declarator"), None)
            start = _statement_start(declaration) if first == parent else parent
    elif parent is not None and parent.type == "assignment_expression" and parent.child_by_field_name("right") == node:
        left = parent.child_by_field_name("left")
        start = parent
        if left is not None and left.type == "identifier":
            name, kind = src.node_text(left), "variable"
        elif left is not None and left.type == "member_expression":
            prop = left.child_by_field_name("property")
            obj = left.child_by_field_name("object")
            if prop is not None:
                name = src.node_text(prop)
                kind = "assigned"
                owner = "".join(src.node_text(obj).split()) if obj is not None else None
    elif parent is not None and parent.type == "pair" and parent.child_by_field_name("value") == node:
        key = parent.child_by_field_name("key")
        if key is not None and key.type in {"property_identifier", "string", "number"}:
            name = _strip_quotes(src.node_text(key))
        kind, start = "object", parent
    elif parent is not None and parent.type in {"public_field_definition", "field_definition"}:
        key = parent.child_by_field_name("name") or parent.child_by_field_name("property")
        if key is not None:
            name = _strip_quotes(src.node_text(key))
        kind, start = "method", parent
    if name is None and node.type in {"function_expression", "function", "generator_function"}:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            name = src.node_text(name_node)
    if name is None:
        kind = "anonymous"
        name = "<anonymous>"
        if parent is not None and parent.type == "arguments":
            call = parent.parent
            callee = call.child_by_field_name("function") if call is not None and call.type == "call_expression" else None
            if callee is not None and callee.type == "member_expression":
                prop = callee.child_by_field_name("property")
                method = src.node_text(prop) if prop is not None else ""
                if method in ROUTE_METHODS:
                    name = f"<anonymous:{method}-handler>"
    return FunctionNode(name, src.first_line(start), src.last_line(node), node, kind, owner)


def _go_receiver_type(src: Source, node: tree_sitter.Node) -> tuple[str, str] | None:
    """(receiver variable, receiver type name) of a Go method declaration."""
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    declaration = next((c for c in receiver.children if c.type == "parameter_declaration"), None)
    if declaration is None:
        return None
    name = declaration.child_by_field_name("name")
    type_node = declaration.child_by_field_name("type")
    while type_node is not None and type_node.type in {"pointer_type", "generic_type", "parenthesized_type"}:
        inner = type_node.child_by_field_name("type")
        if inner is None:
            inner = next((c for c in type_node.named_children), None)
        type_node = inner
    if type_node is None:
        return None
    return (src.node_text(name) if name is not None else "", src.node_text(type_node))


def functions(src: Source) -> list[FunctionNode]:
    found: dict[tuple[int, int], FunctionNode] = {}
    for node in src.walk():
        if src.language == "go":
            if node.type in GO_FUNCTION_NODES and node.child_by_field_name("body") is not None:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                receiver = _go_receiver_type(src, node) if node.type == "method_declaration" else None
                fn = FunctionNode(
                    src.node_text(name_node),
                    src.first_line(node),
                    src.last_line(node),
                    node,
                    "go_method" if node.type == "method_declaration" else "go_function",
                    receiver[1] if receiver else None,
                )
                found.setdefault((fn.start, fn.end), fn)
            continue
        if node.type in FUNCTION_NODES:
            fn = _js_function(src, node)
            if fn is not None:
                found.setdefault((fn.start, fn.end), fn)
    return sorted(found.values(), key=lambda f: (f.start, f.end))


def function_at(src: Source, start: int, end: int) -> FunctionNode | None:
    return next((f for f in functions(src) if f.start == start and f.end == end), None)


def enclosing_function(src: Source, node: tree_sitter.Node) -> tree_sitter.Node | None:
    parent = node.parent
    kinds = GO_FUNCTION_NODES | {"func_literal"} if src.language == "go" else FUNCTION_NODES
    while parent is not None:
        if parent.type in kinds:
            return parent
        parent = parent.parent
    return None


def enclosing_class(src: Source, node: tree_sitter.Node) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type in {"class_declaration", "class", "abstract_class_declaration"}:
            name = parent.child_by_field_name("name")
            label = src.node_text(name) if name is not None else "<anonymous>"
            return f"{label}@{src.first_line(parent)}"
        if parent.type in FUNCTION_NODES and parent is not node:
            return None  # a class declared around a nested function is not this method's class
        parent = parent.parent
    return None


# --------------------------------------------------------------------------------------------
# Calls, identifiers, imports
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    name: str
    line: int
    receiver: str | None  # None for a bare call; "this", an object path, or a Go operand otherwise
    callee: tree_sitter.Node  # the identifier / property node naming the function


def calls(src: Source) -> list[Call]:
    found = []
    for node in src.walk():
        if node.type == "call_expression":
            target = node.child_by_field_name("function")
        elif node.type == "new_expression":
            target = node.child_by_field_name("constructor")
        else:
            continue
        if target is None:
            continue
        if target.type == "identifier":
            found.append(Call(src.node_text(target), src.first_line(target), None, target))
        elif target.type == "member_expression":
            prop = target.child_by_field_name("property")
            obj = target.child_by_field_name("object")
            if prop is not None and obj is not None:
                found.append(Call(src.node_text(prop), src.first_line(prop), "".join(src.node_text(obj).split()), prop))
        elif target.type == "selector_expression":
            field = target.child_by_field_name("field")
            operand = target.child_by_field_name("operand")
            if field is not None and operand is not None:
                found.append(Call(src.node_text(field), src.first_line(field), "".join(src.node_text(operand).split()), field))
    return found


IDENTIFIER_NODES = {"identifier", "shorthand_property_identifier", "shorthand_property_identifier_pattern"}


def identifiers(src: Source, name: str) -> list[tree_sitter.Node]:
    """Every identifier node spelling `name` that refers to a variable (not a property/field name)."""
    return [n for n in src.walk() if n.type in IDENTIFIER_NODES and src.node_text(n) == name]


def _top_level(src: Source) -> list[tree_sitter.Node]:
    return list(src.root.children)


def _require_member(src: Source, node: tree_sitter.Node | None) -> tuple[str, str] | None:
    """(module, property) for `require('m').prop`, else None."""
    if node is None or node.type != "member_expression":
        return None
    module = _is_require_call(src, node.child_by_field_name("object"))
    prop = node.child_by_field_name("property")
    return (module, src.node_text(prop)) if module is not None and prop is not None else None


def _is_require_call(src: Source, node: tree_sitter.Node | None) -> str | None:
    """The module name of `require('m')` (or of `require('m').prop`), else None."""
    member = _require_member(src, node)
    if member is not None:
        return member[0]
    if node is None or node.type != "call_expression":
        return None
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    if fn is None or args is None or src.node_text(fn) != "require":
        return None
    strings = [c for c in args.named_children]
    if len(strings) != 1 or strings[0].type != "string":
        return None
    return _strip_quotes(src.node_text(strings[0]))


def import_nodes(src: Source) -> list[tree_sitter.Node]:
    """Module-level import statements (JS/TS: import / export-from / top-level require declarations)."""
    found = []
    for node in _top_level(src):
        if src.language == "go":
            if node.type == "import_declaration":
                found.append(node)
            continue
        if node.type == "import_statement":
            found.append(node)
        elif node.type == "export_statement" and node.child_by_field_name("source") is not None:
            found.append(node)
        else:
            declaration = node
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration") or node
            if declaration.type in {"lexical_declaration", "variable_declaration"}:
                declarators = [c for c in declaration.named_children if c.type == "variable_declarator"]
                if declarators and all(_is_require_call(src, d.child_by_field_name("value")) for d in declarators):
                    found.append(node)
    return found


def imported_bindings(src: Source) -> set[tuple[str, str]]:
    """(module, "imported as local" | "local" | "<side-effect>") for module-level imports."""
    pairs: set[tuple[str, str]] = set()
    for node in import_nodes(src):
        if src.language == "go":
            for spec in (n for n in src.walk(node) if n.type == "import_spec"):
                path = spec.child_by_field_name("path")
                alias = spec.child_by_field_name("name")
                if path is not None:
                    pairs.add((_strip_quotes(src.node_text(path)), src.node_text(alias) if alias is not None else ""))
            continue
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            require = next((c for c in node.children if c.type == "import_require_clause"), None)
            if require is not None:
                module_node = require.child_by_field_name("source")
                local = next((c for c in require.children if c.type == "identifier"), None)
                if module_node is not None and local is not None:
                    pairs.add((_strip_quotes(src.node_text(module_node)), src.node_text(local)))
                continue
            if source is None:
                continue
            module = _strip_quotes(src.node_text(source))
            clause = next((c for c in node.children if c.type == "import_clause"), None)
            if clause is None:
                pairs.add((module, "<side-effect>"))
                continue
            for part in clause.named_children:
                if part.type == "identifier":
                    pairs.add((module, src.node_text(part)))
                elif part.type == "namespace_import":
                    local = next((c for c in part.named_children if c.type == "identifier"), None)
                    if local is not None:
                        pairs.add((module, f"* as {src.node_text(local)}"))
                elif part.type == "named_imports":
                    for spec in (c for c in part.named_children if c.type == "import_specifier"):
                        imported = spec.child_by_field_name("name")
                        alias = spec.child_by_field_name("alias")
                        if imported is None:
                            continue
                        name = _strip_quotes(src.node_text(imported))
                        pairs.add((module, f"{name} as {src.node_text(alias)}" if alias is not None else name))
            continue
        declaration = node
        if node.type == "export_statement":
            declaration = node.child_by_field_name("declaration")
            if declaration is None:
                continue  # `export ... from` binds no local name
        for declarator in (c for c in declaration.named_children if c.type == "variable_declarator"):
            module = _is_require_call(src, declarator.child_by_field_name("value"))
            target = declarator.child_by_field_name("name")
            if module is None or target is None:
                continue
            member = _require_member(src, declarator.child_by_field_name("value"))
            if target.type == "identifier" and member is not None:
                pairs.add((module, f"{member[1]} as {src.node_text(target)}"))
            elif target.type == "identifier":
                pairs.add((module, src.node_text(target)))
            elif target.type == "object_pattern":
                for part in target.named_children:
                    if part.type == "shorthand_property_identifier_pattern":
                        pairs.add((module, src.node_text(part)))
                    elif part.type == "pair_pattern":
                        key = part.child_by_field_name("key")
                        value = part.child_by_field_name("value")
                        if key is not None and value is not None and value.type == "identifier":
                            pairs.add((module, f"{_strip_quotes(src.node_text(key))} as {src.node_text(value)}"))
                    elif part.type == "object_assignment_pattern":
                        left = part.child_by_field_name("left")
                        if left is not None:
                            pairs.add((module, src.node_text(left)))
            else:
                pairs.add((module, "".join(src.node_text(target).split())))
    return pairs


def import_lines(src: Source) -> set[int]:
    lines: set[int] = set()
    for node in import_nodes(src):
        lines.update(range(src.first_line(node), src.last_line(node) + 1))
    return lines
