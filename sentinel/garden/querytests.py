"""Garden v6: CodeQL security query tests as rule-scoped tool-test records.

Source: the security query tests of github/codeql (MIT) at one pinned revision, for Python, JavaScript /
TypeScript and Go: `<lang>/ql/test/query-tests/Security/**`. Only tests that use CodeQL's inline
expectations (`postprocess: utils/test/InlineExpectationsTestQuery.ql`) are read: their `$ Alert`,
`$ SPURIOUS: Alert` and `$ MISSING: Alert` comments are checked against the query's actual results by
CodeQL's own test runner, so a marker states what the query reports and what the test author expects.

A query expectation establishes the expected behaviour of one rule, not that code is secure:
- `query_test_finding`: a function whose `$ Alert` lines are all real query results the author
  expects (no SPURIOUS or MISSING marker in the function). For a path-problem query every alert's
  source (from the `.expected` result table) lies in the same function, so the excerpt shows the flow.
  CWE: the query's `external/cwe` tags. Evidence: the query's alert message.
- `query_test_negative`: a function of the same test files with no expected alert and no MISSING
  marker, that the test marks safe for this rule: a `$ SPURIOUS: Alert` (the query reports it, the
  author says it is a false positive), or a `$ Source` whose value reaches no reported sink, or an
  explicit OK / GOOD / safe / sanitized comment. Its `label_scope` names the rule; other weaknesses are
  not verified, so the kind stays pending until its independent review sample validates it
  (governance.validated_negative_kinds).

Answer-revealing cues are removed from the input or the function is left out: every comment and
docstring (all markers and GOOD/BAD notes) is dropped from the excerpt; functions whose name says
what they are (bad, good, safe, unsafe, vuln, ...) and files whose name names the weakness are not
used; the input file name is the test file's base name.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path, PurePosixPath
from typing import Any

from sentinel.garden import code

REPOSITORY = "https://github.com/github/codeql"
REPOSITORY_KEY = "github/codeql"
LANGUAGE_ROOTS = {
    "python": "python/ql",
    "javascript": "javascript/ql",
    "go": "go/ql",
}
SPARSE_PATHS = ["/LICENSE"] + [f"/{root}/test/query-tests/Security/" for root in LANGUAGE_ROOTS.values()] + [
    f"/{root}/src/Security/" for root in LANGUAGE_ROOTS.values()
]
INLINE_POSTPROCESS = "utils/test/InlineExpectationsTestQuery.ql"

_MARKER = re.compile(r"\$\s*(SPURIOUS:\s*|MISSING:\s*)?(Alert|Source|Sink)\b(\[[^\]]*\])?")
_LOCATION = re.compile(r"^(?P<file>[^:|]+):(?P<l1>\d+):(?P<c1>\d+):(?P<l2>\d+):(?P<c2>\d+)$")
_CWE_TAG = re.compile(r"external/cwe/cwe-0*(\d+)", re.IGNORECASE)
# a function or file named for what it demonstrates gives the answer away
_REVEALING_NAME = re.compile(
    r"(?i)(bad|good|safe|unsafe|vuln|insecure|secure|evil|attack|malicious|exploit|taint|sanitiz|sanitis|"
    r"inject|xss|ssrf|csrf|traversal|redos|xxe|leak|danger|hardcoded|weak|cleartext|(^|_)ok\d*$|(^|_)nok|notok|fixed)"
)
_NOT_SAFE_COMMENT = re.compile(r"(?i)\b(NOT\s*OK|BAD|unsafe|insecure|vulnerab|TODO|FIXME|false negative|should (be|have)|missing|not (yet )?(detected|flagged|found))\b")
_SAFE_COMMENT = re.compile(r"(?i)(\bOK\b|\bGOOD\b|\bsafe\b|\bsaniti[sz]ed\b|\bescaped\b)")
TOOL_TEST_LANGUAGES = {"python", "javascript", "typescript", "go"}


@dataclass
class Query:
    path: str  # relative to <lang>/ql/src
    query_id: str
    name: str
    kind: str
    precision: str
    cwes: list[str]


@dataclass
class Result:
    file: str
    line: int
    end_line: int
    source_file: str | None
    source_line: int | None
    message: str


@dataclass
class TestUnit:
    language_root: str  # python | javascript | go
    directory: str  # repository-relative
    query: Query
    results: list[Result] = field(default_factory=list)


def ensure_checkout(cache: Path, revision: str, log: Any = print) -> Path:
    """A sparse, blob-filtered checkout of the pinned revision (only the test and query directories)."""
    target = cache / "codeql"
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth=1", "--filter=blob:none", "--no-checkout", "--sparse", REPOSITORY + ".git", str(target)], check=True, timeout=1800)
    subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", "--no-cone", *SPARSE_PATHS], check=True, timeout=600)
    present = subprocess.run(["git", "-C", str(target), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True, check=False).returncode == 0
    if not present:
        subprocess.run(["git", "-C", str(target), "fetch", "-q", "--depth=1", "--filter=blob:none", "origin", revision], check=True, timeout=1800)
    subprocess.run(["git", "-C", str(target), "checkout", "-q", "--force", "--detach", revision], check=True, timeout=1800)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != revision:
        raise RuntimeError(f"CodeQL checkout is at {head}, not the pinned {revision}")
    log(f"querytests: checkout of {REPOSITORY}@{revision[:12]} ready")
    return target


def commit_date(checkout: Path, revision: str) -> str:
    """The pinned revision's commit time in UTC (`2026-11-25T12:02:02Z`)."""
    from datetime import datetime

    out = subprocess.run(["git", "-C", str(checkout), "show", "-s", "--format=%ct", revision], capture_output=True, text=True, check=True)
    return datetime.fromtimestamp(int(out.stdout.strip()), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_query(text: str, path: str) -> Query | None:
    meta: dict[str, str] = {}
    for key in ("id", "name", "kind", "precision"):
        m = re.search(rf"@{key}\s+(.+)", text)
        if m:
            meta[key] = m.group(1).strip()
    tags = re.search(r"@tags(.*?)\*/", text, re.DOTALL)
    cwes = sorted({f"CWE-{int(n)}" for n in _CWE_TAG.findall(tags.group(1) if tags else "")}, key=lambda c: int(c[4:]))
    if "id" not in meta or meta.get("kind") not in {"problem", "path-problem"}:
        return None
    return Query(path, meta["id"], meta.get("name", ""), meta["kind"], meta.get("precision", ""), cwes)


def _location(cell: str) -> tuple[str, int, int] | None:
    m = _LOCATION.match(cell.strip())
    return (m.group("file"), int(m.group("l1")), int(m.group("l2"))) if m else None


def parse_expected(text: str, kind: str) -> list[Result]:
    """Rows of the `#select` table of a `.expected` file."""
    results: list[Result] = []
    section = None
    for line in text.splitlines():
        if not line.startswith("|"):
            section = line.strip()
            continue
        if section != "#select":
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        alert = _location(cells[0]) if cells else None
        if alert is None:
            continue
        if kind == "path-problem" and len(cells) >= 7:
            source = _location(cells[2])
            message, rest = cells[6], cells[7:]
        else:
            source = None
            message, rest = (cells[2], cells[3:]) if len(cells) >= 3 else ("", [])
        for i in range(0, len(rest) - 2, 3):  # placeholder arguments: location | label | text
            message = message.replace("$@", rest[i + 2], 1)
        results.append(Result(alert[0], alert[1], alert[2], source[0] if source else None, source[1] if source else None, message))
    return results


def discover_units(checkout: Path, log: Any = print) -> list[TestUnit]:
    units: list[TestUnit] = []
    for language_root, root in LANGUAGE_ROOTS.items():
        base = checkout / root / "test" / "query-tests" / "Security"
        for qlref in sorted(base.rglob("*.qlref")):
            text = qlref.read_text(encoding="utf-8")
            m = re.search(r"^query:\s*(\S+)", text, re.MULTILINE)
            if not m or INLINE_POSTPROCESS not in text or not m.group(1).startswith("Security/"):
                continue
            query_file = checkout / root / "src" / m.group(1)
            if not query_file.is_file():
                continue
            query = parse_query(query_file.read_text(encoding="utf-8"), m.group(1))
            if query is None or not query.cwes:
                continue
            expected = qlref.with_suffix(".expected")
            results = parse_expected(expected.read_text(encoding="utf-8"), query.kind) if expected.is_file() else []
            units.append(TestUnit(language_root, str(qlref.parent.relative_to(checkout)), query, results))
    log(f"querytests: {len(units)} inline-expectation security test units")
    return units


def _markers(line: str, query_id: str) -> set[str]:
    """Markers on a line that apply to this query (an untagged marker applies to every query of the test)."""
    found = set()
    for m in _MARKER.finditer(line):
        tag = (m.group(3) or "").strip("[]")
        if tag and tag != query_id:
            continue
        found.add(((m.group(1) or "").strip().rstrip(":").upper() + " " + m.group(2)).strip())
    return found


def _comment_text(language: str, source_lines: list[str], start: int, end: int) -> str:
    excerpt = "\n".join(source_lines[start - 1 : end]) + "\n"
    stripped = code.strip_comments(language, excerpt).splitlines()
    return "\n".join(o.removeprefix(s.rstrip()) for o, s in zip(excerpt.splitlines(), stripped))


def clean_excerpt(language: str, source: str, fn: code.Function) -> tuple[str, dict[int, int]] | None:
    """The function without comments, docstrings and comment-only lines; file line -> excerpt line."""
    lines = source.splitlines()
    excerpt = "\n".join(lines[fn.start - 1 : fn.end]) + "\n"
    try:
        stripped = code.strip_comments(language, excerpt).splitlines()
    except Exception:  # noqa: BLE001 - an excerpt that does not parse is not used
        return None
    drop: set[int] = set()
    if language == "python":
        drop = {fn.start - 1 + n for n in code.python_docstring_lines(excerpt)}
    kept: list[str] = []
    mapping: dict[int, int] = {}
    for offset, (original, text) in enumerate(zip(lines[fn.start - 1 : fn.end], stripped)):
        number = fn.start + offset
        if number in drop or (not text.strip() and original.strip()):
            continue
        mapping[number] = len(kept) + 1
        kept.append(text.rstrip())
    while kept and not kept[-1].strip():
        kept.pop()
    if not kept:
        return None
    return "\n".join(kept) + "\n", mapping


def language_of_test_file(path: str) -> str | None:
    language = code.language_of(path)
    return language if language in TOOL_TEST_LANGUAGES else None


def unit_functions(checkout: Path, unit: TestUnit, min_lines: int, max_lines: int) -> list[dict[str, Any]]:
    """Positive and negative functions of one test unit (no record fields yet)."""
    out: list[dict[str, Any]] = []
    directory = checkout / unit.directory
    by_file: dict[str, list[Result]] = {}
    for result in unit.results:
        by_file.setdefault(result.file, []).append(result)
    sources_with_results = {(r.source_file, r.source_line) for r in unit.results if r.source_file}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = str(path.relative_to(directory))
        language = language_of_test_file(relative)
        if language is None or any(part in {"node_modules", "vendor"} for part in PurePosixPath(relative).parts):
            continue
        if relative.endswith(".d.ts"):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        try:
            functions = code.find_functions(language, source)
        except Exception:  # noqa: BLE001
            continue
        results = by_file.get(relative, [])
        reported = {r.line for r in results}
        if not results and not any(_markers(line, unit.query.query_id) for line in lines):
            continue  # the rule has nothing to say about this file
        all_marks = {n: _markers(lines[n - 1], unit.query.query_id) for n in range(1, len(lines) + 1)}
        for fn in functions:
            if _REVEALING_NAME.search(fn.name.strip("<>").split(":", 1)[-1]):
                continue
            size = fn.end - fn.start + 1
            if not min_lines <= size <= max_lines:
                continue
            inner = [f for f in functions if f is not fn and fn.start <= f.start and f.end <= fn.end]
            outer = [f for f in functions if f is not fn and f.start <= fn.start and fn.end <= f.end]
            if any(all_marks[n] for f in inner for n in range(f.start, f.end + 1)):
                continue  # a marker inside a nested function belongs to that function (the innermost one)
            marks = {n: all_marks[n] for n in range(fn.start, fn.end + 1)}
            alerts = sorted(n for n, m in marks.items() if "Alert" in m)
            spurious = sorted(n for n, m in marks.items() if "SPURIOUS Alert" in m)
            missing = [n for n, m in marks.items() if "MISSING Alert" in m]
            source_lines = [n for n, m in marks.items() if "Source" in m]
            fn_results = [r for r in results if fn.start <= r.line <= fn.end]
            comments = _comment_text(language, lines, fn.start, fn.end)
            if missing:
                continue
            if alerts and not spurious:
                if any(n not in reported for n in alerts) or any(r.line not in alerts for r in fn_results):
                    continue  # markers and results disagree: not a clean expectation
                if unit.query.kind == "path-problem" and any(
                    r.source_file != relative or r.source_line is None or not fn.start <= r.source_line <= fn.end for r in fn_results
                ):
                    continue  # the flow starts outside the excerpt
                out.append({"label": "finding", "file": relative, "language": language, "function": fn, "alert_lines": alerts, "message": fn_results[0].message, "source": source, "lines": (fn.start, fn.end)})
            elif not alerts:
                if outer or _NOT_SAFE_COMMENT.search(comments):  # negatives: outermost functions only
                    continue
                if spurious:
                    reason = "query_false_positive"
                elif source_lines and not any((relative, n) in sources_with_results for n in source_lines):
                    reason = "source_reaches_no_sink"
                elif _SAFE_COMMENT.search(comments) and not fn_results:
                    reason = "marked_safe"
                else:
                    continue
                if fn_results and not spurious:
                    continue
                out.append({"label": "no_finding", "file": relative, "language": language, "function": fn, "reason": reason, "source": source, "lines": (fn.start, fn.end)})
    return out


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------

CODE_SOURCE_ID = "codeql-query-tests"
NEUTRAL_FILES = {
    "python": ["app.py", "views.py", "handlers.py", "server.py", "service.py"],
    "javascript": ["index.js", "app.js", "server.js", "routes.js", "handler.js"],
    "typescript": ["index.ts", "app.ts", "server.ts", "routes.ts", "handler.ts"],
    "go": ["main.go", "server.go", "handler.go", "routes.go", "service.go"],
}
# identifiers a test author chose to say what the code is (a real API such as `sanitize` or `mark_safe` is kept)
_REVEALING_IDENTIFIER = re.compile(r"(?i)[A-Za-z_$]*(bad|good|unsafe|vuln|evil|attacker|malicious|exploit|taint|notok)[A-Za-z0-9_$]*")
LANG_TAG = {"python": "py", "javascript": "js", "typescript": "ts", "go": "go"}


def neutral_file(language: str, path: str) -> str:
    names = NEUTRAL_FILES[language]
    return names[int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % len(names)]


def negative_evidence(query: Query, cwe: str) -> str:
    if query.kind == "path-problem":
        return f"The code resembles a {cwe} pattern, but no value that reaches the sensitive operation here makes it exploitable."
    return f"No {cwe} weakness: the security-relevant operation or setting here is used safely."


def build_records(
    checkout: Path,
    revision: str,
    settings: Any,
    dataset_id: str,
    rejected_cwes: set[str],
    max_cwes: int,
    log: Any = print,
    max_findings_per_file: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(records, build log rows) for every inline-expectation security test unit at `revision`."""
    from sentinel.garden import build as build_module
    from sentinel.garden.licenses import detect

    committed_at = commit_date(checkout, revision)
    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    license_file = {"path": "LICENSE", "sha256": hashlib.sha256((checkout / "LICENSE").read_bytes()).hexdigest()}
    detected = detect({"LICENSE": license_text})
    records: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    for unit in discover_units(checkout, log):
        root = LANGUAGE_ROOTS[unit.language_root]
        query_text = (checkout / root / "src" / unit.query.path).read_text(encoding="utf-8")
        qlref = next((checkout / unit.directory).glob("*.qlref"))
        expected = qlref.with_suffix(".expected")
        unit_sha = hashlib.sha256(
            (query_text + "\x00" + qlref.read_text(encoding="utf-8") + "\x00" + (expected.read_text(encoding="utf-8") if expected.is_file() else "")).encode("utf-8")
        ).hexdigest()
        cwes = [c for c in unit.query.cwes if c not in rejected_cwes][:max_cwes]
        outcome: dict[str, int] = {}

        def count(key: str) -> None:
            outcome[key] = outcome.get(key, 0) + 1

        if not cwes:
            log_rows.append({"advisory_id": f"codeql:{unit.query.query_id}", "repository_key": REPOSITORY_KEY, "test_unit": unit.directory, "outcome": "rejected_cwe", "records": 0})
            continue
        for item in unit_functions(checkout, unit, settings.min_function_lines, settings.max_function_lines):
            language, fn, source = item["language"], item["function"], item["source"]
            cleaned = clean_excerpt(language, source, fn)
            if cleaned is None:
                count("excerpt_not_parseable")
                continue
            excerpt, mapping = cleaned
            if _MARKER.search(excerpt) or _REVEALING_IDENTIFIER.search(excerpt):
                count("revealing_identifier")
                continue
            real_path = f"{unit.directory}/{item['file']}"
            shown = neutral_file(language, real_path)
            lines = excerpt.splitlines()
            if not settings.min_function_lines <= len(lines) <= settings.max_function_lines:
                count("excerpt_size")
                continue
            blocks = [(fn.start, fn.end)]
            note, _ = build_module.callee_note(language, shown, source, blocks, settings)
            callers = build_module._callers(language, source, blocks, [fn.name], settings.max_context_callers)
            if callers:
                note = (note + " " if note else "") + "Called in this file by: " + ", ".join(f"`{c}`" for c in callers) + "."
            context = build_module._context(language, shown, fn.name, source, settings, extra=note)
            if _MARKER.search(context) or _REVEALING_IDENTIFIER.search(context):
                count("revealing_context")
                continue
            if item["label"] == "finding":
                span = sorted(mapping[n] for n in item["alert_lines"] if n in mapping)
                if len(span) != len(item["alert_lines"]) or span[-1] - span[0] + 1 > 10:
                    count("alert_span")
                    continue
                kind, decision = "query_test_finding", "finding"
                target = {
                    "cwe": cwes[0],
                    "alternate_cwes": cwes[1:],
                    "start_line": span[0],
                    "end_line": span[-1],
                    "evidence": item["message"].rstrip(".") + ".",
                    "evidence_source": "query_message",
                    "patch": None,
                }
                localization = {"method": "query_test_expectation", "alert_lines": span}
            else:
                kind, decision = "query_test_negative", "no_finding"
                target = {
                    "cwe": None,
                    "alternate_cwes": [],
                    "start_line": None,
                    "end_line": None,
                    "evidence": negative_evidence(unit.query, cwes[0]),
                    "evidence_source": "derived_template",
                    "patch": None,
                    "label_scope": f"no {cwes[0]} weakness according to the expectations of CodeQL rule {unit.query.query_id}; other weaknesses are not verified",
                }
                localization = {"method": "not_applicable"}
            # one record per rule: a directory can test several rules on the same file; identical excerpts of
            # several rules are then exact duplicates (same label) or exact label conflicts (both rejected)
            group = build_module.stable_id(REPOSITORY_KEY, revision, real_path, fn.name, str(fn.start), unit.query.query_id)
            provenance = {
                "code_source_id": CODE_SOURCE_ID,
                "metadata_source_ids": [CODE_SOURCE_ID],
                "advisory_id": f"codeql:{unit.query.query_id}",
                "aliases": [],
                "ecosystem": "GIT",
                "packages": [],
                "published": committed_at,
                "osv_modified": committed_at,
                "osv_record_sha256": unit_sha,
                "repository": REPOSITORY,
                "repository_key": REPOSITORY_KEY,
                "vulnerable_revision": revision,
                "fixed_revision": revision,
                "fix_committed_at": committed_at,
                "file_path": real_path,
                "license": {"detected": detected.spdx, "category": detected.category, "files": [license_file], "status": "unreviewed"},
                "excerpt_revision": revision,
                "excerpt_function": fn.name,
                "excerpt_file_lines": [[fn.start, fn.end]],
                "content_sha256": content_sha256(excerpt),
                "normalized_sha256": code.normalized_sha256(language, excerpt),
                "source_kind": "tool_test",
                "test_unit": unit.directory,
                "query_id": unit.query.query_id,
            }
            if decision == "no_finding":
                provenance["negative_basis"] = item["reason"]
            records.append(
                {
                    "schema_version": "1.0",
                    "dataset_id": dataset_id,
                    "record_id": f"sg-{LANG_TAG[language]}-{kind.replace('_', '-')}-{build_module.stable_id(group, kind)}",
                    "group_id": f"sgg-{group}",
                    "pair_id": None,
                    "kind": kind,
                    "input": {"language": language, "file": shown, "context": context, "code": excerpt},
                    "target": {"decision": decision, **target},
                    "localization": localization,
                    "provenance": provenance,
                }
            )
            count(kind)
        log_rows.append({"advisory_id": f"codeql:{unit.query.query_id}", "repository_key": REPOSITORY_KEY, "test_unit": unit.directory, "outcome": "built" if outcome else "no_usable_function", "records": sum(v for k, v in outcome.items() if k.startswith("query_test")), "detail": dict(sorted(outcome.items()))})
    capped = cap_template_families(records, max_findings_per_file)
    if len(capped) != len(records):
        kept: dict[tuple[str, str], int] = {}
        for record in capped:
            key = (record["provenance"]["test_unit"], record["provenance"]["advisory_id"])
            kept[key] = kept.get(key, 0) + 1
        for row in log_rows:
            if row["records"]:
                now = kept.get((row["test_unit"], row["advisory_id"]), 0)
                row["detail"]["template_family_cap"] = row["records"] - now
                row["records"] = now
    capped.sort(key=lambda r: r["record_id"])
    return capped, log_rows


def cap_template_families(records: list[dict[str, Any]], max_findings_per_file: int | None) -> list[dict[str, Any]]:
    """A test file is one template family: its alert functions are variants of one rule pattern. Keep at most
    `max_findings_per_file` findings per file (deterministic choice by record id); every negative is kept."""
    if not max_findings_per_file:
        return records
    by_file: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["kind"] == "query_test_finding":
            by_file.setdefault(record["provenance"]["file_path"], []).append(record)
    kept = {r["record_id"] for group in by_file.values() for r in sorted(group, key=lambda r: r["record_id"])[:max_findings_per_file]}
    return [r for r in records if r["kind"] != "query_test_finding" or r["record_id"] in kept]
