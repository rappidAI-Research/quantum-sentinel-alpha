"""Stages 2-3 - resolve a candidate's fix commit and turn it into function-level records.

One accepted fix commit yields:
- `vulnerable`: the changed function at the parent (vulnerable) revision -> `finding`, with the CWE
  from the advisory and a localization derived from the upstream fix diff;
- `post_fix`: the same function at the fix revision -> `no_finding` (a secure hard negative scoped to
  the advisory's weakness), carrying the upstream patch restricted to that function;
- `caller_abstention` (when one exists): an unchanged function in the vulnerable file that calls the
  vulnerable function, shown without the callee's body -> `abstain`.

A fix is used only when its logic change is confined to exactly one function, so the function-level
label is unambiguous. Every rejection is reported with a reason; nothing is guessed or repaired.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from sentinel.garden import code
from sentinel.garden.diff import Hunk, parse_unified_diff
from sentinel.garden.gitsource import CommitInfo, GitError, LicenseFile
from sentinel.garden.licenses import detect


class Repo(Protocol):
    def ensure_commit(self, sha: str) -> CommitInfo: ...
    def changed_files(self, parent: str, sha: str) -> list[tuple[str, str]]: ...
    def file_diff(self, parent: str, sha: str, path: str) -> str: ...
    def text(self, revision: str, path: str) -> str: ...
    def root_license_files(self, revision: str) -> list[LicenseFile]: ...


class Rejected(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ExtractionSettings:
    max_source_files_changed: int
    max_changed_functions: int
    max_changed_lines: int
    max_function_lines: int
    min_function_lines: int
    max_excerpt_chars: int
    max_import_lines: int
    max_context_chars: int
    abstention_min_callee_name_length: int


@dataclass
class _Change:
    path: str
    language: str
    old_source: str
    new_source: str
    old_function: code.Function
    new_function: code.Function
    hunks: list[Hunk]
    additions: list[tuple[int, int]]  # new-file line ranges of top-level code added by the fix
    trivial_hunks: list[Hunk] = field(default_factory=list)  # comment/blank-only hunks inside the function
    old_noise_lines: set[int] = field(default_factory=set)  # blank/comment/docstring/import lines (old file)
    new_noise_lines: set[int] = field(default_factory=set)
    old_trivial_lines: set[int] = field(default_factory=set)  # noise + bracket-only: not localizable
    new_trivial_lines: set[int] = field(default_factory=set)


def stable_id(*parts: str, length: int = 16) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]


def _containing(
    functions: list[code.Function], first: int, last: int, insertion: bool, language: str, lines: list[str], inserted: list[str]
) -> code.Function | None:
    if not insertion:
        return code.innermost(functions, first, last)
    anchor = first  # code was inserted after this line (0 = start of file)
    matches = []
    for f in functions:
        if f.start <= anchor < f.end:
            matches.append(f)
        elif language == "python" and anchor == f.end and f.start <= anchor and inserted:
            def_line = lines[f.start - 1] if f.start - 1 < len(lines) else ""
            def_indent = len(def_line) - len(def_line.lstrip())
            body = [line for line in inserted if line.strip()]
            if body and min(len(line) - len(line.lstrip()) for line in body) > def_indent:
                matches.append(f)
    if not matches:
        return None
    return min(matches, key=lambda f: (f.length, -f.start))


def _is_trivial(hunk: Hunk, old_noise: set[int], new_noise: set[int]) -> bool:
    """Only blank, comment, docstring or import lines change (classified on whole files).

    Bracket-only lines are not trivial: moving a closing brace changes control flow."""
    return all(n in old_noise for n, _ in hunk.removed) and all(n in new_noise for n, _ in hunk.added)


def _is_top_level_addition(hunk: Hunk, language: str, new_noise: set[int]) -> bool:
    """A pure insertion of complete top-level code (e.g. a new helper function or constant)."""
    if not hunk.is_insertion:
        return False
    lines = [t for _, t in hunk.added]
    meaningful = [t for n, t in hunk.added if n not in new_noise]
    if not meaningful or meaningful[0][:1].isspace():
        return False
    return code.excerpt_is_well_formed(language, "\n".join(lines) + "\n")


def _locate_changes(repo: Repo, info: CommitInfo, paths: list[str]) -> list[_Change]:
    parent = info.parents[0]
    changes: dict[tuple[str, int, int], _Change] = {}
    additions: dict[str, list[tuple[int, int]]] = {}
    trivial: list[tuple[str, Hunk]] = []
    for path in paths:
        language = code.language_of(path)
        assert language is not None
        try:
            old_source = repo.text(parent, path)
            new_source = repo.text(info.sha, path)
        except GitError as exc:
            raise Rejected("unreadable_source", str(exc)) from exc
        hunks = parse_unified_diff(repo.file_diff(parent, info.sha, path))
        old_lines, new_lines = old_source.splitlines(), new_source.splitlines()
        old_functions = code.find_functions(language, old_source)
        new_functions = code.find_functions(language, new_source)
        old_noise, new_noise = code.noise_lines(language, old_source), code.noise_lines(language, new_source)
        old_trivial, new_trivial = code.trivial_lines(language, old_source), code.trivial_lines(language, new_source)
        for hunk in hunks:
            if _is_trivial(hunk, old_noise, new_noise):
                trivial.append((path, hunk))
                continue
            old_fn = _containing(
                old_functions,
                hunk.old_start,
                hunk.old_start + max(hunk.old_count, 1) - 1,
                hunk.is_insertion,
                language,
                old_lines,
                [t for _, t in hunk.added],
            )
            if old_fn is None and _is_top_level_addition(hunk, language, new_noise):
                additions.setdefault(path, []).append((hunk.new_start, hunk.new_start + hunk.new_count - 1))
                continue
            new_fn = _containing(
                new_functions,
                hunk.new_start,
                hunk.new_start + max(hunk.new_count, 1) - 1,
                hunk.new_count == 0,
                language,
                new_lines,
                [],
            )
            if old_fn is None or new_fn is None:
                raise Rejected("change_outside_function", path)
            if old_fn.name != new_fn.name:
                raise Rejected("function_mismatch", f"{old_fn.name} != {new_fn.name}")
            key = (path, old_fn.start, old_fn.end)
            change = changes.get(key)
            if change is None:
                changes[key] = _Change(path, language, old_source, new_source, old_fn, new_fn, [hunk], [], [], old_noise, new_noise, old_trivial, new_trivial)
            else:
                if change.new_function != new_fn:
                    raise Rejected("function_mismatch", f"{old_fn.name} maps to two functions")
                change.hunks.append(hunk)
    result = list(changes.values())
    for path, hunk in trivial:
        for change in result:
            last = hunk.old_start + max(hunk.old_count, 1) - 1
            if change.path == path and change.old_function.start <= hunk.old_start and last <= change.old_function.end:
                change.trivial_hunks.append(hunk)
    for path, ranges in additions.items():
        owners = [c for c in result if c.path == path]
        if len(owners) != 1:
            raise Rejected("top_level_addition_without_changed_function", path)
        owners[0].additions.extend(ranges)
    return result


def _excerpt(source: str, fn: code.Function) -> str:
    return "\n".join(source.splitlines()[fn.start - 1 : fn.end]) + "\n"


def _fixed_excerpt(change: _Change) -> tuple[str, list[list[int]]]:
    """The fixed function plus any top-level code the fix added, in file order."""
    lines = change.new_source.splitlines()
    blocks = [(change.new_function.start, change.new_function.end)]
    for start, end in change.additions:
        while start <= end and not lines[start - 1].strip():
            start += 1
        while end >= start and not lines[end - 1].strip():
            end -= 1
        if start <= end:
            blocks.append((start, end))
    blocks.sort()
    for (_, previous_end), (start, _) in zip(blocks, blocks[1:]):
        if start <= previous_end:
            raise Rejected("overlapping_fix_blocks")
    text = "\n\n".join("\n".join(lines[a - 1 : b]) for a, b in blocks) + "\n"
    return text, [[a, b] for a, b in blocks]


def _span(change: _Change) -> tuple[int, int, str]:
    """Vulnerable span in file coordinates and its kind."""
    lines: list[int] = []
    kinds = set()
    fn = change.old_function
    for hunk in change.hunks:
        removed = [n for n, _ in hunk.removed if n not in change.old_trivial_lines]
        added = [n for n, _ in hunk.added if n not in change.new_trivial_lines]
        structural = any(n not in change.old_noise_lines for n, _ in hunk.removed) or any(
            n not in change.new_noise_lines for n, _ in hunk.added
        )
        if removed:
            lines.extend(removed)
            kinds.add("changed_lines")
        elif added or structural:
            # logic was only added, or only block delimiters moved: localize the change point and
            # the line after it (e.g. the statement a moved brace now guards)
            anchor = max(hunk.old_start, fn.start)
            lines.extend([anchor, min(anchor + 1, fn.end)])
            kinds.add("insertion_point")
    if not lines:
        raise Rejected("no_localizable_lines")
    kind = kinds.pop() if len(kinds) == 1 else "mixed"
    return min(lines), max(lines), kind


def _context(
    language: str, path: str, function: str, source: str, settings: ExtractionSettings, extra: str = ""
) -> str:
    head = f"Function `{function}` from {path}."
    if extra:
        head += " " + extra
    imports = code.import_lines(language, source, limit=settings.max_import_lines)
    if not imports:
        return head + " No imports in this file."
    text = head + " File imports: "
    kept: list[str] = []
    for line in imports:
        candidate = text + "; ".join(kept + [line]) + "."
        if len(candidate) > settings.max_context_chars:
            break
        kept.append(line)
    return text + "; ".join(kept) + "." if kept else head


def _patch(path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3
    )
    return "\n".join(diff) + "\n"


def _diff_consistent(change: _Change, patch: str) -> bool:
    """The function-level patch must contain exactly the upstream diff's non-blank changed lines."""
    new_lines = change.new_source.splitlines()
    hunks = change.hunks + change.trivial_hunks
    removed = sorted(t for h in hunks for _, t in h.removed if t.strip())
    added = [t for h in hunks for _, t in h.added if t.strip()]
    added += [new_lines[n - 1] for a, b in change.additions for n in range(a, b + 1) if new_lines[n - 1].strip()]
    body = [line for line in patch.splitlines()[2:] if not line.startswith("@@")]
    patch_removed = sorted(line[1:] for line in body if line.startswith("-") and line[1:].strip())
    patch_added = sorted(line[1:] for line in body if line.startswith("+") and line[1:].strip())
    return removed == patch_removed and sorted(added) == patch_added


def _find_caller(change: _Change, settings: ExtractionSettings) -> tuple[code.Function, int] | None:
    """First unchanged function in the vulnerable file that really calls the vulnerable function."""
    callee = change.old_function
    if callee.name.startswith("<") or len(callee.name) < settings.abstention_min_callee_name_length:
        return None
    functions = code.find_functions(change.language, change.old_source)
    for number in code.call_lines(change.language, change.old_source, callee.name):
        if callee.start <= number <= callee.end:
            continue
        caller = code.innermost(functions, number, number)
        if caller is None or caller.name.startswith("<") or caller.name == callee.name:
            continue
        if caller.start <= callee.start <= caller.end or callee.start <= caller.start <= callee.end:
            continue  # nested in / around the vulnerable function
        if not settings.min_function_lines <= caller.length <= settings.max_function_lines:
            continue
        return caller, number
    return None


def build_records(candidate: dict[str, Any], repo: Repo, settings: ExtractionSettings, dataset_id: str, code_source_id: str, metadata_source_ids: list[str]) -> list[dict[str, Any]]:
    """Resolve one candidate into records. Raises `Rejected` with a reason when it is not usable."""
    try:
        info = repo.ensure_commit(candidate["fix_commit"])
    except GitError as exc:
        raise Rejected("unresolvable_commit", str(exc)) from exc
    if not info.parents:
        raise Rejected("root_commit")
    parent = info.parents[0]  # for a merge commit, the first-parent diff is the merged change

    allowed_languages = code.ECOSYSTEM_LANGUAGES[candidate["ecosystem"]]
    try:
        changed = repo.changed_files(parent, info.sha)
    except GitError as exc:
        raise Rejected("unresolvable_commit", str(exc)) from exc
    sources = [
        (status, path)
        for status, path in changed
        if code.language_of(path) in allowed_languages and code.path_role(path) == "source"
    ]
    if not sources:
        raise Rejected("no_source_change_in_language")
    if any(status != "M" for status, _ in sources):
        raise Rejected("source_file_added_or_deleted")
    if len(sources) > settings.max_source_files_changed:
        raise Rejected("too_many_source_files")

    changes = _locate_changes(repo, info, [path for _, path in sources])
    if not changes:
        raise Rejected("no_logic_change")
    if len(changes) > settings.max_changed_functions:
        raise Rejected("multiple_functions_changed", str(len(changes)))
    change = changes[0]
    fn, new_fn = change.old_function, change.new_function
    added_lines = sum(b - a + 1 for a, b in change.additions)
    changed_lines = sum(h.changed_line_count for h in change.hunks + change.trivial_hunks) + added_lines
    if changed_lines > settings.max_changed_lines:
        raise Rejected("fix_too_large", str(changed_lines))
    for f in (fn, new_fn):
        if not settings.min_function_lines <= f.length <= settings.max_function_lines:
            raise Rejected("function_size_out_of_bounds", str(f.length))

    vulnerable_code = _excerpt(change.old_source, fn)
    fixed_code, fixed_ranges = _fixed_excerpt(change)
    if max(len(vulnerable_code), len(fixed_code)) > settings.max_excerpt_chars:
        raise Rejected("excerpt_too_large")
    if code.normalized_sha256(change.language, vulnerable_code) == code.normalized_sha256(change.language, fixed_code):
        raise Rejected("cosmetic_fix")
    span_start, span_end, span_kind = _span(change)
    patch = _patch(change.path, vulnerable_code, fixed_code)

    try:
        license_files = repo.root_license_files(info.sha)
    except GitError:
        license_files = []
    detected = detect({f.path: f.text for f in license_files})

    language = code.language_of(change.path)
    assert language is not None
    cwe, alternates = candidate["cwe_ids"][0], candidate["cwe_ids"][1:]
    group = stable_id(candidate["repository_key"], info.sha, change.path, fn.name)
    lang_tag = {"python": "py", "javascript": "js", "typescript": "ts", "go": "go"}[language]
    base_provenance = {
        "code_source_id": code_source_id,
        "metadata_source_ids": list(metadata_source_ids),
        "advisory_id": candidate["advisory_id"],
        "aliases": candidate["aliases"],
        "ecosystem": candidate["ecosystem"],
        "packages": candidate["packages"],
        "published": candidate["published"],
        "osv_modified": candidate["osv_modified"],
        "osv_record_sha256": candidate["osv_record_sha256"],
        "repository": candidate["repository"],
        "repository_key": candidate["repository_key"],
        "vulnerable_revision": parent,
        "fixed_revision": info.sha,
        "fix_committed_at": info.committed_at,
        "file_path": change.path,
        "license": {
            "detected": detected.spdx,
            "category": detected.category,
            "files": [{"path": f.path, "sha256": f.sha256} for f in license_files],
            "status": "unreviewed",
        },
    }

    def record(
        kind: str,
        decision: str,
        excerpt: str,
        revision: str,
        function: code.Function,
        context: str,
        target: dict[str, Any],
        localization: dict[str, Any],
        ranges: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "record_id": f"sg-{lang_tag}-{kind.replace('_', '-')}-{stable_id(group, kind)}",
            "group_id": f"sgg-{group}",
            "pair_id": f"sgp-{group}" if kind in {"vulnerable", "post_fix"} else None,
            "kind": kind,
            "input": {"language": language, "file": change.path, "context": context, "code": excerpt},
            "target": {"decision": decision, **target},
            "localization": localization,
            "provenance": {
                **base_provenance,
                "excerpt_revision": revision,
                "excerpt_function": function.name,
                "excerpt_file_lines": ranges or [[function.start, function.end]],
                "content_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                "normalized_sha256": code.normalized_sha256(language, excerpt),
            },
        }

    context = _context(language, change.path, fn.name, change.old_source, settings)
    fixed_context = _context(language, change.path, new_fn.name, change.new_source, settings)
    records = [
        record(
            "vulnerable",
            "finding",
            vulnerable_code,
            parent,
            fn,
            context,
            {
                "cwe": cwe,
                "alternate_cwes": alternates,
                "start_line": span_start - fn.start + 1,
                "end_line": span_end - fn.start + 1,
                "evidence": candidate["summary"],
                "evidence_source": "advisory_summary",
                "patch": patch,
            },
            {
                "method": "upstream_fix_diff",
                "span_kind": span_kind,
                "file_span": [span_start, span_end],
                "changed_lines": changed_lines,
                "fix_added_top_level_lines": added_lines,
                "diff_consistent": _diff_consistent(change, patch),
            },
        ),
        record(
            "post_fix",
            "no_finding",
            fixed_code,
            info.sha,
            new_fn,
            fixed_context,
            {
                "cwe": None,
                "alternate_cwes": [],
                "start_line": None,
                "end_line": None,
                "evidence": f"Patched version of this function: the upstream fix for the {cwe} weakness is applied.",
                "evidence_source": "derived_template",
                "patch": None,
                "label_scope": f"no {cwe}" + (f"/{'/'.join(alternates)}" if alternates else "") + " weakness after the upstream fix; other weaknesses are not verified",
            },
            {"method": "not_applicable"},
            fixed_ranges,
        ),
    ]

    caller = _find_caller(change, settings)
    if caller is not None:
        caller_fn, call_line = caller
        caller_code = _excerpt(change.old_source, caller_fn)
        note = f"`{fn.name}` is defined elsewhere in {change.path}; its body is not included."
        records.append(
            record(
                "caller_abstention",
                "abstain",
                caller_code,
                parent,
                caller_fn,
                _context(language, change.path, caller_fn.name, change.old_source, settings, extra=note),
                {
                    "cwe": None,
                    "alternate_cwes": [],
                    "start_line": None,
                    "end_line": None,
                    "evidence": f"`{fn.name}` is called here, but its implementation is not included, so it cannot be determined whether this call is safe.",
                    "evidence_source": "derived_template",
                    "patch": None,
                    "label_scope": f"the vulnerable callee `{fn.name}` ({cwe}) is withheld from the excerpt",
                },
                {"method": "not_applicable", "withheld_callee": fn.name, "call_line": call_line - caller_fn.start + 1},
            )
        )
    return records
