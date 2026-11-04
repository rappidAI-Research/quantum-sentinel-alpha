"""Parse `git diff -U0` output for one file into hunks with old/new line numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    removed: list[tuple[int, str]] = field(default_factory=list)  # (old line number, text)
    added: list[tuple[int, str]] = field(default_factory=list)  # (new line number, text)

    @property
    def is_insertion(self) -> bool:
        return self.old_count == 0

    @property
    def changed_line_count(self) -> int:
        return len(self.removed) + len(self.added)


def parse_unified_diff(text: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    current: Hunk | None = None
    old_line = new_line = 0
    for line in text.splitlines():
        match = _HUNK.match(line)
        if match:
            old_start, old_count, new_start, new_count = match.groups()
            current = Hunk(
                old_start=int(old_start),
                old_count=1 if old_count is None else int(old_count),
                new_start=int(new_start),
                new_count=1 if new_count is None else int(new_count),
            )
            hunks.append(current)
            old_line, new_line = current.old_start, current.new_start
            continue
        if current is None:  # file header ("diff --git", "---", "+++", "index") precedes the first hunk
            continue
        if line.startswith("-"):
            current.removed.append((old_line, line[1:]))
            old_line += 1
        elif line.startswith("+"):
            current.added.append((new_line, line[1:]))
            new_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
        # "\ No newline at end of file" and anything else carries no line
    for hunk in hunks:
        if len(hunk.removed) != hunk.old_count or len(hunk.added) != hunk.new_count:
            raise ValueError(
                f"hunk @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@ "
                f"does not match its body ({len(hunk.removed)} removed, {len(hunk.added)} added)"
            )
    return hunks
