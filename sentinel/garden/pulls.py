"""Garden v5: resolve an advisory's pull-request reference to the commit that merged it.

About 720 GitHub-reviewed advisories reference a fix pull request but no fix commit. The commit search
(`gh api search/commits`, the only GitHub API this project uses) finds the default-branch commit
whose first message line is GitHub's merge-commit line ("Merge pull request #N from ...") or a
squash title ending in "(#N)" (`osv.merged_commit`). Its first-parent diff is the merged change, so
the candidate then goes through the unchanged build, quality checks and audit like any other.

Resolutions are frozen in `<committed_dir>/pull_request_resolutions.jsonl`: `discover` uses them
before any search, so a rebuild needs upstream git only. A failed search is never frozen (it is
retried); "not_found" and "ambiguous" are, and their advisories stay rejected.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from sentinel.garden.osv import merged_commit


def gh_commit_search(repository_key: str, number: int) -> list[dict[str, str]]:
    query = f'repo:{repository_key} "#{number}"'
    done = subprocess.run(
        ["gh", "api", "-X", "GET", "search/commits", "-f", f"q={query}", "-f", "per_page=100", "--jq", "[.items[] | {sha, message: .commit.message}]"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if done.returncode != 0:
        message = done.stderr.strip()[:300] or f"gh exited {done.returncode}"
        if "HTTP 422" in message:  # the repository cannot be searched (renamed, moved or deleted): not retried
            raise LookupError(message)
        raise RuntimeError(message)
    return json.loads(done.stdout or "[]")


class PullRequestResolver:
    def __init__(
        self,
        frozen_path: Path,
        search: Callable[[str, int], list[dict[str, str]]] = gh_commit_search,
        interval: float = 2.2,  # the search API allows 30 requests per minute
        log: Callable[[str], None] = print,
    ) -> None:
        self.frozen_path = frozen_path
        self.search = search
        self.interval = interval
        self.log = log
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        if frozen_path.is_file():
            for line in frozen_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[(row["repository_key"], row["number"])] = row
        self._last = 0.0

    def resolve(self, repository_key: str, number: int) -> dict[str, Any]:
        key = (repository_key, number)
        if key in self.rows:
            return self.rows[key]
        for attempt in range(5):
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                commits = self.search(repository_key, number)
                break
            except LookupError:
                row = {"repository_key": repository_key, "number": number, "status": "repository_not_searchable"}
                self._record(key, row)
                return row
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                self.log(f"pull request search {repository_key}#{number} failed ({exc}); retrying")
                time.sleep(30 * (attempt + 1))
        else:
            return {"repository_key": repository_key, "number": number, "status": "search_failed"}
        row = {"repository_key": repository_key, "number": number, **merged_commit(number, commits)}
        self._record(key, row)
        return row

    def _record(self, key: tuple[str, int], row: dict[str, Any]) -> None:
        self.rows[key] = row
        self.write()  # incremental: an interrupted discovery keeps every resolution it already made

    def write(self) -> None:
        rows = [self.rows[k] for k in sorted(self.rows)]
        self.frozen_path.parent.mkdir(parents=True, exist_ok=True)
        self.frozen_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
