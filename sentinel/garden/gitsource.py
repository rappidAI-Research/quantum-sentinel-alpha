"""Stage 2 - resolution: fetch a fix commit and its parent from the upstream repository.

Each upstream repository gets one bare, partial (`--filter=tree:0`), shallow cache under the
Garden work directory. Only the fix commit and its parent are fetched; trees and blobs are
downloaded on demand, so a diff only pulls the directories along the changed paths (a monorepo
fix costs megabytes instead of a full tree per commit). Everything is addressed by full commit
SHA, so a rebuild reads identical bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sentinel.garden.licenses import LICENSE_FILE

SHA = re.compile(r"^[0-9a-f]{40}$")
FILTER = "tree:0"  # lazily fetch trees and blobs; see module docstring
FETCHED_MARKER = "qsa-fetched-commits"  # commits fetched explicitly with --depth=2
_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true", "LC_ALL": "C"}


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    parents: tuple[str, ...]
    committed_at: str  # ISO-8601 committer date
    authored_at: str


@dataclass(frozen=True)
class LicenseFile:
    path: str
    sha256: str
    text: str


class UpstreamRepo:
    def __init__(self, cache_root: Path, repository_url: str, repository_key: str) -> None:
        self.url = repository_url
        self.path = cache_root / repository_key.replace("/", "__")
        if not (self.path / "HEAD").is_file():
            self.path.mkdir(parents=True, exist_ok=True)
            self._git("init", "--bare", "-q")
            self._git("remote", "add", "origin", repository_url)
            self._git("config", "remote.origin.promisor", "true")
            self._git("config", "remote.origin.partialclonefilter", FILTER)
            self._git("config", "extensions.partialClone", "origin")

    def _git(self, *args: str, timeout: int = 120, binary: bool = False) -> str | bytes:
        try:
            done = subprocess.run(
                ["git", "-C", str(self.path), *args],
                capture_output=True,
                timeout=timeout,
                env=_ENV,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {args[0]} timed out") from exc
        if done.returncode != 0:
            message = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise GitError(f"git {args[0]} failed: {message[-1] if message else done.returncode}")
        return done.stdout if binary else done.stdout.decode("utf-8", "replace")

    def has_commit(self, sha: str) -> bool:
        # Never probe with `cat-file`/`rev-parse`: in a partial clone a missing object triggers a lazy
        # fetch of that commit's whole closure (all ancestors and their trees), ignoring --depth.
        # That silently pulled entire histories of monorepos (10 M objects, 2.5 GB for grafana).
        marker = self.path / FETCHED_MARKER
        return marker.is_file() and sha in marker.read_text(encoding="utf-8").split()

    def _mark_fetched(self, sha: str) -> None:
        with (self.path / FETCHED_MARKER).open("a", encoding="utf-8") as handle:
            handle.write(sha + "\n")

    def ensure_commit(self, sha: str) -> CommitInfo:
        if not SHA.match(sha):
            raise GitError(f"not a full commit SHA: {sha}")
        # A commit that is the shallow boundary of another fetch (e.g. a fix commit that is also the
        # parent of a later fix) looks parentless and its parent may be missing: deepen it again.
        if not self.has_commit(sha) or sha in self._shallow():
            self._git("fetch", "-q", "--depth=2", f"--filter={FILTER}", "origin", sha, timeout=300)
            self._mark_fetched(sha)
        # Parents come from the raw commit object, which shallow grafts do not alter.
        raw = str(self._git("cat-file", "commit", sha))
        header = raw.split("\n\n", 1)[0].splitlines()
        parents = tuple(line.split()[1] for line in header if line.startswith("parent "))
        line = str(self._git("show", "-s", "--format=%H%n%cI%n%aI", sha)).strip().splitlines()
        if line[0] != sha:
            raise GitError(f"resolved {line[0]} instead of {sha}")
        if sha in self._shallow() and parents:
            raise GitError(f"{sha} is still a shallow boundary after deepening")
        return CommitInfo(sha=sha, parents=parents, committed_at=line[1], authored_at=line[2])

    def _shallow(self) -> set[str]:
        path = self.path / "shallow"
        return set(path.read_text(encoding="utf-8").split()) if path.is_file() else set()

    def changed_files(self, parent: str, sha: str) -> list[tuple[str, str]]:
        out = str(self._git("diff", "--name-status", "--no-renames", "--no-ext-diff", parent, sha))
        files = []
        for row in out.splitlines():
            status, _, path = row.partition("\t")
            files.append((status[:1], path))
        return files

    def file_diff(self, parent: str, sha: str, path: str) -> str:
        return str(
            self._git("diff", "-U0", "--no-color", "--no-renames", "--no-ext-diff", "--text", parent, sha, "--", path, timeout=300)
        )

    def blob_bytes(self, revision: str, path: str) -> bytes:
        return bytes(self._git("show", f"{revision}:{path}", binary=True, timeout=300))

    def text(self, revision: str, path: str) -> str:
        data = self.blob_bytes(revision, path)
        if b"\x00" in data:
            raise GitError(f"binary file: {path}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitError(f"non-UTF-8 file: {path}") from exc

    def root_license_files(self, revision: str) -> list[LicenseFile]:
        names = str(self._git("ls-tree", "--name-only", revision)).splitlines()
        files = []
        for name in sorted(n for n in names if LICENSE_FILE.match(n)):
            try:
                data = self.blob_bytes(revision, name)
            except GitError:
                continue  # a directory named LICENSES etc.
            files.append(
                LicenseFile(path=name, sha256=hashlib.sha256(data).hexdigest(), text=data.decode("utf-8", "replace"))
            )
        return files
