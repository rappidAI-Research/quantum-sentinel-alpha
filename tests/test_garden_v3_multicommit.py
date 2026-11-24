"""Garden v3: advisories that name several fix commits in one repository.

A record is built only when one fix can be established; a fix split over different commits, or a
companion commit that changes anything but tests/docs, is rejected (never guessed)."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from sentinel.garden.build import Rejected, build_records, resolve_fix_commit
from sentinel.garden.gitsource import CommitInfo, GitError, LicenseFile
from sentinel.garden.osv import parse_candidate
from tests.garden_fixtures import MIT_TEXT, PY_NEW, PY_OLD, SETTINGS, candidate, git_diff
from tests.test_garden_discovery import OPTIONS, SHA, advisory

V3 = dataclasses.replace(SETTINGS, member_additions=True, version_hunks_as_noise=True, tooling_files_excluded=True)
BASE, BASE2 = "0" * 40, "9" * 40
A, B, C = "a" * 39 + "1", "b" * 39 + "2", "c" * 39 + "3"


class MultiRepo:
    """Commits as {sha: (parent, committed_at, {path: text})} with real git diffs between parent and commit."""

    def __init__(self, trees: dict[str, dict[str, str]], commits: dict[str, tuple[str, str, dict[str, str]]]) -> None:
        self.trees = dict(trees)
        self.commits = commits
        for sha, (parent, _, files) in commits.items():
            self.trees[sha] = {**self.trees[parent], **files}

    def ensure_commit(self, sha: str) -> CommitInfo:
        if sha not in self.commits:
            raise GitError("unknown commit")
        parent, when, _ = self.commits[sha]
        return CommitInfo(sha=sha, parents=(parent,), committed_at=when, authored_at=when)

    def changed_files(self, parent: str, sha: str) -> list[tuple[str, str]]:
        old, new = self.trees[parent], self.trees[sha]
        return [("M" if p in old else "A", p) for p in sorted(new) if old.get(p) != new[p]]

    def file_diff(self, parent: str, sha: str, path: str) -> str:
        return git_diff(self.trees[parent].get(path, ""), self.trees[sha].get(path, ""), path)

    def text(self, revision: str, path: str) -> str:
        if path not in self.trees[revision]:
            raise GitError(f"{path} does not exist at {revision}")  # as the real repository does
        return self.trees[revision][path]

    def root_license_files(self, revision: str) -> list[LicenseFile]:
        return [LicenseFile("LICENSE", hashlib.sha256(MIT_TEXT.encode()).hexdigest(), MIT_TEXT)]


def multi(*shas: str) -> dict:
    cand = candidate()
    cand["fix_commit"], cand["fix_commits"] = None, sorted(shas)
    return cand


def _build(repo, cand):
    return build_records(cand, repo, V3, "sentinel-garden-test", "ghsa-osv-fix-commits", ["osv"])


BRANCH2_OLD = PY_OLD.replace("import sqlite3\n", "import sqlite3\nimport logging\n")  # release branch differs slightly


def test_code_commit_plus_docs_commit_uses_the_code_commit() -> None:
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD, "CHANGELOG.md": "x\n", "tests/test_orders.py": "def test_x():\n    pass\n"}},
                     {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": PY_NEW}),
                      B: (A, "2025-03-02T10:00:00+00:00", {"CHANGELOG.md": "x\nfixed SQL injection\n",
                                                           "tests/test_orders.py": "def test_x():\n    assert True\n"})})
    records = {r["kind"]: r for r in _build(repo, multi(A, B))}
    assert records["vulnerable"]["provenance"]["fixed_revision"] == A
    assert records["vulnerable"]["provenance"]["vulnerable_revision"] == BASE


def test_identical_backport_uses_the_earliest_commit() -> None:
    backport_new = PY_NEW.replace("import sqlite3\n", "import sqlite3\nimport logging\n")
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD}, BASE2: {"shop/orders.py": BRANCH2_OLD}},
                     {A: (BASE, "2025-03-05T10:00:00+00:00", {"shop/orders.py": PY_NEW}),
                      B: (BASE2, "2025-03-01T10:00:00+00:00", {"shop/orders.py": backport_new})})
    info = resolve_fix_commit(multi(A, B), repo, V3)
    assert info.sha == B  # same removed/added lines on both branches; the earlier one is the fix date


def test_fix_split_over_different_commits_is_rejected() -> None:
    partial = PY_OLD.replace("conn.execute(query)", "conn.execute(query, ())")
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD}},
                     {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": partial}),
                      B: (A, "2025-03-02T10:00:00+00:00", {"shop/orders.py": PY_NEW})})
    with pytest.raises(Rejected, match="multi_commit_distinct_changes"):
        _build(repo, multi(A, B))


@pytest.mark.parametrize("companion", ["requirements.txt", "requirements.txt.lock", "setup.cfg", "go.mod", "vendor/lib/sql.py", "native/orders.c", "package.json"])
def test_companion_commit_that_may_itself_be_the_fix_is_rejected(companion) -> None:
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD, companion: "old\n"}},
                     {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": PY_NEW}),
                      B: (A, "2025-03-02T10:00:00+00:00", {companion: "new\n"})})
    with pytest.raises(Rejected, match="multi_commit_unexplained_change"):
        _build(repo, multi(A, B))


def test_no_runtime_source_change_in_any_commit_is_rejected() -> None:
    repo = MultiRepo({BASE: {"README.md": "a\n", "docs/x.md": "b\n"}},
                     {A: (BASE, "2025-03-01T10:00:00+00:00", {"README.md": "c\n"}),
                      B: (A, "2025-03-02T10:00:00+00:00", {"docs/x.md": "d\n"})})
    with pytest.raises(Rejected, match="no_source_change_in_language"):
        _build(repo, multi(A, B))


def test_unresolvable_commit_is_rejected() -> None:
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD}}, {A: (BASE, "2025-03-01T10:00:00+00:00", {"shop/orders.py": PY_NEW})})
    with pytest.raises(Rejected, match="unresolvable_commit"):
        _build(repo, multi(A, C))


def test_discovery_keeps_multi_commit_advisories_only_when_enabled_and_in_one_repository() -> None:
    two = advisory(references=[{"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{SHA}"},
                               {"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{'1' * 40}"}])
    assert parse_candidate(two, "Go", **OPTIONS) == (None, "multiple_fix_commits")  # default: v1/v2 behaviour
    cand, reason = parse_candidate(two, "Go", **OPTIONS, allow_multiple_fix_commits=True)
    assert reason == "ok" and cand["fix_commit"] is None and cand["fix_commits"] == sorted([SHA, "1" * 40])
    other_repo = advisory(references=[{"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{SHA}"},
                                      {"type": "WEB", "url": f"https://github.com/other/fork/commit/{'1' * 40}"}])
    assert parse_candidate(other_repo, "Go", **OPTIONS, allow_multiple_fix_commits=True) == (None, "multiple_fix_commits")
    single, _ = parse_candidate(advisory(), "Go", **OPTIONS, allow_multiple_fix_commits=True)
    assert "fix_commits" not in single and single["fix_commit"] == SHA


def test_multi_commit_candidates_never_displace_single_commit_candidates(tmp_path, monkeypatch) -> None:
    """Discovery with multi-commit fixes enabled is a superset of discovery without them."""
    import yaml

    from sentinel.garden import pipeline

    def adv(i: int, shas: list[str]) -> dict:
        return advisory(id=f"GHSA-{i:04d}-aaaa-bbbb", references=[{"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{s}"} for s in shas])

    records = {a["id"]: a for a in [adv(1, [f"{1:040d}", f"{2:040d}"]), adv(2, [f"{3:040d}"]), adv(3, [f"{4:040d}"])]}

    class FakeOsv:
        def __init__(self, *args) -> None:
            pass

        def advisory_ids(self, ecosystem, prefix, refresh=False):
            return sorted(records) if ecosystem == "Go" else []

        def records(self, ecosystem, ids, workers):
            return [records[i] for i in ids]

    monkeypatch.setattr(pipeline, "OsvClient", FakeOsv)
    base = yaml.safe_load(pipeline.DEFAULT_CONFIG_PATH.read_text())
    results = {}
    for flag in (False, True):
        raw = {**base, "workdir": "artifacts/test-discover", "committed_dir": str(tmp_path / str(flag)),
               "discovery": {**base["discovery"], "ecosystems": ["Go"], "max_candidates_per_repository": 2, "multi_commit_fixes": flag}}
        (tmp_path / f"{flag}.yaml").write_text(yaml.safe_dump(raw))
        (tmp_path / str(flag)).mkdir()
        pipeline.discover(pipeline.load_config(tmp_path / f"{flag}.yaml"), log=lambda *a: None)
        results[flag] = [json.loads(line)["advisory_id"] for line in (tmp_path / str(flag) / "candidates.jsonl").read_text().splitlines()]
    assert results[False] == ["GHSA-0002-aaaa-bbbb", "GHSA-0003-aaaa-bbbb"]
    assert results[True] == results[False]  # the cap of 2 is taken by single-commit candidates first


TWO_CALLS = '''import subprocess


def run_both(user):
    subprocess.run(user, shell=True)
    log = "between the calls"
    subprocess.run(user, shell=True)
    return log
'''


def test_same_edit_at_two_places_in_two_commits_is_a_partial_fix_not_a_backport() -> None:
    # review finding: identical line multisets from sequential commits must not count as one fix; the
    # first commit's function is still vulnerable (the second call) and must never become a hard negative
    one = TWO_CALLS.replace("subprocess.run(user, shell=True)", "subprocess.run(user, shell=False)", 1)
    both = one.replace("subprocess.run(user, shell=True)", "subprocess.run(user, shell=False)", 1)
    repo = MultiRepo({BASE: {"tools/runner.py": TWO_CALLS}},
                     {A: (BASE, "2025-03-01T10:00:00+00:00", {"tools/runner.py": one}),
                      B: (A, "2025-03-02T10:00:00+00:00", {"tools/runner.py": both})})
    with pytest.raises(Rejected, match="multi_commit_distinct_changes"):
        _build(repo, multi(A, B))


def test_earliest_instant_is_chosen_across_time_zones() -> None:
    # review finding: 2026-06-30T23:45-01:00 is after 2026-07-01T00:30+01:00 although it sorts first as text
    backport_new = PY_NEW.replace("import sqlite3\n", "import sqlite3\nimport logging\n")
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD}, BASE2: {"shop/orders.py": BRANCH2_OLD}},
                     {A: (BASE, "2026-06-30T23:45:00-01:00", {"shop/orders.py": PY_NEW}),
                      B: (BASE2, "2026-07-01T00:30:00+01:00", {"shop/orders.py": backport_new})})
    assert resolve_fix_commit(multi(A, B), repo, V3).sha == B


def test_identical_backports_that_add_a_file_are_compared_not_rejected() -> None:
    # a file the commit adds has no parent version: its hunks keep absolute positions instead of failing
    helper = "def safe(value):\n    return str(value)\n"
    backport_new = PY_NEW.replace("import sqlite3\n", "import sqlite3\nimport logging\n")
    repo = MultiRepo({BASE: {"shop/orders.py": PY_OLD}, BASE2: {"shop/orders.py": BRANCH2_OLD}},
                     {A: (BASE, "2025-03-05T10:00:00+00:00", {"shop/orders.py": PY_NEW, "shop/safe.py": helper}),
                      B: (BASE2, "2025-03-01T10:00:00+00:00", {"shop/orders.py": backport_new, "shop/safe.py": helper})})
    assert resolve_fix_commit(multi(A, B), repo, V3).sha == B
