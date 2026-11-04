"""Offline fixtures for Sentinel Garden tests: an in-memory upstream repository with real git diffs."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sentinel.garden.build import ExtractionSettings
from sentinel.garden.gitsource import CommitInfo, GitError, LicenseFile

PARENT = "a" * 40
FIX = "b" * 40

MIT_TEXT = """MIT License

Copyright (c) 2024 Example

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
"""

SETTINGS = ExtractionSettings(
    max_source_files_changed=4,
    max_changed_functions=1,
    max_changed_lines=40,
    max_function_lines=150,
    min_function_lines=3,
    max_excerpt_chars=12000,
    max_import_lines=12,
    max_context_chars=1000,
    abstention_min_callee_name_length=6,
)


def git_diff(old: str, new: str, path: str) -> str:
    """Produce `git diff -U0` output for two file versions using real git."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        a.write_text(old, encoding="utf-8")
        b.write_text(new, encoding="utf-8")
        done = subprocess.run(
            ["git", "diff", "--no-index", "-U0", "--no-color", str(a), str(b)], capture_output=True, text=True, check=False
        )
        return done.stdout


class FakeRepo:
    def __init__(self, old: dict[str, str], new: dict[str, str], license_text: str | None = MIT_TEXT, parents: tuple[str, ...] = (PARENT,)) -> None:
        self.files = {PARENT: old, FIX: new}
        self.license_text = license_text
        self.parents = parents

    def ensure_commit(self, sha: str) -> CommitInfo:
        if sha != FIX:
            raise GitError("unknown commit")
        return CommitInfo(sha=FIX, parents=self.parents, committed_at="2025-03-01T10:00:00+00:00", authored_at="2025-03-01T10:00:00+00:00")

    def changed_files(self, parent: str, sha: str) -> list[tuple[str, str]]:
        old, new = self.files[parent], self.files[sha]
        result = []
        for path in sorted(set(old) | set(new)):
            if path not in old:
                result.append(("A", path))
            elif path not in new:
                result.append(("D", path))
            elif old[path] != new[path]:
                result.append(("M", path))
        return result

    def file_diff(self, parent: str, sha: str, path: str) -> str:
        return git_diff(self.files[parent][path], self.files[sha][path], path)

    def text(self, revision: str, path: str) -> str:
        return self.files[revision][path]

    def root_license_files(self, revision: str) -> list[LicenseFile]:
        if self.license_text is None:
            return []
        return [LicenseFile("LICENSE", hashlib.sha256(self.license_text.encode()).hexdigest(), self.license_text)]


def candidate(ecosystem: str = "PyPI", cwes: list[str] | None = None, repo: str = "example/webapp", advisory: str = "GHSA-2222-3333-4444") -> dict[str, Any]:
    return {
        "advisory_id": advisory,
        "aliases": ["CVE-2025-0001"],
        "ecosystem": ecosystem,
        "packages": ["webapp"],
        "published": "2025-03-02T00:00:00Z",
        "osv_modified": "2025-03-05T00:00:00Z",
        "osv_record_sha256": "c" * 64,
        "cwe_ids": cwes or ["CWE-89"],
        "severity": "HIGH",
        "summary": "SQL injection in webapp order listing",
        "repository": f"https://github.com/{repo}",
        "repository_key": repo.lower(),
        "fix_commit": FIX,
    }


PY_OLD = '''import sqlite3

from flask import request


def list_orders(conn, status):
    query = "SELECT id FROM orders WHERE status = '%s'" % status
    rows = conn.execute(query)
    return [row[0] for row in rows]


def orders_view(conn):
    status = request.args.get("status", "open")
    return {"orders": list_orders(conn, status)}
'''

PY_NEW = '''import sqlite3

from flask import request


def list_orders(conn, status):
    query = "SELECT id FROM orders WHERE status = ?"
    rows = conn.execute(query, (status,))
    return [row[0] for row in rows]


def orders_view(conn):
    status = request.args.get("status", "open")
    return {"orders": list_orders(conn, status)}
'''

GO_OLD = '''package auth

import (
	"log/slog"
)

func validateToken(token string, ok bool) error {
	if !ok {
		slog.Debug("failed to authenticate",
			slog.String("token", token))
		return errDenied
	}
	return nil
}
'''

GO_NEW = '''package auth

import (
	"log/slog"
)

func validateToken(token string, ok bool) error {
	if !ok {
		slog.Debug("failed to authenticate",
			slog.String("token", redactToken(token)))
		return errDenied
	}
	return nil
}

// redactToken hides all but the last four characters.
func redactToken(token string) string {
	if len(token) <= 4 {
		return "[REDACTED]"
	}
	return "[REDACTED]" + token[len(token)-4:]
}
'''
