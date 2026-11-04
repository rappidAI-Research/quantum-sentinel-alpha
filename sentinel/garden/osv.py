"""Stage 1 - discovery: GitHub-reviewed advisories from the OSV export -> fix-commit candidates.

OSV and the GitHub Advisory Database are used as *metadata* only (advisory ID, CWE, summary,
publication date and the reference to the upstream fix commit). Source code is never taken from
them: it is resolved from the upstream repository at immutable revisions in stage 2.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

_COMMIT_URL = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/(?:pull/\d+/)?commits?/([0-9a-fA-F]{7,40})(?:[/?#].*)?$"
)
_REPO_URL = re.compile(r"^https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
CWE_ID = re.compile(r"^CWE-[1-9][0-9]*$")
USER_AGENT = "quantum-sentinel-alpha-sentinel-garden/1.0"


def sample_key(salt: str, advisory_id: str) -> str:
    return hashlib.sha256(f"{salt}\x00{advisory_id}".encode("utf-8")).hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class OsvClient:
    """Reads the public OSV GCS export, caching every downloaded file under `cache_dir`."""

    def __init__(self, bucket: str, cache_dir: Path, fetch: Callable[[str], bytes] = _http_get) -> None:
        self.bucket = bucket.rstrip("/")
        self.cache_dir = cache_dir
        self.fetch = fetch

    def _cached(self, relative: str, refresh: bool = False) -> bytes:
        path = self.cache_dir / relative
        if path.is_file() and not refresh:
            return path.read_bytes()
        data = self.fetch(f"{self.bucket}/{relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def advisory_ids(self, ecosystem: str, prefix: str, refresh: bool = False) -> list[str]:
        text = self._cached(f"{ecosystem}/modified_id.csv", refresh=refresh).decode("utf-8")
        ids = {row[1] for row in csv.reader(io.StringIO(text)) if len(row) == 2 and row[1].startswith(prefix)}
        return sorted(ids)

    def record(self, ecosystem: str, advisory_id: str) -> dict[str, Any] | None:
        try:
            data = self._cached(f"{ecosystem}/{advisory_id}.json")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        document = json.loads(data)
        return document if isinstance(document, dict) else None

    def records(self, ecosystem: str, advisory_ids: list[str], workers: int) -> list[dict[str, Any] | None]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda advisory_id: self.record(ecosystem, advisory_id), advisory_ids))


def fix_commit_refs(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Distinct (owner, repo, sha) commit references, plus GIT-range `fixed` events."""
    refs: set[tuple[str, str, str]] = set()
    for reference in record.get("references") or []:
        match = _COMMIT_URL.match(str(reference.get("url", "")))
        if match:
            owner, repo, sha = match.groups()
            refs.add((owner, repo, sha.lower()))
    for affected in record.get("affected") or []:
        for range_ in affected.get("ranges") or []:
            if range_.get("type") != "GIT":
                continue
            repo_match = _REPO_URL.match(str(range_.get("repo", "")))
            if not repo_match:
                continue
            for event in range_.get("events") or []:
                fixed = str(event.get("fixed", ""))
                if re.fullmatch(r"[0-9a-fA-F]{40}", fixed):
                    refs.add((repo_match.group(1), repo_match.group(2), fixed.lower()))
    return sorted(refs)


def package_repositories(record: dict[str, Any]) -> set[str]:
    repos = set()
    for reference in record.get("references") or []:
        if reference.get("type") in {"PACKAGE", "WEB"}:
            match = _REPO_URL.match(str(reference.get("url", "")))
            if match:
                repos.add(f"{match.group(1)}/{match.group(2)}".lower())
    return repos


def parse_candidate(
    record: dict[str, Any],
    ecosystem: str,
    *,
    published_before: str,
    require_reviewed: bool,
    rejected_cwes: set[str],
    max_cwes: int,
) -> tuple[dict[str, Any] | None, str]:
    """Return (candidate, "ok") or (None, rejection_reason). Pure: no I/O."""
    if record.get("withdrawn"):
        return None, "withdrawn"
    database = record.get("database_specific") or {}
    if require_reviewed and not database.get("github_reviewed"):
        return None, "not_github_reviewed"
    published = str(record.get("published", ""))
    if not published or published >= published_before:
        return None, "published_after_snapshot"
    ecosystems = {((a.get("package") or {}).get("ecosystem")) for a in record.get("affected") or []}
    if ecosystem not in ecosystems:
        return None, "ecosystem_mismatch"

    cwes = [c for c in database.get("cwe_ids") or [] if isinstance(c, str)]
    if not cwes:
        return None, "no_cwe"
    if any(not CWE_ID.match(c) for c in cwes):
        return None, "invalid_cwe"
    specific = [c for c in cwes if c not in rejected_cwes]
    if not specific:
        return None, "cwe_not_localizable"
    if len(specific) > max_cwes:
        return None, "too_many_cwes"

    refs = fix_commit_refs(record)
    if not refs:
        return None, "no_fix_commit"
    full = [r for r in refs if len(r[2]) == 40]
    if not full:
        return None, "short_commit_reference"
    shas = {sha for _, _, sha in full}
    if len(shas) > 1:
        return None, "multiple_fix_commits"
    sha = shas.pop()
    package_repos = package_repositories(record)
    owner, repo, _ = sorted(full, key=lambda r: (f"{r[0]}/{r[1]}".lower() not in package_repos, r[0].lower(), r[1].lower()))[0]

    summary = str(record.get("summary") or "").strip()
    if not summary:
        return None, "no_summary"
    packages = sorted(
        {(a.get("package") or {}).get("name", "") for a in record.get("affected") or [] if (a.get("package") or {}).get("ecosystem") == ecosystem}
    )
    candidate = {
        "advisory_id": record["id"],
        "aliases": sorted(str(a) for a in record.get("aliases") or []),
        "ecosystem": ecosystem,
        "packages": [p for p in packages if p],
        "published": published,
        "osv_modified": str(record.get("modified", "")),
        "osv_record_sha256": record_sha256(record),
        "cwe_ids": specific,
        "severity": database.get("severity"),
        "summary": summary[:300],
        "repository": f"https://github.com/{owner}/{repo}",
        "repository_key": f"{owner}/{repo}".lower(),
        "fix_commit": sha,
    }
    return candidate, "ok"
