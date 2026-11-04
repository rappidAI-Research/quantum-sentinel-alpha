import json
from pathlib import Path

from sentinel.garden.osv import OsvClient, fix_commit_refs, parse_candidate, sample_key

SHA = "3b171578ca44dfd75ca3c5cbddc7b44c600a7b49"
OPTIONS = dict(published_before="2026-11-01T00:00:00Z", require_reviewed=True, rejected_cwes={"CWE-1395", "CWE-693"}, max_cwes=2)


def advisory(**overrides) -> dict:
    record = {
        "id": "GHSA-3fxj-6jh8-hvhx",
        "published": "2025-06-25T18:21:37Z",
        "modified": "2025-07-01T00:00:00Z",
        "aliases": ["CVE-2025-1"],
        "summary": "chi has an IP spoofing vulnerability in RealIP",
        "affected": [{"package": {"name": "github.com/go-chi/chi/v5", "ecosystem": "Go"}}],
        "references": [
            {"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{SHA}"},
            {"type": "PACKAGE", "url": "https://github.com/go-chi/chi"},
        ],
        "database_specific": {"cwe_ids": ["CWE-290"], "github_reviewed": True, "severity": "MODERATE"},
    }
    record.update(overrides)
    return record


def test_candidate_resolves_single_fix_commit() -> None:
    candidate, reason = parse_candidate(advisory(), "Go", **OPTIONS)
    assert reason == "ok"
    assert candidate["repository"] == "https://github.com/go-chi/chi"
    assert candidate["repository_key"] == "go-chi/chi"
    assert candidate["fix_commit"] == SHA
    assert candidate["cwe_ids"] == ["CWE-290"]
    assert len(candidate["osv_record_sha256"]) == 64


def test_rejection_reasons() -> None:
    db = advisory()["database_specific"]
    cases = {
        "not_github_reviewed": advisory(database_specific={**db, "github_reviewed": False}),
        "withdrawn": advisory(withdrawn="2025-08-01T00:00:00Z"),
        "published_after_snapshot": advisory(published="2026-11-02T00:00:00Z"),
        "ecosystem_mismatch": advisory(affected=[{"package": {"name": "x", "ecosystem": "npm"}}]),
        "no_cwe": advisory(database_specific={**db, "cwe_ids": []}),
        "cwe_not_localizable": advisory(database_specific={**db, "cwe_ids": ["CWE-1395"]}),
        "too_many_cwes": advisory(database_specific={**db, "cwe_ids": ["CWE-79", "CWE-80", "CWE-116"]}),
        "no_fix_commit": advisory(references=[{"type": "WEB", "url": "https://example.com"}]),
        "short_commit_reference": advisory(references=[{"type": "WEB", "url": "https://github.com/go-chi/chi/commit/3b17157"}]),
        "multiple_fix_commits": advisory(
            references=[
                {"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{SHA}"},
                {"type": "WEB", "url": f"https://github.com/go-chi/chi/commit/{'1' * 40}"},
            ]
        ),
    }
    for expected, record in cases.items():
        candidate, reason = parse_candidate(record, "Go", **OPTIONS)
        assert candidate is None and reason == expected, expected


def test_rejected_cwe_is_dropped_but_specific_one_kept() -> None:
    db = advisory()["database_specific"]
    candidate, _ = parse_candidate(advisory(database_specific={**db, "cwe_ids": ["CWE-693", "CWE-79"]}), "Go", **OPTIONS)
    assert candidate["cwe_ids"] == ["CWE-79"]


def test_commit_refs_from_pull_request_urls_and_git_ranges() -> None:
    record = advisory(
        references=[{"type": "FIX", "url": f"https://github.com/o/r/pull/12/commits/{SHA}"}],
        affected=[
            {
                "package": {"name": "x", "ecosystem": "Go"},
                "ranges": [{"type": "GIT", "repo": "https://github.com/o/r", "events": [{"introduced": "0"}, {"fixed": SHA}]}],
            }
        ],
    )
    assert fix_commit_refs(record) == [("o", "r", SHA)]


def test_sample_order_is_deterministic_and_salted() -> None:
    ids = [f"GHSA-{i}" for i in range(20)]
    first = sorted(ids, key=lambda i: sample_key("s1", i))
    assert first == sorted(ids, key=lambda i: sample_key("s1", i))
    assert first != sorted(ids, key=lambda i: sample_key("s2", i))


def test_osv_client_caches_downloads(tmp_path: Path) -> None:
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        if url.endswith("modified_id.csv"):
            return b"2025-01-01T00:00:00Z,GHSA-aaaa-bbbb-cccc\n2025-01-01T00:00:00Z,PYSEC-2025-1\n"
        return json.dumps(advisory()).encode()

    client = OsvClient("https://bucket", tmp_path, fetch=fetch)
    assert client.advisory_ids("Go", "GHSA-") == ["GHSA-aaaa-bbbb-cccc"]
    assert client.records("Go", ["GHSA-aaaa-bbbb-cccc"], workers=2)[0]["id"] == "GHSA-3fxj-6jh8-hvhx"
    client.advisory_ids("Go", "GHSA-")
    client.record("Go", "GHSA-aaaa-bbbb-cccc")
    assert len(calls) == 2  # second access is served from the cache
