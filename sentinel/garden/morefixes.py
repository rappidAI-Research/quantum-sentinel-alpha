"""Garden v6: MoreFixes as a discovery index (metadata only, never its code).

MoreFixes (Akhondali et al., PROMISE '24; Zenodo record 20776007, CC-BY-4.0) is a PostgreSQL dump of
CVE fix commits mined from NVD and the GitHub Advisory Database. Garden v6 reads only its metadata
tables (`fixes`, `cve`, `cwe_classification`, `commits`, `file_change` without code, `repository`),
streamed from the pinned `.sql.gz` without restoring the database. The code comes from the pinned
upstream commit, like every other Garden record (`build.build_records`).

Evidence grading of a (CVE, fix commit) pair:
- `nvd_reference`: one of the CVE's NVD references names the commit (by its full hash or as a
  pull-request / commit URL of the repository): the CVE record itself names the fix (checked here).
- `ghsa_reference`: no NVD reference names it, but MoreFixes records that a GitHub Advisory Database
  record references the commit directly (rel_type GHSD_DIRECT_COMMIT). Upstream evidence relayed by
  MoreFixes, not re-checked against the advisory; kept as its own grade in every record.
- `commit_message_reference`: neither, but the commit message itself names the CVE or its GHSA id (the
  maintainers say the commit is about this advisory). The weakest accepted grade, kept in every record.
- heuristic: only MoreFixes' mining (a Prospector score) links the commit to the CVE. Not accepted:
  such a commit is a candidate fix, not evidence (`discover` rejects it as `fix_commit_heuristic_only`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sentinel.garden import code

ZENODO_RECORD = "20776007"
DUMP_NAME = "dump-2026-06-20.sql.gz"
DUMP_MD5 = "bdff6120f1e65626c77d42656162316e"

# columns read from each table (everything else, including all code and diffs, is skipped)
TABLES: dict[str, tuple[str, ...]] = {
    "fixes": ("cve_id", "hash", "repo_url", "rel_type", "score"),
    "cve": ("cve_id", "published_date", "last_modified_date", "description", "reference_json"),
    "cwe_classification": ("cve_id", "cwe_id"),
    "repository": ("repo_url", "repo_language", "date_created"),
    "commits": ("hash", "repo_url", "msg", "parents", "merge", "committer_date"),
    "file_change": ("hash", "old_path", "new_path", "change_type", "num_lines_added", "num_lines_deleted"),
}

_ESCAPE = re.compile(r"\\(?:([0-7]{1,3})|x([0-9a-fA-F]{1,2})|(.))")
_SIMPLE = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}
_COPY = re.compile(r"^COPY public\.(\w+) \(([^)]*)\) FROM stdin;$")
_REPO = re.compile(r"^https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def _unescape(field: str) -> str | None:
    if field == "\\N":
        return None
    if "\\" not in field:
        return field

    def sub(m: re.Match[str]) -> str:
        if m.group(1):
            return chr(int(m.group(1), 8))
        if m.group(2):
            return chr(int(m.group(2), 16))
        return _SIMPLE.get(m.group(3), m.group(3))

    return _ESCAPE.sub(sub, field)


def copy_rows(lines: Iterator[str], tables: dict[str, tuple[str, ...]] = TABLES) -> Iterator[tuple[str, dict[str, str | None]]]:
    """(table, row) for the wanted columns of the wanted tables of a `pg_dump` text stream."""
    table: str | None = None
    indexes: list[tuple[str, int]] = []
    for line in lines:
        if table is None:
            if line.startswith("COPY "):
                m = _COPY.match(line.rstrip("\n"))
                if m and m.group(1) in tables:
                    columns = [c.strip() for c in m.group(2).split(",")]
                    indexes = [(c, columns.index(c)) for c in tables[m.group(1)]]
                    table = m.group(1)
            continue
        if line.startswith("\\.") and line.rstrip("\n") == "\\.":
            table = None
            continue
        fields = line.rstrip("\n").split("\t")
        yield table, {name: _unescape(fields[i]) for name, i in indexes}


def repository_key(url: str | None) -> str | None:
    m = _REPO.match((url or "").strip())
    return f"{m.group(1)}/{m.group(2)}".lower() if m else None


def extract_index(dump: Path, out_dir: Path, log: Any = print) -> dict[str, Any]:
    """Stream the dump once and write the metadata tables as gzipped JSONL under `out_dir`.

    Rows of `commits` and `file_change` are kept only for target-language work: a commit is kept when
    one of its changed files has a Garden language extension (decided from `file_change`, which the
    dump stores after `commits`, so commits are buffered by hash first)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    writers = {t: gzip.open(out_dir / f"{t}.jsonl.gz", "wt", encoding="utf-8") for t in ("fixes", "cve", "cwe_classification", "repository")}
    commits: dict[str, dict[str, Any]] = {}
    changes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proc = subprocess.Popen(["zcat", str(dump)], stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1 << 20)
    assert proc.stdout is not None
    try:
        for table, row in copy_rows(proc.stdout):
            counts[table] += 1
            if table == "commits":
                msg = row["msg"] or ""
                row["msg"] = msg[:4000]
                commits[(row["hash"] or "") + "\x00" + (row["repo_url"] or "")] = row
            elif table == "file_change":
                paths = [p for p in (row["new_path"], row["old_path"]) if p and p != "None"]
                if any(code.language_of(p) for p in paths):
                    changes[row["hash"] or ""].append(row)
            else:
                writers[table].write(json.dumps(row, sort_keys=True) + "\n")
            if sum(counts.values()) % 500000 == 0:
                log(f"morefixes: {dict(counts)}")
    finally:
        proc.stdout.close()
        proc.wait()
        for w in writers.values():
            w.close()
    kept = 0
    with gzip.open(out_dir / "commits.jsonl.gz", "wt", encoding="utf-8") as w:
        for key in sorted(commits):
            row = commits[key]
            if row["hash"] in changes:
                w.write(json.dumps(row, sort_keys=True) + "\n")
                kept += 1
    with gzip.open(out_dir / "file_change.jsonl.gz", "wt", encoding="utf-8") as w:
        for sha in sorted(changes):
            for row in changes[sha]:
                w.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"rows": dict(sorted(counts.items())), "target_language_commits": kept, "target_language_file_changes": sum(len(v) for v in changes.values())}
    (out_dir / "index_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"morefixes: index written {summary}")
    return summary


def read_table(index_dir: Path, table: str) -> Iterator[dict[str, Any]]:
    with gzip.open(index_dir / f"{table}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def reference_urls(reference_json: str | None) -> list[str]:
    if not reference_json:
        return []
    try:
        refs = json.loads(reference_json)
    except json.JSONDecodeError:
        return []
    urls = []
    for ref in refs if isinstance(refs, list) else []:
        url = ref.get("url") if isinstance(ref, dict) else None
        if isinstance(url, str):
            urls.append(url.strip())
    return urls


_COMMIT_URL = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/(?:pull/\d+/)?commits?/([0-9a-fA-F]{7,40})", re.IGNORECASE)
_GHSA = re.compile(r"GHSA(?:-[23456789cfghjmpqrvwx]{4}){3}")


def referenced_commit(urls: list[str], repo_key: str, sha: str) -> bool:
    """The CVE's references name this commit of this repository (an abbreviated hash counts if >= 7)."""
    for url in urls:
        m = _COMMIT_URL.search(url)
        if m and f"{m.group(1)}/{m.group(2)}".lower() == repo_key and sha.lower().startswith(m.group(3).lower()):
            return True
    return False


def ghsa_aliases(urls: list[str]) -> list[str]:
    return sorted({m.group(0) for url in urls for m in _GHSA.finditer(url)})


def cve_record_sha256(cve: dict[str, Any], cwes: list[str]) -> str:
    payload = {k: cve.get(k) for k in ("cve_id", "published_date", "last_modified_date", "description", "reference_json")}
    payload["cwes"] = sorted(cwes)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def iso_time(value: str | None) -> str:
    """MoreFixes stores NVD timestamps as text ('2021-05-11T15:15:08.123' or '... +00'); normalize to UTC Z."""
    text = (value or "").strip().replace(" ", "T")
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}:\d{2}))?", text)
    if not m:
        return ""
    return f"{m.group(1)}T{m.group(2) or '00:00:00'}Z"


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# boilerplate sentences of CNA-generated descriptions that say nothing about the weakness
_BOILERPLATE = re.compile(
    r"(?i)^(a vulnerability (has been|was) (found|identified|discovered|detected|classified)|it is (recommended|possible|suggested)|"
    r"the (exploit|identifier|patch|vendor|associated identifier)|vdb-|this (issue|vulnerability) (is|has been) (fixed|patched|addressed)|"
    r"(users|upgrade|update|version) .* (should|recommended|fixed|patched)|this is fixed|the attack (may|can) be (launched|initiated))"
)
_WEAKNESS_WORDS = re.compile(
    r"(?i)\b(vulnerab|inject|xss|cross[- ]site|travers|overflow|bypass|allow|attacker|denial of service|disclos|forger|escalat|execut|leak|"
    r"unauthori|improper|missing|unsafe|insecure|arbitrary|sanitiz|validat|manipulation|expos|redirect|deserializ|prototype pollution|redos|"
    r"regular expression|timing|weak|hard-?coded|privilege|authenticat|authoriz)"
)


def summary_of(description: str, limit: int = 320) -> str:
    """The first sentence of the CVE description that describes the weakness (product blurbs and CNA
    boilerplate are skipped), plus the next one when it is short; at most `limit` characters."""
    text = " ".join((description or "").split())
    sentences = [x for x in _SENTENCE.split(text) if x]
    useful = [i for i, x in enumerate(sentences) if not _BOILERPLATE.match(x)]
    weak = [i for i in useful if _WEAKNESS_WORDS.search(sentences[i])]
    if not sentences:
        return ""
    first = weak[0] if weak else (useful[0] if useful else 0)
    summary = sentences[first]
    following = [i for i in weak if i > first]
    if len(summary) < 80 and following:
        summary += " " + sentences[following[0]]
    if len(summary) > limit:
        summary = summary[: limit - 1].rsplit(" ", 1)[0] + "…"
    return summary


def osv_like(cve: dict[str, Any], aliases: list[str], urls: list[str]) -> dict[str, Any]:
    """The CVE as an OSV-shaped record, so the audit and the review packages read it like an advisory."""
    return {
        "id": cve["cve_id"],
        "aliases": aliases,
        "summary": summary_of(cve.get("description") or ""),
        "details": cve.get("description") or "",
        "published": iso_time(cve.get("published_date")),
        "modified": iso_time(cve.get("last_modified_date")),
        "references": [{"type": "WEB", "url": u} for u in urls],
    }


CANDIDATE_SET = "garden-v6-morefixes"
EVIDENCE_GRADES = ("nvd_reference", "ghsa_reference", "commit_message_reference")  # strongest first


def names_advisory(message: str, ids: list[str]) -> bool:
    """The commit message names the CVE or one of its GHSA ids as a whole word."""
    return any(re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", message, re.IGNORECASE) for i in ids if i)
_CWE = re.compile(r"^CWE-[1-9]\d*$")


def load_index(index_dir: Path) -> dict[str, Any]:
    cves = {r["cve_id"]: r for r in read_table(index_dir, "cve") if r.get("cve_id")}
    cwes: dict[str, list[str]] = defaultdict(list)
    for row in read_table(index_dir, "cwe_classification"):
        if row.get("cve_id") and row.get("cwe_id"):
            cwes[row["cve_id"]].append(row["cwe_id"].strip())
    fixes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_table(index_dir, "fixes"):
        if row.get("cve_id") and row.get("hash"):
            fixes[row["cve_id"]].append(row)
    commits: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_table(index_dir, "commits"):
        key = repository_key(row.get("repo_url"))
        if key and row.get("hash"):
            commits[(key, row["hash"])] = row
    changes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_table(index_dir, "file_change"):
        changes[row["hash"]].append(row)
    repositories = {repository_key(r.get("repo_url")): r for r in read_table(index_dir, "repository")}
    by_repository: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cve_id, rows in fixes.items():
        for row in rows:
            key = repository_key(row.get("repo_url"))
            if key and len(row["hash"]) == 40:
                by_repository[key].append((cve_id, row["hash"]))
    return {
        "cves": cves,
        "cwes": cwes,
        "fixes": fixes,
        "commits": commits,
        "changes": changes,
        "repositories": repositories,
        "fixes_by_repository": dict(by_repository),
    }


def runtime_language_files(changes: list[dict[str, Any]]) -> list[str]:
    paths = {p for row in changes for p in (row.get("new_path"), row.get("old_path")) if p and p != "None"}
    return sorted(p for p in paths if code.language_of(p) and code.path_role(p) == "source")


def repository_vulns(index: dict[str, Any], repo_key: str, path: str | None = None) -> list[dict[str, Any]]:
    """Every CVE MoreFixes links to a fix commit of this repository, as OSV-shaped records whose FIX
    references are those commits (for the audit's later-fix check; any evidence grade). With `path`, only
    commits whose recorded file changes touch that file (MoreFixes records them for target-language files)."""
    if "fixes_by_repository" not in index:  # built once: the audit asks for every (repository, file)
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for cve_id, rows in index["fixes"].items():
            for row in rows:
                key = repository_key(row.get("repo_url"))
                if key and len(row["hash"]) == 40:
                    grouped[key].append((cve_id, row["hash"]))
        index["fixes_by_repository"] = dict(grouped)
    by_cve: dict[str, list[str]] = defaultdict(list)
    for cve_id, sha in index["fixes_by_repository"].get(repo_key, []):
        if path is not None and not any(path in (c.get("new_path"), c.get("old_path")) for c in index["changes"].get(sha, [])):
            continue
        by_cve[cve_id].append(sha)
    out = []
    for cve_id, shas in sorted(by_cve.items()):
        cve = index["cves"].get(cve_id, {"cve_id": cve_id})
        urls = reference_urls(cve.get("reference_json"))
        record = osv_like({**cve, "cve_id": cve_id}, ghsa_aliases(urls), urls)
        owner, name = repo_key.split("/", 1)
        record["references"] = record["references"] + [{"type": "FIX", "url": f"https://github.com/{owner}/{name}/commit/{sha}"} for sha in sorted(set(shas))]
        out.append(record)
    return out


def discover_candidates(
    index: dict[str, Any],
    frozen: list[dict[str, Any]],
    settings: dict[str, Any],
    rejected_cwes: set[str],
    max_cwes: int,
    salt: str,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Candidates for CVEs no frozen candidate already covers.

    Returns (candidates, outcomes, osv-shaped CVE records by id, corroborations). A CVE that is an
    alias of a frozen advisory, or whose fix commit a frozen candidate already uses, adds evidence to
    that candidate (a corroboration row), never a second candidate."""
    known_ids: dict[str, str] = {}
    for c in frozen:
        for i in [c["advisory_id"], *c.get("aliases", [])]:
            known_ids[i] = c["advisory_id"]
    known_commits = {(c["repository_key"], sha): c["advisory_id"] for c in frozen for sha in (c.get("fix_commits") or [c["fix_commit"]]) if sha}
    per_repo: dict[str, int] = defaultdict(int)
    for c in frozen:
        per_repo[c["repository_key"]] += 1
    cutoff = settings["published_before"]
    cap = settings["max_candidates_per_repository"]
    outcomes: dict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    corroborations: list[dict[str, str]] = []
    order = sorted(index["fixes"], key=lambda cve_id: hashlib.sha256(f"{salt}{cve_id}".encode()).hexdigest())
    for cve_id in order:
        cve = index["cves"].get(cve_id)
        rows = index["fixes"][cve_id]
        target = [r for r in rows if (index["changes"].get(r["hash"]) and runtime_language_files(index["changes"][r["hash"]]))]
        if not target:
            outcomes["no_target_language_change"] += 1
            continue
        if cve is None:
            outcomes["cve_record_missing"] += 1
            continue
        urls = reference_urls(cve.get("reference_json"))
        aliases = ghsa_aliases(urls)
        known = next((known_ids[i] for i in [cve_id, *aliases] if i in known_ids), None)
        if known:
            outcomes["corroborates_frozen_advisory"] += 1
            corroborations.append({"cve_id": cve_id, "advisory_id": known, "kind": "same_vulnerability"})
            continue
        published = iso_time(cve.get("published_date"))
        if not published or published >= cutoff:
            outcomes["published_after_cutoff"] += 1
            continue
        raw_cwes = sorted({c for c in index["cwes"].get(cve_id, [])})
        cwes = [c for c in raw_cwes if _CWE.match(c)]
        if not cwes:
            outcomes["no_specific_cwe"] += 1
            continue
        if any(c in rejected_cwes for c in cwes):
            outcomes["rejected_cwe"] += 1
            continue
        if len(cwes) > max_cwes:
            outcomes["too_many_cwes"] += 1
            continue
        # evidence that ties a commit to the CVE: its NVD references name it (verified here, from the CVE
        # record), else a GitHub Advisory Database record names it (MoreFixes rel_type GHSD_DIRECT_COMMIT)
        graded: dict[str, dict[str, list[str]]] = {grade: defaultdict(list) for grade in EVIDENCE_GRADES}
        for row in target:
            key = repository_key(row.get("repo_url"))
            if not key or len(row["hash"]) != 40:
                continue
            message = (index["commits"].get((key, row["hash"])) or {}).get("msg") or ""
            if referenced_commit(urls, key, row["hash"]):
                graded["nvd_reference"][key].append(row["hash"])
            elif row.get("rel_type") == "GHSD_DIRECT_COMMIT":
                graded["ghsa_reference"][key].append(row["hash"])
            elif names_advisory(message, [cve_id, *aliases]):
                graded["commit_message_reference"][key].append(row["hash"])
        evidence = next((grade for grade in EVIDENCE_GRADES if graded[grade]), None)
        if evidence is None:
            outcomes["fix_commit_heuristic_only"] += 1
            continue
        by_repo = graded[evidence]
        if len(by_repo) > 1:
            outcomes["fix_in_several_repositories"] += 1
            continue
        key, shas = next(iter(by_repo.items()))
        shas = sorted(set(shas))
        if any((key, sha) in known_commits for sha in shas):
            outcomes["corroborates_frozen_fix_commit"] += 1
            corroborations.append({"cve_id": cve_id, "advisory_id": known_commits[next((key, s) for s in shas if (key, s) in known_commits)], "kind": "same_fix_commit"})
            continue
        if len(shas) > settings.get("max_fix_commits", 3):
            outcomes["too_many_fix_commits"] += 1
            continue
        if per_repo[key] >= cap:
            outcomes["repository_cap"] += 1
            continue
        commit_rows = [index["commits"].get((key, s)) for s in shas]
        if any(r is None for r in commit_rows):
            outcomes["commit_metadata_missing"] += 1
            continue
        per_repo[key] += 1
        for sha in shas:
            known_commits[(key, sha)] = cve_id
        known_ids[cve_id] = cve_id
        owner, name = (index["commits"][(key, shas[0])]["repo_url"].rstrip("/").split("/")[-2:])
        candidate = {
            "advisory_id": cve_id,
            "aliases": aliases,
            "cwe_ids": sorted(cwes, key=lambda c: int(c[4:]))[:max_cwes],
            "ecosystem": "GIT",
            "packages": [],
            "published": published,
            "osv_modified": iso_time(cve.get("last_modified_date")),
            "osv_record_sha256": cve_record_sha256(cve, cwes),
            "repository": f"https://github.com/{owner}/{name.removesuffix('.git')}",
            "repository_key": key,
            "severity": None,
            "summary": summary_of(cve.get("description") or ""),
            "fix_commit": shas[0] if len(shas) == 1 else None,
            "candidate_set": CANDIDATE_SET,
            "fix_evidence": evidence,
        }
        if len(shas) > 1:
            candidate["fix_commits"] = shas
        candidates.append(candidate)
        records[cve_id] = osv_like(cve, aliases, urls)
        outcomes[f"candidate_{evidence}"] += 1
    return sorted(candidates, key=lambda c: c["advisory_id"]), dict(sorted(outcomes.items())), records, corroborations
