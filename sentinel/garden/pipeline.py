"""Sentinel Garden pipeline: discover -> build -> curate (-> verify / check).

    discover  network: OSV export (metadata)       -> <committed>/candidates.jsonl (frozen)
    build     network: upstream git (by full SHA)  -> <workdir>/records.raw.jsonl + build_log.jsonl
    curate    offline                              -> <workdir>/{records,train,validation,staging,holdout_reserved,rejected}.jsonl
                                                      <committed>/{manifest.jsonl,stats.json,REPORT.md,REVIEW_QUEUE.md}
    verify    rebuilds from the frozen candidates and compares every content hash with the manifest
    check     offline consistency check of the committed manifest (CI / foundation)

Committed files are metadata only: no upstream code is stored in this repository.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from sentinel.data.registry import load_registry, validate_registry
from sentinel.eval.io import case_set_sha256, validate_cases
from sentinel.garden import build as build_module
from sentinel.garden import human_review, render, supervision, syntax
from sentinel.garden.build import ExtractionSettings, Rejected
from sentinel.garden.audit import OsvQuery, load_exclusions, load_manual_review, manual_exclusions, run_audit, summarize
from sentinel.garden.audit import report_markdown as audit_report_markdown
from sentinel.garden.governance import gate, load_approvals
from sentinel.garden.review import DepsDev, collect_evidence, write_review_outputs
from sentinel.garden.gitsource import UpstreamRepo
from sentinel.garden.osv import OsvClient, parse_candidate, sample_key
from sentinel.garden.quality import (
    QualitySettings,
    check_pairs,
    check_record,
    contamination,
    deduplicate,
    load_eval_references,
    load_record_schema,
)
from sentinel.garden.split import SplitSettings, assign_split, check_split_invariants, load_split_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "garden" / "seed_v1.yaml"
PIPELINE_VERSION = "3"  # 2: tree-sitter parse trees; 3: manifest supervision hashes, patch task, no fixed confidence
MANIFEST_DROP_FIELDS = ("input", "target")  # code, context, evidence text and patch stay out of Git


@dataclass(frozen=True)
class GardenConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def dataset_id(self) -> str:
        return str(self.raw["dataset_id"])

    @property
    def workdir(self) -> Path:
        return PROJECT_ROOT / self.raw["workdir"]

    @property
    def committed_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["committed_dir"]

    @property
    def code_source_id(self) -> str:
        return str(self.raw["sources"]["code_source"])

    @property
    def metadata_source_ids(self) -> list[str]:
        return list(self.raw["sources"]["metadata_sources"])

    @property
    def extraction(self) -> ExtractionSettings:
        return ExtractionSettings(**self.raw["extraction"])

    @property
    def quality(self) -> QualitySettings:
        return QualitySettings(**self.raw["quality"])

    @property
    def split(self) -> SplitSettings:
        return load_split_policy(PROJECT_ROOT / self.raw["split"]["policy"])


def load_config(path: Path) -> GardenConfig:
    data = path.read_bytes()
    raw = yaml.safe_load(data)
    required = {"schema_version", "dataset_id", "sources", "workdir", "committed_dir", "discovery", "selection", "extraction", "quality", "split", "contamination"}
    if not isinstance(raw, dict) or required - set(raw):
        raise ValueError(f"garden config must define {sorted(required)}")
    if raw["schema_version"] != "1.0":
        raise ValueError("unsupported garden config schema_version")
    if not str(raw["workdir"]).startswith("artifacts/"):
        raise ValueError("workdir must live under the Git-ignored artifacts/ directory")
    config = GardenConfig(path=path, raw=raw, sha256=hashlib.sha256(data).hexdigest())
    # validate typed sections early
    config.extraction, config.quality, config.split  # noqa: B018
    return config


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Stage 1: discover
# --------------------------------------------------------------------------------------------


EXTENSION_SET = "garden-v5-extension"


def discover_extension(config: GardenConfig, log: Any = print, resolver: Any = None) -> dict[str, Any]:
    """Garden v5: candidates beyond the frozen list, appended to it (the frozen rows stay byte-identical).

    From the same cached OSV snapshot, in the same salted order, after the frozen list's commits and
    per-repository counts:
    - advisories that reference exactly one pull request and no fix commit, resolved to the commit that
      merged it (`sentinel.garden.pulls`, frozen in `pull_request_resolutions.jsonl`);
    - advisories the per-repository cap left out, up to `extension.max_candidates_per_repository`.
    Every new candidate is marked `candidate_set: garden-v5-extension`. The source approvals in
    data/registry.yaml cover the frozen list only, so records of these candidates carry the blocker
    `source_approval_not_extended` until the owner extends them (`curate_records`)."""
    from sentinel.garden.pulls import PullRequestResolver

    d, sel, ext = config.raw["discovery"], config.raw["selection"], config.raw["discovery"]["extension"]
    client = OsvClient(d["osv_bucket"], config.workdir / "cache" / "osv")
    path = config.committed_dir / "candidates.jsonl"
    frozen = _read_jsonl(path)
    frozen_ids = {c["advisory_id"] for c in frozen}
    seen: set[tuple[str, str]] = {(c["repository_key"], sha) for c in frozen for sha in c.get("fix_commits") or [c["fix_commit"]]}
    per_repo: Counter[str] = Counter(c["repository_key"] for c in frozen)
    per_ecosystem: Counter[str] = Counter(c["ecosystem"] for c in frozen)
    resolver = resolver or PullRequestResolver(config.committed_dir / "pull_request_resolutions.jsonl", log=log)
    added: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    for ecosystem in d["ecosystems"]:
        ids = client.advisory_ids(ecosystem, d["advisory_prefix"])
        ordered = sorted(ids, key=lambda advisory_id: sample_key(d["sample_salt"], advisory_id))[: d["inspect_per_ecosystem"]]
        for advisory_id, record in zip(ordered, client.records(ecosystem, ordered, d["workers"])):
            if record is None or advisory_id in frozen_ids:
                continue
            candidate, reason = parse_candidate(
                record,
                ecosystem,
                published_before=d["published_before"],
                require_reviewed=d["require_github_reviewed"],
                rejected_cwes=set(sel["rejected_cwes"]),
                max_cwes=sel["max_cwes"],
                allow_multiple_fix_commits=bool(d.get("multi_commit_fixes", False)),
                allow_pull_requests=bool(ext.get("pull_request_fixes", False)),
            )
            if candidate is None:
                continue
            if candidate.get("fix_pull_request"):
                number = candidate["fix_pull_request"]["number"]
                resolution = resolver.resolve(candidate["repository_key"], number)
                if resolution["status"] != "resolved":
                    outcomes[f"pull_request_{resolution['status']}"] += 1
                    continue
                candidate["fix_commit"] = resolution["commit"]
                candidate["fix_pull_request"] = {"number": number, "merged_by": resolution["kind"]}
            commits = [(candidate["repository_key"], sha) for sha in candidate.get("fix_commits") or [candidate["fix_commit"]]]
            if any(c in seen for c in commits):
                outcomes["duplicate_fix_commit"] += 1
                continue
            if per_repo[candidate["repository_key"]] >= ext["max_candidates_per_repository"]:
                outcomes["extension_repository_cap"] += 1
                continue
            if per_ecosystem[ecosystem] >= d["max_candidates_per_ecosystem"]:
                outcomes["ecosystem_cap"] += 1
                continue
            seen.update(commits)
            per_repo[candidate["repository_key"]] += 1
            per_ecosystem[ecosystem] += 1
            candidate["candidate_set"] = EXTENSION_SET
            added.append(candidate)
            outcomes["candidate_pull_request" if candidate.get("fix_pull_request") else "candidate_over_v3_repository_cap"] += 1
    resolver.write()
    rows = frozen + sorted(added, key=lambda c: c["advisory_id"])
    _write_jsonl(path, rows)
    log(f"discover: {len(added)} extension candidates {dict(sorted(outcomes.items()))}")
    return {"added": len(added), "outcomes": dict(sorted(outcomes.items()))}


def morefixes_index_dir(config: GardenConfig) -> Path:
    return PROJECT_ROOT / config.raw["discovery"]["morefixes"]["index_dir"]


def morefixes_index(config: GardenConfig, log: Any = print) -> dict[str, Any]:
    """Garden v6: check the pinned MoreFixes dump (md5 from Zenodo) and stream its metadata tables into the index."""
    import hashlib as _hashlib

    from sentinel.garden import morefixes

    m = config.raw["discovery"]["morefixes"]
    dump = PROJECT_ROOT / m["dump"]
    digest = _hashlib.md5(usedforsecurity=False)
    with dump.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    if digest.hexdigest() != m["dump_md5"]:
        raise ValueError(f"{dump} does not match dump_md5 (download it again from Zenodo record {morefixes.ZENODO_RECORD})")
    return morefixes.extract_index(dump, morefixes_index_dir(config), log)


def discover_morefixes(config: GardenConfig, log: Any = print) -> dict[str, Any]:
    """Garden v6: CVE fix commits from the MoreFixes index, appended after the frozen candidate rows.

    Only fix commits the CVE's own NVD references name are candidates (`morefixes.discover_candidates`);
    a CVE already covered by a frozen candidate (alias or same fix commit) is written to
    `morefixes_corroborations.jsonl` instead. The CVE records are cached OSV-shaped for the audit."""
    from sentinel.garden import morefixes

    d, sel = config.raw["discovery"], config.raw["selection"]
    m = d["morefixes"]
    index = morefixes.load_index(morefixes_index_dir(config))
    path = config.committed_dir / "candidates.jsonl"
    frozen = [c for c in _read_jsonl(path) if c.get("candidate_set") != morefixes.CANDIDATE_SET]
    settings = {
        "published_before": d["published_before"],
        "max_candidates_per_repository": m["max_candidates_per_repository"],
        "max_fix_commits": m.get("max_fix_commits", 3),
    }
    added, outcomes, cve_records, corroborations = morefixes.discover_candidates(
        index, frozen, settings, set(sel["rejected_cwes"]), sel["max_cwes"], d["sample_salt"]
    )
    write_morefixes_records(config, cve_records)
    _write_jsonl(path, frozen + added)
    _write_jsonl(config.committed_dir / "morefixes_corroborations.jsonl", sorted(corroborations, key=lambda r: (r["cve_id"], r["advisory_id"])))
    log(f"discover: {len(added)} MoreFixes candidates {outcomes}")
    return {"added": len(added), "cves_with_fixes": len(index["fixes"]), "outcomes": outcomes, "corroborations": len(corroborations)}


def write_morefixes_records(config: GardenConfig, cve_records: dict[str, dict[str, Any]]) -> None:
    directory = config.workdir / "cache" / "osv" / "GIT"
    directory.mkdir(parents=True, exist_ok=True)
    for cve_id, record in cve_records.items():
        (directory / f"{cve_id}.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def candidate_sources(config: GardenConfig, candidate: dict[str, Any]) -> tuple[str, list[str]]:
    """Registry source ids of a candidate: Garden v6 MoreFixes candidates have their own (pending) sources."""
    m = (config.raw.get("discovery") or {}).get("morefixes")
    if m and candidate.get("candidate_set") == "garden-v6-morefixes":
        return m["code_source_id"], list(m["metadata_source_ids"])
    return config.code_source_id, config.metadata_source_ids


def build_query_tests(config: GardenConfig, log: Any = print) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Garden v6: rule-scoped records from the pinned CodeQL security query tests (sentinel/garden/querytests.py)."""
    from sentinel.garden import querytests

    q = config.raw["query_tests"]
    checkout = querytests.ensure_checkout(config.workdir / "cache", q["revision"], log)
    sel = config.raw["selection"]
    records, rows = querytests.build_records(checkout, q["revision"], config.extraction, config.dataset_id, set(sel["rejected_cwes"]), sel["max_cwes"], log)
    log(f"build: {len(records)} query-test records from {len(rows)} test units")
    return records, rows


def discover(config: GardenConfig, refresh: bool = False, log: Any = print) -> dict[str, Any]:
    d, sel = config.raw["discovery"], config.raw["selection"]
    if d.get("frozen_candidates"):
        # Garden v4: the candidate list of an earlier version, unchanged (no OSV query), so the source
        # approvals bound to that list's sha256 cover exactly the same advisories and fix commits.
        source = PROJECT_ROOT / d["frozen_candidates"]
        if file_sha256(source) != d["frozen_candidates_sha256"]:
            raise ValueError(f"{source} does not match frozen_candidates_sha256; the frozen list must not change")
        config.committed_dir.mkdir(parents=True, exist_ok=True)
        (config.committed_dir / "candidates.jsonl").write_bytes(source.read_bytes())
        stats = json.loads((source.parent / "discovery_stats.json").read_text(encoding="utf-8"))
        stats["frozen_candidates"] = {"path": d["frozen_candidates"], "sha256": d["frozen_candidates_sha256"]}
        _write_json(config.committed_dir / "discovery_stats.json", stats)
        log(f"discover: frozen candidate list {d['frozen_candidates']} copied unchanged")
        if d.get("extension"):
            stats["extension"] = discover_extension(config, log)
            _write_json(config.committed_dir / "discovery_stats.json", stats)
        if d.get("morefixes"):
            stats["morefixes"] = discover_morefixes(config, log)
            _write_json(config.committed_dir / "discovery_stats.json", stats)
        return stats
    client = OsvClient(d["osv_bucket"], config.workdir / "cache" / "osv")
    candidates: list[dict[str, Any]] = []
    seen_commits: set[tuple[str, str]] = set()
    per_repo: Counter[str] = Counter()
    stats: dict[str, Any] = {"ecosystems": {}}
    for ecosystem in d["ecosystems"]:
        ids = client.advisory_ids(ecosystem, d["advisory_prefix"], refresh=refresh)
        ordered = sorted(ids, key=lambda advisory_id: sample_key(d["sample_salt"], advisory_id))[: d["inspect_per_ecosystem"]]
        log(f"discover {ecosystem}: {len(ids)} {d['advisory_prefix']} advisories in export, inspecting {len(ordered)}")
        records = client.records(ecosystem, ordered, d["workers"])
        reasons: Counter[str] = Counter()
        accepted = 0
        multi_commit = bool(d.get("multi_commit_fixes", False))
        deferred: list[dict[str, Any]] = []

        def admit(candidate: dict[str, Any]) -> bool:
            nonlocal accepted
            commits = [(candidate["repository_key"], sha) for sha in candidate.get("fix_commits") or [candidate["fix_commit"]]]
            if any(commit in seen_commits for commit in commits):
                reasons["duplicate_fix_commit"] += 1
                return False
            if per_repo[candidate["repository_key"]] >= d["max_candidates_per_repository"]:
                reasons["repository_cap"] += 1
                return False
            if accepted >= d["max_candidates_per_ecosystem"]:
                reasons["ecosystem_cap"] += 1
                return False
            seen_commits.update(commits)
            per_repo[candidate["repository_key"]] += 1
            accepted += 1
            candidates.append(candidate)
            return True

        for advisory_id, record in zip(ordered, records):
            if record is None:
                reasons["record_missing"] += 1
                continue
            candidate, reason = parse_candidate(
                record,
                ecosystem,
                published_before=d["published_before"],
                require_reviewed=d["require_github_reviewed"],
                rejected_cwes=set(sel["rejected_cwes"]),
                max_cwes=sel["max_cwes"],
                allow_multiple_fix_commits=multi_commit,
            )
            if candidate is None:
                reasons[reason] += 1
                continue
            if candidate.get("fix_commits"):
                deferred.append(candidate)  # second pass: never displaces a single-commit candidate
                continue
            admit(candidate)
        # Multi-commit candidates only take the cap slots single-commit candidates left free, so a
        # dataset version that enables them is a superset of the one that does not.
        for candidate in deferred:
            if admit(candidate):
                reasons["candidate_multi_commit"] += 1
        reasons["candidate"] = accepted
        stats["ecosystems"][ecosystem] = {
            "advisories_in_export": len(ids),
            "inspected": len(ordered),
            "outcomes": dict(sorted(reasons.items())),
        }
    candidates.sort(key=lambda c: c["advisory_id"])
    _write_jsonl(config.committed_dir / "candidates.jsonl", candidates)
    _write_json(config.committed_dir / "discovery_stats.json", stats)
    log(f"discover: {len(candidates)} candidates -> {config.committed_dir / 'candidates.jsonl'}")
    return stats


# --------------------------------------------------------------------------------------------
# Stage 2-3: build
# --------------------------------------------------------------------------------------------


def build(
    config: GardenConfig, workdir: Path | None = None, workers: int = 6, log: Any = print, processes: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workdir = workdir or config.workdir
    candidates = _read_jsonl(config.committed_dir / "candidates.jsonl")
    records, build_log = build_candidates(config, candidates, workers, log, processes=processes)
    if config.raw.get("query_tests"):
        tool_records, tool_log = build_query_tests(config, log)
        records = sorted(records + tool_records, key=lambda r: r["record_id"])
        build_log = build_log + tool_log
    _write_jsonl(workdir / "records.raw.jsonl", records)
    _write_jsonl(workdir / "build_log.jsonl", build_log)
    log(f"build: {len(records)} raw records from {sum(1 for r in build_log if r['outcome'] == 'built')}/{len(candidates)} candidates")
    return records, build_log


def _build_repository(config: GardenConfig, repo_key: str, group: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]], str]]:
    """Build one repository's candidates (one worker per repository: its git cache is never shared)."""
    results = []
    repo = UpstreamRepo(config.workdir / "cache" / "git", group[0]["repository"], repo_key)
    for candidate in group:
        try:
            code_source_id, metadata_source_ids = candidate_sources(config, candidate)
            records = build_module.build_records(candidate, repo, config.extraction, config.dataset_id, code_source_id, metadata_source_ids)
            results.append((candidate, records, "built"))
        except Rejected as exc:
            results.append((candidate, [], exc.reason))
        except Exception as exc:  # noqa: BLE001 - one broken upstream must not stop the run
            results.append((candidate, [], f"error:{type(exc).__name__}"))
    return results


def build_candidates(
    config: GardenConfig, candidates: list[dict[str, Any]], workers: int = 6, log: Any = print, processes: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the given candidates from the upstream cache (fetching what is missing); writes nothing.

    `processes` > 1 builds repositories in that many processes (parsing is CPU-bound; threads share
    one core); the result is the same, as every candidate is built independently and sorted."""
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_repo[candidate["repository_key"]].append(candidate)

    def process(repo_key: str) -> list[tuple[dict[str, Any], list[dict[str, Any]], str]]:
        return _build_repository(config, repo_key, by_repo[repo_key])

    records: list[dict[str, Any]] = []
    build_log: list[dict[str, Any]] = []
    keys = sorted(by_repo)
    if processes > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool: Any = ProcessPoolExecutor(max_workers=processes)
        mapped = pool.map(_build_repository, [config] * len(keys), keys, [by_repo[k] for k in keys], chunksize=4)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        mapped = pool.map(process, keys)
    with pool:
        for done, results in enumerate(mapped, start=1):
            for candidate, built, outcome in results:
                records.extend(built)
                build_log.append(
                    {"advisory_id": candidate["advisory_id"], "repository_key": candidate["repository_key"], "outcome": outcome, "records": len(built)}
                )
            if done % 50 == 0:
                log(f"build: {done}/{len(by_repo)} repositories")
    records.sort(key=lambda r: r["record_id"])
    build_log.sort(key=lambda row: row["advisory_id"])
    return records, build_log


# --------------------------------------------------------------------------------------------
# Stage 4-6: curate (quality, dedup, contamination, split, governance, export)
# --------------------------------------------------------------------------------------------


# Garden v4 evaluation roles (docs/GARDEN_V4.md). The frozen split policy decides the split; the role
# adds what earlier versions already used for evaluation, so that v4 can select on, and finally test on,
# repositories no model of ours has been evaluated on.
EVALUATION_ROLES = ("training", "retired_development", "development", "previous_holdout", "final_test", "temporal_quarantine")
UNTOUCHED_ROLES = {"development", "final_test"}


def evaluation_history(config: GardenConfig) -> tuple[set[str], set[str]] | None:
    """(retired development repositories, previously exported holdout repositories), or None without roles.

    Retired: repositories of the earlier development set (accepted, training-approved validation
    records of `retired_development_manifest`). Previous holdout: repositories whose holdout cases an
    earlier version exported (accepted holdout_reserved records of `previous_holdout_manifests`)."""
    settings = config.raw.get("evaluation")
    if not settings:
        return None
    retired = {
        row["provenance"]["repository_key"]
        for row in _read_jsonl(PROJECT_ROOT / settings["retired_development_manifest"])
        if row["split"] == "validation" and row["quality"]["status"] == "accepted" and row["governance"]["approval_status"] == "training_approved"
    }
    previous = {
        row["provenance"]["repository_key"]
        for path in settings["previous_holdout_manifests"]
        for row in _read_jsonl(PROJECT_ROOT / path)
        if row["split"] == "holdout_reserved" and row["quality"]["status"] == "accepted"
    }
    return retired, previous


def source_approved_advisories(config: GardenConfig) -> set[str] | None:
    """Garden v5: the advisories the registry's source approvals cover (`governance.source_approval_covers`,
    a candidate list bound by sha256), or None when every candidate of the dataset is covered."""
    covers = (config.raw.get("governance") or {}).get("source_approval_covers")
    if not covers:
        return None
    path = PROJECT_ROOT / covers["path"]
    if file_sha256(path) != covers["sha256"]:
        raise ValueError(f"{path} does not match governance.source_approval_covers.sha256")
    return {c["advisory_id"] for c in _read_jsonl(path)}


def unvalidated_negative_kinds(config: GardenConfig) -> set[str]:
    """Garden v5: negative kinds whose labels rest on a heuristic stay pending until an independent
    review of their sample validates the kind and the owner lists it in `governance.validated_negative_kinds`."""
    governance = config.raw.get("governance") or {}
    kinds = {"defended_sibling"}
    if config.raw.get("query_tests"):  # Garden v6: tool-test negatives are rule-scoped, a heuristic negative kind as well
        kinds.add("query_test_negative")
    return kinds - set(governance.get("validated_negative_kinds") or [])


def apply_negative_validation(record: dict[str, Any], unvalidated: set[str]) -> None:
    if record["kind"] not in unvalidated:
        return
    blocker = f"negative_kind_not_validated:{record['kind']}"
    governance = record["governance"]
    if blocker in governance["blockers"]:
        return
    status = "excluded" if governance["approval_status"] == "excluded" else "pending_review"
    record["governance"] = {"approval_status": status, "blockers": [blocker] + governance["blockers"]}


def apply_source_coverage(record: dict[str, Any], covered: set[str]) -> None:
    """A record of a candidate outside the approved list cannot be trained on until the owner extends
    the source approval to it (the registry approval names the list it covers)."""
    if record["provenance"]["advisory_id"] in covered:
        return
    governance = record["governance"]
    if governance["approval_status"] == "excluded":
        if "source_approval_not_extended" not in governance["blockers"]:
            governance["blockers"] = governance["blockers"] + ["source_approval_not_extended"]
        return
    blockers = [b for b in governance["blockers"] if b != "source_approval_not_extended"]
    record["governance"] = {"approval_status": "pending_review", "blockers": ["source_approval_not_extended"] + blockers}


def evaluation_role(record: dict[str, Any], retired: set[str], previous: set[str]) -> str:
    repository = record["provenance"]["repository_key"]
    split = record["split"]
    if split == "validation":
        return "retired_development" if repository in retired else "development"
    if split == "holdout_reserved":
        return "previous_holdout" if repository in previous else "final_test"
    return "training" if split == "train" else "temporal_quarantine"


def apply_evaluation_role(record: dict[str, Any], history: tuple[set[str], set[str]]) -> None:
    """Set the role; a development record is never exported for training, whatever its approvals."""
    role = evaluation_role(record, *history)
    record["evaluation_role"] = role
    if role == "development":
        governance = record["governance"]
        blockers = ["evaluation_role:development"] + [b for b in governance["blockers"] if b != "evaluation_role:development"]
        record["governance"] = {"approval_status": "excluded", "blockers": blockers}


def external_benchmarks(config: GardenConfig) -> dict[str, Any] | None:
    """Garden v6: vulnerabilities of external benchmarks we evaluate on (`evaluation.external_benchmarks`).

    Each file lists `{benchmark, id, aliases, repository_key, commits}` rows. A record of any of them
    (same advisory or alias, or same repository and fix/vulnerable commit) is never trained on."""
    entries = ((config.raw.get("evaluation") or {}).get("external_benchmarks")) or []
    if not entries:
        return None
    ids: dict[str, str] = {}
    commits: dict[tuple[str, str], str] = {}
    for entry in entries:
        path = PROJECT_ROOT / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match its sha256 in evaluation.external_benchmarks")
        for row in _read_jsonl(path):
            label = f"{row['benchmark']}:{row['id']}"
            for i in [row["id"], *row.get("aliases", [])]:
                ids[i] = label
            for sha in row.get("commits", []):
                commits[(row["repository_key"], sha)] = label
    return {"ids": ids, "commits": commits}


def apply_external_benchmark(record: dict[str, Any], benchmarks: dict[str, Any]) -> None:
    provenance = record["provenance"]
    label = next((benchmarks["ids"][i] for i in [provenance["advisory_id"], *provenance["aliases"]] if i in benchmarks["ids"]), None)
    if label is None:
        key = provenance["repository_key"]
        label = next(
            (benchmarks["commits"][(key, sha)] for sha in [provenance["vulnerable_revision"], provenance["fixed_revision"], *provenance.get("fix_commit_chain", [])] if (key, sha) in benchmarks["commits"]),
            None,
        )
    if label is None:
        return
    record["evaluation_role"] = "external_benchmark"
    blocker = f"external_benchmark:{label}"
    governance = record["governance"]
    record["governance"] = {"approval_status": "excluded", "blockers": [blocker] + [b for b in governance["blockers"] if b != blocker]}


def apply_governance(
    config: GardenConfig,
    records: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    approvals: dict[str, Any],
    validated_negative_kinds: set[str] | None = None,
) -> None:
    """Set every record's governance (and evaluation role) from its quality, split, the registry sources and
    the repository approvals. `validated_negative_kinds` overrides the config's list (approval scenarios only)."""
    # The training loader refuses every record of a repository that also has holdout or
    # temporal-quarantine records (sentinel/train/data.py). The same rule applies here, so an
    # approval never exports a record the loader would refuse (a new dataset version can add
    # post-cutoff records to an approved repository).
    held_out = {r["provenance"]["repository_key"] for r in records if r["split"] in {"holdout_reserved", "temporal_quarantine"}}
    history = evaluation_history(config)
    covered = source_approved_advisories(config)
    benchmarks = external_benchmarks(config)
    unvalidated = unvalidated_negative_kinds(config)
    if validated_negative_kinds is not None:
        unvalidated -= validated_negative_kinds
    for record in records:
        status, blockers = gate(record, sources, approvals)
        if status == "training_approved" and record["provenance"]["repository_key"] in held_out:
            status, blockers = "excluded", ["repository_has_holdout_or_quarantine_records"]
        record["governance"] = {"approval_status": status, "blockers": blockers}
        if covered is not None and record["provenance"]["code_source_id"] == config.code_source_id:
            # the source-approval list covers the GHSA/OSV source; other sources have their own registry entries
            apply_source_coverage(record, covered)
        if "governance" in config.raw:
            apply_negative_validation(record, unvalidated)
        if history is not None:
            apply_evaluation_role(record, history)
        if benchmarks:
            apply_external_benchmark(record, benchmarks)


def curate_records(config: GardenConfig, raw: list[dict[str, Any]], approvals: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    sel = config.raw["selection"]
    validator = Draft202012Validator(load_record_schema())
    registry = load_registry()
    validate_registry(registry)
    sources = {s["source_id"]: s for s in registry["sources"]}
    approvals = load_approvals() if approvals is None else approvals

    records = [dict(r) for r in sorted(raw, key=lambda r: r["record_id"])]
    failed: dict[str, list[str]] = {}
    flags: dict[str, list[str]] = {}
    for record in records:
        f, fl = check_record(
            record, validator, set(sel["rejected_cwes"]), set(sel["coarse_cwes"]), config.extraction.max_context_chars, config.extraction.min_function_lines
        )
        if not f and not render.reply_matches_contract(record):
            f.append("target_matches_eval_contract")
        failed[record["record_id"]], flags[record["record_id"]] = f, fl
    for record_id, pair_failures in check_pairs(records).items():
        failed[record_id].extend(pair_failures)
    for record_id, reasons in load_exclusions(config.committed_dir / "audit" / "exclusions.yaml").items():
        if record_id in failed:
            failed[record_id].extend(f"audit:{reason}" for reason in reasons)

    # a vulnerable record that fails a record-level check invalidates its derived siblings
    broken_groups = {
        r["group_id"]: all(c.startswith("audit:") for c in failed[r["record_id"]])
        for r in records
        if r["kind"] == "vulnerable" and failed[r["record_id"]]
    }
    for record in records:
        if record["kind"] != "vulnerable" and record["group_id"] in broken_groups and not failed[record["record_id"]]:
            # audit-only exclusions stay recognisable so a re-run of the audit sees the same population
            failed[record["record_id"]].append("audit:sibling_excluded" if broken_groups[record["group_id"]] else "sibling_rejected")

    # Deduplication ignores audit exclusions: otherwise excluding the kept copy of a duplicate would
    # promote its twin, and the next audit would see a different population (no fixed point).
    clean = [r for r in records if all(c.startswith("audit:") for c in failed[r["record_id"]])]
    duplicates = deduplicate(clean, config.quality)
    references = load_eval_references(
        [PROJECT_ROOT / p for p in config.raw["contamination"]["eval_case_dirs"]],
        [PROJECT_ROOT / p for p in config.raw["contamination"]["coding_suites"]],
    )
    contaminated = contamination(records, references, config.quality)

    for record in records:
        rid = record["record_id"]
        quality: dict[str, Any] = {"status": "accepted", "failed_checks": sorted(set(failed[rid])), "flags": sorted(set(flags[rid]))}
        if rid in duplicates:
            reason, of = duplicates[rid]
            quality["failed_checks"].append(reason)
            if of:
                quality["duplicate_of"] = of
        if rid in contaminated:
            quality["failed_checks"].append(contaminated[rid])
        if quality["failed_checks"]:
            quality["status"] = "rejected"
        record["quality"] = quality
        record["split"] = assign_split(record, config.split)

    apply_governance(config, records, sources, approvals)
    for record in records:
        errors = list(validator.iter_errors(record))
        if errors:
            raise ValueError(f"curated record {record['record_id']} violates the schema: {errors[0].message}")

    problems = check_split_invariants(records, config.split)
    if problems:
        raise ValueError("split invariants violated: " + "; ".join(problems[:5]))
    return records


def manifest_row(record: dict[str, Any], pairs: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Metadata-only manifest row. `supervision` binds the Git-ignored input, target and rendered
    training examples to this committed row (sentinel.garden.supervision)."""
    if pairs is None:
        pairs = supervision.pairs_by_id([record])
    row = {k: v for k, v in record.items() if k not in MANIFEST_DROP_FIELDS}
    target = record["target"]
    row["label"] = {
        "language": record["input"]["language"],
        "decision": target["decision"],
        "cwe": target["cwe"],
        "alternate_cwes": target["alternate_cwes"],
        "start_line": target["start_line"],
        "end_line": target["end_line"],
        "evidence_source": target["evidence_source"],
        "lines": len(record["input"]["code"].splitlines()),
        "patch_sha256": hashlib.sha256(target["patch"].encode("utf-8")).hexdigest() if target["patch"] else None,
    }
    row["supervision"] = supervision.supervision(record, pairs)
    return row


def statistics(config: GardenConfig, records: list[dict[str, Any]], build_log: list[dict[str, Any]], discovery_stats: dict[str, Any]) -> dict[str, Any]:
    accepted = [r for r in records if r["quality"]["status"] == "accepted"]

    def count(rows: list[dict[str, Any]], key: Any) -> dict[str, int]:
        return dict(sorted(Counter(key(r) for r in rows).items()))

    failed_checks: Counter[str] = Counter()
    for record in records:
        for check in record["quality"]["failed_checks"]:
            if check.startswith("eval_"):
                check = check.split(":")[0]
            elif check.startswith("audit:"):
                check = ":".join(check.split(":")[:2])
            failed_checks[check] += 1
    blockers: Counter[str] = Counter(b for r in accepted for b in r["governance"]["blockers"])
    accepted_findings = [r for r in accepted if r["target"]["decision"] == "finding"]
    repos_by_split: dict[str, set[str]] = defaultdict(set)
    for record in accepted:
        repos_by_split[record["split"]].add(record["provenance"]["repository_key"])
    return {
        "dataset_id": config.dataset_id,
        "pipeline_version": PIPELINE_VERSION,
        "syntax_versions": syntax.versions(),
        "config_sha256": config.sha256,
        "split_policy": {"policy_id": config.split.policy_id, "sha256": file_sha256(PROJECT_ROOT / config.raw["split"]["policy"])},
        "discovery": discovery_stats,
        "build": {
            "candidates": len(build_log),
            "outcomes": count(build_log, lambda r: r["outcome"]),
        },
        "records": {
            "total": len(records),
            "accepted": len(accepted),
            "rejected": len(records) - len(accepted),
            "failed_checks": dict(sorted(failed_checks.items())),
        },
        "accepted": {
            "by_kind": count(accepted, lambda r: r["kind"]),
            "by_decision": count(accepted, lambda r: r["target"]["decision"]),
            "by_language": count(accepted, lambda r: r["input"]["language"]),
            "by_language_and_decision": count(accepted, lambda r: f"{r['input']['language']}/{r['target']['decision']}"),
            "by_split": count(accepted, lambda r: r["split"]),
            "by_split_and_decision": count(accepted, lambda r: f"{r['split']}/{r['target']['decision']}"),
            "repositories_by_split": {k: len(v) for k, v in sorted(repos_by_split.items())},
            "finding_cwes": count(accepted_findings, lambda r: r["target"]["cwe"]),
            "finding_span_kind": count(accepted_findings, lambda r: r["localization"].get("span_kind") or r["localization"]["method"]),
            "complete_pairs": sum(
                1 for pair, kinds in _pairs(accepted).items() if kinds == {"vulnerable", "post_fix"}
            ),
            "license_detected": count(accepted, lambda r: r["provenance"]["license"]["detected"]),
            "license_category": count(accepted, lambda r: r["provenance"]["license"]["category"]),
            "flags": dict(sorted(Counter(f for r in accepted for f in r["quality"]["flags"]).items())),
            "fix_year": count(accepted, lambda r: r["provenance"]["fix_committed_at"][:4]),
        },
        "supervision": supervision_statistics(records),
        "governance": {
            "by_approval_status": count(records, lambda r: r["governance"]["approval_status"]),
            "accepted_blockers": dict(sorted(blockers.items())),
            "training_approved": sum(1 for r in records if r["governance"]["approval_status"] == "training_approved"),
        },
        **({"by_source": source_funnel(records, build_log)} if config.raw.get("query_tests") or (config.raw.get("discovery") or {}).get("morefixes") else {}),
    }


def source_funnel(records: list[dict[str, Any]], build_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Garden v6: the funnel of each label source, from built candidates to exported training examples."""
    from sentinel.garden.analysis import source_of

    by_advisory: dict[str, str] = {}
    for record in records:
        by_advisory.setdefault(record["provenance"]["advisory_id"], source_of(record))
    funnel: dict[str, Any] = {}
    sources = sorted({source_of(r) for r in records} | {"ghsa_osv"})
    for source in sources:
        rows = [r for r in records if source_of(r) == source]
        accepted = [r for r in rows if r["quality"]["status"] == "accepted"]
        approved = [r for r in accepted if r["governance"]["approval_status"] == "training_approved"]
        train = [r for r in approved if r["split"] == "train"]
        if source == "codeql_tool_test":
            log_rows = [row for row in build_log if row.get("test_unit")]
        else:
            log_rows = [row for row in build_log if not row.get("test_unit") and by_advisory.get(row["advisory_id"], "") == source]
        funnel[source] = {
            "candidates_with_records": len({row["advisory_id"] for row in log_rows if row["records"]}),
            "records": len(rows),
            "accepted": len(accepted),
            "accepted_by_decision": dict(sorted(Counter(r["target"]["decision"] for r in accepted).items())),
            "accepted_unique_vulnerabilities": len({r["provenance"]["advisory_id"] for r in accepted if r["target"]["decision"] == "finding"}),
            "rejected_top_checks": dict(Counter(c.split(":")[0] for r in rows if r["quality"]["status"] == "rejected" for c in r["quality"]["failed_checks"]).most_common(8)),
            "by_approval_status": dict(sorted(Counter(r["governance"]["approval_status"] for r in accepted).items())),
            "pending_blockers": dict(Counter(b.split(":")[0] for r in accepted if r["governance"]["approval_status"] == "pending_review" for b in r["governance"]["blockers"]).most_common(6)),
            "training_approved": len(approved),
            "train_detection_examples": len(train),
            "train_patch_examples": len(supervision.patch_examples(train)),
        }
    return funnel


def supervision_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    from sentinel.eval.patch import PATCH_PROMPT_VERSION
    from sentinel.eval.prompt import PROMPT_VERSION

    pairs = supervision.pairs_by_id(records)
    vulnerable = [r for r in records if r["kind"] == "vulnerable"]
    eligibility = {r["record_id"]: supervision.supervision(r, pairs)["patch_eligibility"] for r in vulnerable}
    eligible = [r for r in vulnerable if eligibility[r["record_id"]] == "eligible"]
    return {
        "version": supervision.SUPERVISION_VERSION,
        "detection": {"prompt_version": PROMPT_VERSION, "render_version": render.RENDER_VERSION, "confidence": "not_supervised"},
        "patch": {
            "prompt_version": PATCH_PROMPT_VERSION,
            "render_version": supervision.PATCH_RENDER_VERSION,
            "vulnerable_records_by_eligibility": dict(sorted(Counter(eligibility.values()).items())),
            "eligible_by_split": dict(sorted(Counter(r["split"] for r in eligible).items())),
        },
    }


def _pairs(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["pair_id"]:
            pairs[record["pair_id"]].add(record["kind"])
    return pairs


EXPORT_FILES = (
    "records.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "train.chat.jsonl",
    "train.patch.chat.jsonl",
    "staging.jsonl",
    "holdout_reserved.jsonl",
    "holdout_reserved.eval_cases.jsonl",
    "holdout_reserved.patch_cases.jsonl",
    "rejected.jsonl",
)


def write_exports(workdir: Path, records: list[dict[str, Any]]) -> None:
    """Deterministic JSONL exports. Only `training_approved` records reach train/validation files."""

    def select(predicate: Any) -> list[dict[str, Any]]:
        return [r for r in records if predicate(r)]

    approved = select(lambda r: r["governance"]["approval_status"] == "training_approved")
    for record in approved:  # defence in depth: re-check the gate's invariants at the write boundary
        if record["quality"]["status"] != "accepted" or record["split"] not in {"train", "validation"}:
            raise AssertionError(f"gate invariant violated for {record['record_id']}")
    _write_jsonl(workdir / "records.jsonl", records)
    _write_jsonl(workdir / "train.jsonl", [r for r in approved if r["split"] == "train"])
    _write_jsonl(workdir / "validation.jsonl", [r for r in approved if r["split"] == "validation"])
    approved_train = [r for r in approved if r["split"] == "train"]
    _write_jsonl(workdir / "train.chat.jsonl", [render.to_chat_example(r) for r in approved_train])
    # patch examples need both halves of a verified pair; both are approved together (same repository)
    _write_jsonl(workdir / "train.patch.chat.jsonl", supervision.patch_examples(approved_train))
    _write_jsonl(
        workdir / "staging.jsonl",
        select(lambda r: r["quality"]["status"] == "accepted" and r["governance"]["approval_status"] == "pending_review"),
    )
    roles = any("evaluation_role" in r for r in records)
    # with evaluation roles (v4) the holdout export holds the previously exported holdout only; the
    # untouched final test is written separately, so a routine holdout run cannot spend it
    holdout = select(
        lambda r: r["quality"]["status"] == "accepted" and r["split"] == "holdout_reserved" and (not roles or r["evaluation_role"] == "previous_holdout")
    )
    write_holdout_exports(workdir, holdout)
    _write_jsonl(workdir / "rejected.jsonl", select(lambda r: r["quality"]["status"] == "rejected"))
    names = EXPORT_FILES
    if roles:
        write_role_exports(workdir, records)
        names = EXPORT_FILES + ROLE_EXPORT_FILES
    sums = "".join(f"{file_sha256(workdir / name)}  {name}\n" for name in names)
    (workdir / "SHA256SUMS").write_text(sums, encoding="utf-8")


ROLE_EXPORT_FILES = (
    "development/dev.eval_cases.jsonl",
    "development/dev.patch_cases.jsonl",
    "development/manifest.json",
    "development/records.jsonl",
    "final_test.jsonl",
    "final_test.eval_cases.jsonl",
    "final_test.patch_cases.jsonl",
)
DEVELOPMENT_SET_VERSION = "2"  # 1: sentinel.eval.devset (Garden v3 approved validation); 2: Garden v4 development role


def write_holdout_exports(workdir: Path, holdout: list[dict[str, Any]]) -> None:
    _write_jsonl(workdir / "holdout_reserved.jsonl", holdout)
    holdout_cases = [render.to_eval_case(r) for r in holdout]
    validate_cases(holdout_cases)
    _write_jsonl(workdir / "holdout_reserved.eval_cases.jsonl", holdout_cases)
    _write_jsonl(workdir / "holdout_reserved.patch_cases.jsonl", supervision.patch_cases(holdout))


def development_manifest(dataset_id: str, records: list[dict[str, Any]], cases: list[dict[str, Any]], patch_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Same shape as `sentinel.eval.devset`, so `sentinel.eval.select_checkpoint --devset` accepts it."""
    return {
        "devset_version": DEVELOPMENT_SET_VERSION,
        "dataset_id": dataset_id,
        "role": "development (checkpoint selection); repository-disjoint from every earlier evaluation set; not the final test",
        "source": "accepted records of validation-split repositories outside the earlier development set (evaluation use; not training data)",
        "records": len(records),
        "repositories": len({r["provenance"]["repository_key"] for r in records}),
        "detection": {
            "cases_sha256": case_set_sha256(cases),
            "count": len(cases),
            "by_expected_decision": dict(sorted(Counter(c["expected_decision"] for c in cases).items())),
            "by_language": dict(sorted(Counter(c["language"] for c in cases).items())),
        },
        "patch": {"cases_sha256": case_set_sha256(patch_cases), "count": len(patch_cases)},
    }


def write_role_exports(workdir: Path, records: list[dict[str, Any]]) -> None:
    """Garden v4: the development set (devset format) and the sealed final test set."""
    accepted = [r for r in records if r["quality"]["status"] == "accepted"]
    development = [r for r in accepted if r["evaluation_role"] == "development"]
    final = [r for r in accepted if r["evaluation_role"] == "final_test"]
    cases = []
    for record in development:
        case = render.to_eval_case(record)
        case["source_kind"] = "development"
        cases.append(case)
    validate_cases(cases)
    patch_cases = supervision.patch_cases(development)
    directory = workdir / "development"
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "dev.eval_cases.jsonl", cases)
    _write_jsonl(directory / "dev.patch_cases.jsonl", patch_cases)
    _write_jsonl(directory / "records.jsonl", development)
    dataset_id = development[0]["dataset_id"] if development else (records[0]["dataset_id"] if records else "")
    _write_json(directory / "manifest.json", development_manifest(dataset_id, development, cases, patch_cases))
    _write_jsonl(workdir / "final_test.jsonl", final)
    final_cases = [render.to_eval_case(r) for r in final]
    validate_cases(final_cases)
    _write_jsonl(workdir / "final_test.eval_cases.jsonl", final_cases)
    _write_jsonl(workdir / "final_test.patch_cases.jsonl", supervision.patch_cases(final))


def external_resources(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything outside this repository that a reconstruction needs (no credentials required)."""
    repos: dict[str, dict[str, Any]] = {}
    for record in records:
        provenance = record["provenance"]
        entry = repos.setdefault(
            provenance["repository_key"], {"url": provenance["repository"] + ".git", "commits": set(), "records": 0}
        )
        entry["commits"].update({provenance["vulnerable_revision"], provenance["fixed_revision"]})
        entry["records"] += 1
    return {
        "description": "Upstream git repositories and immutable commits needed to rebuild every manifest record. "
        "Fetched anonymously over HTTPS with `git fetch --depth=2 --filter=blob:none origin <fix-sha>`; "
        "the vulnerable commit is the fix commit's first parent.",
        "repositories": [
            {"repository_key": key, "url": v["url"], "commits": sorted(v["commits"]), "records": v["records"]}
            for key, v in sorted(repos.items())
        ],
        "optional_services": {
            "OSV export (discover only; candidates are frozen)": "https://osv-vulnerabilities.storage.googleapis.com",
            "OSV API (audit)": "https://api.osv.dev/v1",
            "deps.dev API (review)": "https://api.deps.dev/v3",
        },
    }


def curate(config: GardenConfig, workdir: Path | None = None, log: Any = print) -> dict[str, Any]:
    workdir = workdir or config.workdir
    raw = _read_jsonl(workdir / "records.raw.jsonl")
    build_log = _read_jsonl(workdir / "build_log.jsonl")
    records = curate_records(config, raw)
    write_exports(workdir, records)

    committed = config.committed_dir
    discovery_stats = json.loads((committed / "discovery_stats.json").read_text(encoding="utf-8"))
    stats = statistics(config, records, build_log, discovery_stats)
    stats["candidates_sha256"] = file_sha256(committed / "candidates.jsonl")
    pairs = supervision.pairs_by_id(records)
    _write_jsonl(committed / "manifest.jsonl", [manifest_row(r, pairs) for r in records])
    stats["manifest_sha256"] = file_sha256(committed / "manifest.jsonl")
    _write_json(committed / "stats.json", stats)
    _write_json(committed / "EXTERNAL_RESOURCES.json", external_resources(records))
    (committed / "REPORT.md").write_text(report_markdown(stats), encoding="utf-8")
    log(
        f"curate: {stats['records']['accepted']}/{stats['records']['total']} records accepted, "
        f"{stats['governance']['training_approved']} training-approved -> {workdir}"
    )
    return stats


# --------------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------------


def _table(mapping: dict[str, Any], key_label: str, value_label: str = "Records") -> str:
    lines = [f"| {key_label} | {value_label} |", "| --- | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in mapping.items()]
    return "\n".join(lines)


def report_markdown(stats: dict[str, Any]) -> str:
    acc = stats["accepted"]
    langs = sorted({k.split("/")[0] for k in acc["by_language_and_decision"]})
    decisions = ["finding", "no_finding", "abstain"]
    grid = ["| Language | finding | no_finding | abstain | Total |", "| --- | ---: | ---: | ---: | ---: |"]
    for lang in langs:
        row = [acc["by_language_and_decision"].get(f"{lang}/{d}", 0) for d in decisions]
        grid.append(f"| {lang} | {' | '.join(map(str, row))} | {sum(row)} |")
    totals = [acc["by_decision"].get(d, 0) for d in decisions]
    grid.append(f"| **Total** | {' | '.join(map(str, totals))} | {sum(totals)} |")
    split_grid = ["| Split | finding | no_finding | abstain | Repositories |", "| --- | ---: | ---: | ---: | ---: |"]
    for split in ["train", "validation", "holdout_reserved", "temporal_quarantine"]:
        row = [acc["by_split_and_decision"].get(f"{split}/{d}", 0) for d in decisions]
        split_grid.append(f"| {split} | {' | '.join(map(str, row))} | {acc['repositories_by_split'].get(split, '-')} |")
    discovery = ["| Ecosystem | In export | Inspected | Candidates |", "| --- | ---: | ---: | ---: |"]
    for eco, s in stats["discovery"]["ecosystems"].items():
        discovery.append(f"| {eco} | {s['advisories_in_export']} | {s['inspected']} | {s['outcomes'].get('candidate', 0)} |")
    return f"""# Sentinel Garden - {stats['dataset_id']} report

Generated by `python -m sentinel.garden curate` (pipeline v{stats['pipeline_version']}, config sha256 `{stats['config_sha256'][:12]}`).
All numbers are computed from `manifest.jsonl`; nothing in this file is hand-edited.

**Approval status: {stats['governance']['training_approved']} training-approved records.** Every accepted record is
`pending_review` until its repository is approved in `data/garden/approvals.yaml` and the
registry sources it depends on are approved. Accepted records are a non-approved *staging* dataset.

## Accepted records

{chr(10).join(grid)}

Complete vulnerable/fixed pairs: {acc['complete_pairs']}.

## Splits (repository-level, temporal cutoff applied)

{chr(10).join(split_grid)}

## CWE distribution of accepted findings

{len(acc['finding_cwes'])} distinct primary CWEs. Records per CWE, most frequent first:

{_table(dict(sorted(acc['finding_cwes'].items(), key=lambda item: (-item[1], int(item[0][4:])))), 'CWE')}

## Localization of accepted findings

{_table(acc['finding_span_kind'], 'Span kind')}

## Discovery (OSV export, GitHub-reviewed advisories)

{chr(10).join(discovery)}

Discovery outcomes:

{_table({f"{eco}: {k}": v for eco, s in stats['discovery']['ecosystems'].items() for k, v in s['outcomes'].items()}, 'Ecosystem: outcome', 'Advisories')}

## Build (fix commit -> function-level records)

{_table(stats['build']['outcomes'], 'Outcome', 'Candidates')}

## Quality

Records: {stats['records']['total']} built, {stats['records']['accepted']} accepted, {stats['records']['rejected']} rejected.

{_table(stats['records']['failed_checks'] or {'(none)': 0}, 'Failed check')}

Flags on accepted records (not rejections):

{_table(acc['flags'] or {'(none)': 0}, 'Flag')}

## Licenses detected (unreviewed - detection is not approval)

{_table(acc['license_detected'], 'Detected license')}

## Governance

{_table(stats['governance']['by_approval_status'], 'Approval status')}

Blockers on accepted records:

{_table(stats['governance']['accepted_blockers'] or {'(none)': 0}, 'Blocker')}

## Fix commit year (accepted records)

{_table(acc['fix_year'], 'Year')}
"""


# --------------------------------------------------------------------------------------------
# review / audit
# --------------------------------------------------------------------------------------------


def review(config: GardenConfig, log: Any = print) -> dict[str, Any]:
    """Collect rights-review evidence for the committed manifest and write the review queue (network)."""
    committed = config.committed_dir
    rows = _read_jsonl(committed / "manifest.jsonl")
    candidates = _read_jsonl(committed / "candidates.jsonl")
    evidence = collect_evidence(rows, candidates, config.workdir / "cache" / "git", DepsDev(config.workdir / "cache" / "depsdev"))
    summary = write_review_outputs(evidence, committed, file_sha256(committed / "manifest.jsonl"))
    log(f"review: {summary['repositories']} repositories, tiers {summary['by_tier']} -> {committed / 'REVIEW_QUEUE.md'}")
    return summary


def review_packages(config: GardenConfig, log: Any = print) -> list[dict[str, Any]]:
    """Blind evidence packages for an independent human review (offline: records + OSV cache).

    Packages contain upstream code and stay in the Git-ignored work directory; the committed
    audit/human_review_sample.jsonl lists the items and the sha256 of each package.
    """
    committed = config.committed_dir
    audit_dir = committed / "audit"
    automated = _read_jsonl(audit_dir / "automated.jsonl") if (audit_dir / "automated.jsonl").is_file() else []
    # Sample what will be trained on: the training-approved records once approvals exist, otherwise the
    # proposed training cohort (scripts/training_cohort.py), otherwise every exportable record.
    records = _read_jsonl(config.workdir / "records.jsonl")
    cohort = {r["provenance"]["repository_key"] for r in records if r["governance"]["approval_status"] == "training_approved"} or None
    cohort_path = committed / "cohort" / "training_cohort.json"
    if cohort is None and cohort_path.is_file():
        document = json.loads(cohort_path.read_text(encoding="utf-8"))
        cohort = {r["repository_key"] for r in document["repositories"] if r["proposed_decision"] == "recommended_for_review"}
    index = human_review.write_packages(
        records,
        config.workdir / "cache" / "osv",
        automated,
        load_manual_review(audit_dir / "manual_review.yaml"),
        config.workdir / "review_packages",
        audit_dir,
        cohort,
    )
    log(f"review-packages: {len(index)} items -> {config.workdir / 'review_packages'} (index: {audit_dir / 'human_review_sample.jsonl'})")
    return index


def audit(config: GardenConfig, log: Any = print) -> dict[str, Any]:
    """Automated quality audit of the curated records (network: OSV API, upstream git)."""
    committed = config.committed_dir
    records = _read_jsonl(config.workdir / "records.jsonl")
    candidates = _read_jsonl(committed / "candidates.jsonl")
    repository_vulns, repository_packages = None, None
    if (config.raw.get("audit") or {}).get("cross_source_later_fixes"):
        # Garden v6: a post-fix or defended-sibling negative is refuted by a later fix of ANY known advisory
        # of its repository (all packages, and every CVE MoreFixes links to a fix commit there), not only
        # by advisories of the record's own packages
        from sentinel.garden import morefixes

        index = morefixes.load_index(morefixes_index_dir(config))
        cache: dict[str, list[dict[str, Any]]] = {}

        def repository_vulns(key: str) -> list[dict[str, Any]]:
            if key not in cache:
                cache[key] = morefixes.repository_vulns(index, key)
            return cache[key]

        repository_packages = defaultdict(set)
        for c in candidates:
            for package in c["packages"]:
                repository_packages[c["repository_key"]].add((c["ecosystem"], package))
    result = run_audit(
        records,
        candidates,
        config.workdir / "cache" / "git",
        config.workdir / "cache" / "osv",
        OsvQuery(config.workdir / "cache" / "osv_query"),
        containment=config.quality.cross_split_containment,
        repository_vulns=repository_vulns,
        repository_packages=repository_packages,
    )
    audit_dir = committed / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(audit_dir / "automated.jsonl", result["groups"])
    _write_jsonl(audit_dir / "sample.jsonl", result["sample"])
    exclusions_path = audit_dir / "exclusions.yaml"
    manual = load_manual_review(audit_dir / "manual_review.yaml")
    # An incomplete later-fix check (OSV or an upstream commit unavailable) must not clear the
    # hard-negative exclusions an earlier, complete run established for the same record.
    if exclusions_path.is_file():
        previous = yaml.safe_load(exclusions_path.read_text(encoding="utf-8")) or {}
        incomplete = {g["records"]["post_fix"] for g in result["groups"] if g.get("later_fix_check") == "incomplete" and "post_fix" in g["records"]}
        incomplete |= {  # Garden v5: a defended sibling's later-fix check is a hard-negative check too
            g["records"]["defended_sibling"] for g in result["groups"] if g.get("sibling_later_fix_check") == "incomplete" and "defended_sibling" in g["records"]
        }
        present = {(e["record_id"], e["reason"]) for e in result["exclusions"]}
        carried = [
            e for e in previous.get("automated") or []
            if e["record_id"] in incomplete
            and e["reason"].split(":")[0] in {"later_fix_changes_same_function", "reported_incomplete_fix", "later_fix_unverifiable"}
            and (e["record_id"], e["reason"]) not in present
        ]
        if carried:
            log(f"audit: {len(carried)} later-fix exclusions kept from the previous run (incomplete recheck)")
            result["exclusions"] = sorted(result["exclusions"] + carried, key=lambda e: (e["record_id"], e["reason"]))
    index_path = audit_dir / "human_review_sample.jsonl"
    index = _read_jsonl(index_path) if index_path.is_file() else []
    human = human_review.load_human_review(audit_dir / "human_review.yaml", index)
    document = {
        "audit_version": result["audit_version"],
        "automated": result["exclusions"],
        "manual": manual_exclusions(manual),
        "human": human_review.human_exclusions(human),
    }
    exclusions_path.write_text(
        "# Generated by `python -m sentinel.garden audit` (automated section), the AI-assisted manual review (manual section)\n"
        "# and the independent human review (human section, from audit/human_review.yaml written by a person).\n"
        "# Applied by `curate` as failed checks `audit:<reason>`. Records are only ever excluded, never relabelled.\n"
        + yaml.safe_dump(document, sort_keys=False, width=200),
        encoding="utf-8",
    )
    summary = summarize(result, manual)
    summary["human_review"] = human_review.summarize_human(human, index)
    _write_json(audit_dir / "summary.json", summary)
    (committed / "AUDIT_REPORT.md").write_text(audit_report_markdown(summary, result, manual), encoding="utf-8")
    log(f"audit: {summary['groups']} groups, {len(result['exclusions'])} automated exclusions -> {audit_dir}")
    return summary


# --------------------------------------------------------------------------------------------
# verify / check
# --------------------------------------------------------------------------------------------


def verify(config: GardenConfig, log: Any = print) -> list[str]:
    """Rebuild every record from the frozen candidates and compare with the committed manifest."""
    manifest = {row["record_id"]: row for row in _read_jsonl(config.committed_dir / "manifest.jsonl")}
    with tempfile.TemporaryDirectory(prefix="garden-verify-") as tmp:
        raw, _ = build(config, workdir=Path(tmp), log=log)
    rebuilt = {r["record_id"]: r for r in raw}
    problems = []
    for record_id in sorted(set(manifest) | set(rebuilt)):
        if record_id not in rebuilt:
            problems.append(f"{record_id}: in manifest but not rebuilt")
        elif record_id not in manifest:
            problems.append(f"{record_id}: rebuilt but not in manifest")
        elif rebuilt[record_id]["provenance"]["content_sha256"] != manifest[record_id]["provenance"]["content_sha256"]:
            problems.append(f"{record_id}: content hash differs")
    log(f"verify: {len(rebuilt)} rebuilt, {len(manifest)} in manifest, {len(problems)} problems")
    return problems


def _reachable(url: str) -> bool:
    import subprocess

    try:
        done = subprocess.run(
            ["git", "ls-remote", "--exit-code", url, "HEAD"],
            capture_output=True,
            timeout=60,
            env={**__import__("os").environ, "GIT_TERMINAL_PROMPT": "0"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def reconstruct(config: GardenConfig, workers: int = 6, log: Any = print) -> dict[str, Any]:
    """Rebuild the full records from pinned upstream revisions after a fresh checkout.

    Committed metadata is never modified. Records whose upstream is unavailable are reported as
    missing and are never substituted; any content-hash mismatch fails the reconstruction.
    """
    committed = config.committed_dir
    manifest = {row["record_id"]: row for row in _read_jsonl(committed / "manifest.jsonl")}
    resources = json.loads((committed / "EXTERNAL_RESOURCES.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reachable = dict(zip(
            [r["repository_key"] for r in resources["repositories"]],
            pool.map(lambda r: _reachable(r["url"]), resources["repositories"]),
        ))
    unreachable = sorted(k for k, ok in reachable.items() if not ok)
    log(f"reconstruct: {len(reachable) - len(unreachable)}/{len(reachable)} upstream repositories reachable")
    raw, build_log = build(config, workers=workers, log=log)
    rebuilt = {r["record_id"]: r for r in raw}
    missing = sorted(set(manifest) - set(rebuilt))
    unexpected = sorted(set(rebuilt) - set(manifest))
    mismatched = sorted(
        rid for rid in set(manifest) & set(rebuilt)
        if rebuilt[rid]["provenance"]["content_sha256"] != manifest[rid]["provenance"]["content_sha256"]
    )
    records = curate_records(config, raw)
    curated = {r["record_id"]: r for r in records}
    # The split depends only on the repository, the fix date and the frozen policy, so it never
    # legitimately changes; acceptance can (a kept duplicate whose twin is missing upstream).
    split_drift = sorted(rid for rid in curated if rid in manifest and curated[rid]["split"] != manifest[rid]["split"])
    status_drift = sorted(
        rid for rid in curated if rid in manifest and curated[rid]["quality"]["status"] != manifest[rid]["quality"]["status"]
    )
    write_exports(config.workdir, records)
    report = {
        "manifest_records": len(manifest),
        "rebuilt_records": len(rebuilt),
        "missing_records": missing,
        "missing_because_unreachable": sorted({manifest[r]["provenance"]["repository_key"] for r in missing} & set(unreachable)),
        "unexpected_records": unexpected,
        "content_hash_mismatches": mismatched,
        "split_drift": split_drift,
        "status_drift": status_drift,
        "unreachable_repositories": unreachable,
        "exports": (config.workdir / "SHA256SUMS").read_text(encoding="utf-8").splitlines(),
        # Missing upstream data can legitimately shift deduplication (a kept duplicate disappears);
        # without missing records any acceptance drift means the rebuild is not deterministic. Split
        # drift is never acceptable.
        "ok": not mismatched and not unexpected and not split_drift and (not status_drift or bool(missing)),
    }
    _write_json(config.workdir / "reconstruction_report.json", report)
    log(
        f"reconstruct: {len(rebuilt)}/{len(manifest)} records rebuilt, {len(missing)} missing, "
        f"{len(mismatched)} hash mismatches -> {config.workdir} (verify with `sha256sum -c SHA256SUMS`)"
    )
    return report


TRAINING_EXPORT_FILES = ("train.jsonl", "validation.jsonl", "train.chat.jsonl", "train.patch.chat.jsonl")


def _rebuild_verified(config: GardenConfig, wanted: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], workers: int, log: Any) -> list[dict[str, Any]]:
    """Rebuild the `wanted` manifest records from upstream and verify them against the manifest
    (content hash and every supervision hash); quality, split, governance and evaluation role come
    from the manifest. Any difference is an error; nothing is repaired."""
    advisories = {row["provenance"]["advisory_id"] for row in wanted.values()}
    candidates = [c for c in _read_jsonl(config.committed_dir / "candidates.jsonl") if c["advisory_id"] in advisories]
    log(f"materialize: {len(wanted)} records from {len(candidates)} advisories in {len({c['repository_key'] for c in candidates})} repositories")
    built, _ = build_candidates(config, candidates, workers, log)
    if config.raw.get("query_tests") and any(row["provenance"].get("source_kind") == "tool_test" for row in wanted.values()):
        built += build_query_tests(config, log)[0]  # Garden v6: tool tests come from the pinned CodeQL checkout
    by_id = {r["record_id"]: r for r in built}
    missing = sorted(set(wanted) - set(by_id))
    if missing:
        raise ValueError(f"{len(missing)} records could not be rebuilt from upstream (first: {missing[:3]})")
    # Every rebuilt record takes quality, split and governance from the manifest (siblings included,
    # so patch eligibility sees the same pair states curate saw).
    known = []
    for rid, raw in by_id.items():
        if rid not in manifest:
            raise ValueError(f"{rid}: rebuilt but not in the committed manifest (code or config differs)")
        row = manifest[rid]
        if raw["provenance"]["content_sha256"] != row["provenance"]["content_sha256"]:
            raise ValueError(f"{rid}: rebuilt code differs from the manifest content hash")
        fields = {"quality": row["quality"], "split": row["split"], "governance": row["governance"]}
        if "evaluation_role" in row:
            fields["evaluation_role"] = row["evaluation_role"]
        known.append({**raw, **fields})
    pairs = supervision.pairs_by_id(known)
    records = []
    for record in sorted(known, key=lambda r: r["record_id"]):
        if record["record_id"] not in wanted:
            continue
        bound = supervision.supervision(record, pairs)
        for key in supervision.HASH_FIELDS:
            if bound[key] != manifest[record["record_id"]]["supervision"][key]:
                raise ValueError(f"{record['record_id']}: rebuilt {key} differs from the manifest")
        records.append(record)
    return records


def materialize(config: GardenConfig, workers: int = 6, log: Any = print) -> dict[str, Any]:
    """Rebuild ONLY the training-approved records of the committed manifest and write the training exports.

    For a training machine after a fresh checkout: it fetches the commits of the approved
    repositories' approved advisories only (not every candidate of the dataset), checks every
    rebuilt record against the manifest (content hash and all supervision hashes) and against the
    governance gate with today's registry and approvals, and writes exactly the files
    `sentinel.train` reads plus SHA256SUMS. Quality, split and approval status are taken from the
    committed manifest, never recomputed. It does not touch records.raw.jsonl, so a later `curate`
    in the same work directory is unaffected. Any difference is an error; nothing is repaired.
    """
    committed = config.committed_dir
    manifest = {row["record_id"]: row for row in _read_jsonl(committed / "manifest.jsonl")}
    approved = {rid: row for rid, row in manifest.items() if row["governance"]["approval_status"] == "training_approved"}
    if not approved:
        raise ValueError("the committed manifest has no training-approved record (approvals and curate first)")
    registry = load_registry()
    validate_registry(registry)
    sources = {s["source_id"]: s for s in registry["sources"]}
    approvals = load_approvals()
    for rid, row in approved.items():
        status, blockers = gate(row, sources, approvals)
        if status != "training_approved":
            raise ValueError(f"{rid} is training_approved in the manifest but fails the gate now: {blockers} (rerun curate)")
    records = _rebuild_verified(config, approved, manifest, workers, log)
    candidates = {r["provenance"]["advisory_id"] for r in records}
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    workdir = config.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(workdir / "train.jsonl", train)
    _write_jsonl(workdir / "validation.jsonl", validation)
    _write_jsonl(workdir / "train.chat.jsonl", [render.to_chat_example(r) for r in train])
    _write_jsonl(workdir / "train.patch.chat.jsonl", supervision.patch_examples(train))
    sums = "".join(f"{file_sha256(workdir / name)}  {name}\n" for name in TRAINING_EXPORT_FILES)
    (workdir / "SHA256SUMS").write_text(sums, encoding="utf-8")
    report = {
        "manifest_sha256": file_sha256(committed / "manifest.jsonl"),
        "approved_records": len(approved),
        "train": len(train),
        "validation": len(validation),
        "repositories": len({r["provenance"]["repository_key"] for r in records}),
        "advisories_built": len(candidates),
        "exports": sums.splitlines(),
    }
    _write_json(workdir / "materialize_report.json", report)
    log(f"materialize: train {len(train)}, validation {len(validation)} -> {workdir} (verify with `sha256sum -c SHA256SUMS`)")
    return report


EVALUATION_SUMS = "SHA256SUMS.evaluation"


def materialize_evaluation(config: GardenConfig, workers: int = 6, log: Any = print) -> dict[str, Any]:
    """Garden v4, evaluation machine: rebuild only the evaluation-role records (development, final test,
    previously exported holdout) of the committed manifest, verify them like `materialize`, and write
    the development set (`development/`, devset format), `final_test.*` and `holdout_reserved.*` with
    their own checksum file. Training exports are not touched."""
    committed = config.committed_dir
    manifest = {row["record_id"]: row for row in _read_jsonl(committed / "manifest.jsonl")}
    if not any("evaluation_role" in row for row in manifest.values()):
        raise ValueError("this dataset version has no evaluation roles (Garden v4 and later)")
    wanted = {
        rid: row for rid, row in manifest.items()
        if row["quality"]["status"] == "accepted" and row["evaluation_role"] in {"development", "final_test", "previous_holdout"}
    }
    records = _rebuild_verified(config, wanted, manifest, workers, log)
    workdir = config.workdir
    write_role_exports(workdir, records)
    write_holdout_exports(workdir, [r for r in records if r["evaluation_role"] == "previous_holdout"])
    names = ROLE_EXPORT_FILES + ("holdout_reserved.jsonl", "holdout_reserved.eval_cases.jsonl", "holdout_reserved.patch_cases.jsonl")
    sums = "".join(f"{file_sha256(workdir / name)}  {name}\n" for name in names)
    (workdir / EVALUATION_SUMS).write_text(sums, encoding="utf-8")
    counts = Counter(r["evaluation_role"] for r in records)
    log(f"materialize-eval: {dict(sorted(counts.items()))} -> {workdir} (verify with `sha256sum -c {EVALUATION_SUMS}`)")
    return {"records": dict(sorted(counts.items())), "exports": sums.splitlines()}


def check(config: GardenConfig) -> dict[str, Any]:
    """Offline consistency check of the committed dataset metadata (used by CI)."""
    committed = config.committed_dir
    stats = json.loads((committed / "stats.json").read_text(encoding="utf-8"))
    if stats["manifest_sha256"] != file_sha256(committed / "manifest.jsonl"):
        raise ValueError("manifest.jsonl does not match stats.json (regenerate with `curate`)")
    if stats["candidates_sha256"] != file_sha256(committed / "candidates.jsonl"):
        raise ValueError("candidates.jsonl does not match stats.json (regenerate with `build` + `curate`)")
    if stats["config_sha256"] != config.sha256:
        raise ValueError("garden config changed since the manifest was generated (rerun the pipeline)")
    if stats.get("syntax_versions") != syntax.versions():
        raise ValueError(
            f"parser/grammar versions {syntax.versions()} differ from the ones the manifest was built with "
            f"{stats.get('syntax_versions')} (install the versions pinned in pyproject.toml)"
        )
    policy_path = PROJECT_ROOT / config.raw["split"]["policy"]
    if stats["split_policy"] != {"policy_id": config.split.policy_id, "sha256": file_sha256(policy_path)}:
        raise ValueError("split policy changed since the manifest was generated")
    rows = _read_jsonl(committed / "manifest.jsonl")
    if any("input" in row or "target" in row for row in rows):
        raise ValueError("manifest must not contain code, context or targets")
    for row in rows:
        problem = supervision.manifest_supervision_problem(row)
        if problem:
            raise ValueError(f"{row['record_id']}: {problem} (rerun curate)")
    registry = load_registry()
    sources = {s["source_id"]: s for s in registry["sources"]}
    approvals = load_approvals()
    for row in rows:
        status, _ = gate(row, sources, approvals)
        recorded = row["governance"]["approval_status"]
        if recorded == "training_approved" and status != "training_approved":
            raise ValueError(f"{row['record_id']} is marked training_approved but fails the gate now")
    if "governance" in config.raw:
        for row in rows:
            if row["kind"] in unvalidated_negative_kinds(config) and row["governance"]["approval_status"] == "training_approved":
                raise ValueError(f"{row['record_id']}: unvalidated negative kind is training-approved (rerun curate)")
    covered = source_approved_advisories(config)
    if covered is not None:
        for row in rows:
            if row["provenance"]["advisory_id"] not in covered and "source_approval_not_extended" not in row["governance"]["blockers"]:
                raise ValueError(f"{row['record_id']}: candidate outside the approved list without its blocker (rerun curate)")
    history = evaluation_history(config)
    for row in rows:
        if history is None:
            if "evaluation_role" in row:
                raise ValueError(f"{row['record_id']}: evaluation role without an `evaluation` section in the config")
            continue
        if row.get("evaluation_role") != evaluation_role(row, *history):
            raise ValueError(f"{row['record_id']}: evaluation role differs from the one the split and history give (rerun curate)")
        if row["evaluation_role"] in UNTOUCHED_ROLES and row["governance"]["approval_status"] == "training_approved":
            raise ValueError(f"{row['record_id']}: {row['evaluation_role']} record must never be training-approved")
    split_rows = [r for r in rows if r["quality"]["status"] == "accepted"]
    problems = check_split_invariants(split_rows, config.split)
    if problems:
        raise ValueError("split invariants violated: " + "; ".join(problems[:5]))
    accepted = sum(1 for r in rows if r["quality"]["status"] == "accepted")
    if accepted != stats["records"]["accepted"] or len(rows) != stats["records"]["total"]:
        raise ValueError("stats.json record counts do not match manifest.jsonl")
    approved = sum(1 for r in rows if r["governance"]["approval_status"] == "training_approved")
    by_id = {r["record_id"]: r for r in rows}
    for record_id, reasons in load_exclusions(committed / "audit" / "exclusions.yaml").items():
        row = by_id.get(record_id)
        if row is None:
            raise ValueError(f"audit exclusion refers to unknown record {record_id}")
        if row["quality"]["status"] != "rejected" or not any(c.startswith("audit:") for c in row["quality"]["failed_checks"]):
            raise ValueError(f"audit exclusion not applied to {record_id} (rerun curate)")
    resources_path = committed / "EXTERNAL_RESOURCES.json"
    if json.loads(resources_path.read_text(encoding="utf-8")) != json.loads(json.dumps(external_resources(rows))):
        raise ValueError("EXTERNAL_RESOURCES.json does not match manifest.jsonl (rerun curate)")
    index_path = committed / "audit" / "human_review_sample.jsonl"
    if index_path.is_file():
        human_review.load_human_review(committed / "audit" / "human_review.yaml", _read_jsonl(index_path))
    review_summary = committed / "review" / "summary.json"
    if review_summary.is_file():
        if json.loads(review_summary.read_text(encoding="utf-8"))["manifest_sha256"] != stats["manifest_sha256"]:
            raise ValueError("review queue is stale: rerun `python -m sentinel.garden review`")
    return {"records": len(rows), "accepted": accepted, "training_approved": approved}
