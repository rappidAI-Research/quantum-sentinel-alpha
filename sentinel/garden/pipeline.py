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
from sentinel.eval.io import validate_cases
from sentinel.garden import build as build_module
from sentinel.garden import render
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
PIPELINE_VERSION = "1"
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


def discover(config: GardenConfig, refresh: bool = False, log: Any = print) -> dict[str, Any]:
    d, sel = config.raw["discovery"], config.raw["selection"]
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
            )
            if candidate is None:
                reasons[reason] += 1
                continue
            commit = (candidate["repository_key"], candidate["fix_commit"])
            if commit in seen_commits:
                reasons["duplicate_fix_commit"] += 1
                continue
            if per_repo[candidate["repository_key"]] >= d["max_candidates_per_repository"]:
                reasons["repository_cap"] += 1
                continue
            if accepted >= d["max_candidates_per_ecosystem"]:
                reasons["ecosystem_cap"] += 1
                continue
            seen_commits.add(commit)
            per_repo[candidate["repository_key"]] += 1
            accepted += 1
            candidates.append(candidate)
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


def build(config: GardenConfig, workdir: Path | None = None, workers: int = 6, log: Any = print) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workdir = workdir or config.workdir
    candidates = _read_jsonl(config.committed_dir / "candidates.jsonl")
    cache = config.workdir / "cache" / "git"
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_repo[candidate["repository_key"]].append(candidate)

    def process(repo_key: str) -> list[tuple[dict[str, Any], list[dict[str, Any]], str]]:
        results = []
        group = by_repo[repo_key]
        repo = UpstreamRepo(cache, group[0]["repository"], repo_key)
        for candidate in group:
            try:
                records = build_module.build_records(
                    candidate, repo, config.extraction, config.dataset_id, config.code_source_id, config.metadata_source_ids
                )
                results.append((candidate, records, "built"))
            except Rejected as exc:
                results.append((candidate, [], exc.reason))
            except Exception as exc:  # noqa: BLE001 - one broken upstream must not stop the run
                results.append((candidate, [], f"error:{type(exc).__name__}"))
        return results

    records: list[dict[str, Any]] = []
    build_log: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, results in enumerate(pool.map(process, sorted(by_repo)), start=1):
            for candidate, built, outcome in results:
                records.extend(built)
                build_log.append(
                    {"advisory_id": candidate["advisory_id"], "repository_key": candidate["repository_key"], "outcome": outcome, "records": len(built)}
                )
            if done % 50 == 0:
                log(f"build: {done}/{len(by_repo)} repositories")
    records.sort(key=lambda r: r["record_id"])
    build_log.sort(key=lambda row: row["advisory_id"])
    _write_jsonl(workdir / "records.raw.jsonl", records)
    _write_jsonl(workdir / "build_log.jsonl", build_log)
    log(f"build: {len(records)} raw records from {sum(1 for r in build_log if r['outcome'] == 'built')}/{len(candidates)} candidates")
    return records, build_log


# --------------------------------------------------------------------------------------------
# Stage 4-6: curate (quality, dedup, contamination, split, governance, export)
# --------------------------------------------------------------------------------------------


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

    clean = [r for r in records if not failed[r["record_id"]]]
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
        status, blockers = gate(record, sources, approvals)
        record["governance"] = {"approval_status": status, "blockers": blockers}
        errors = list(validator.iter_errors(record))
        if errors:
            raise ValueError(f"curated record {rid} violates the schema: {errors[0].message}")

    problems = check_split_invariants(records, config.split)
    if problems:
        raise ValueError("split invariants violated: " + "; ".join(problems[:5]))
    return records


def manifest_row(record: dict[str, Any]) -> dict[str, Any]:
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
    return row


def statistics(config: GardenConfig, records: list[dict[str, Any]], build_log: list[dict[str, Any]], discovery_stats: dict[str, Any]) -> dict[str, Any]:
    accepted = [r for r in records if r["quality"]["status"] == "accepted"]

    def count(rows: list[dict[str, Any]], key: Any) -> dict[str, int]:
        return dict(sorted(Counter(key(r) for r in rows).items()))

    failed_checks: Counter[str] = Counter()
    for record in records:
        for check in record["quality"]["failed_checks"]:
            failed_checks[check.split(":")[0] if check.startswith("eval_") else check] += 1
    blockers: Counter[str] = Counter(b for r in accepted for b in r["governance"]["blockers"])
    accepted_findings = [r for r in accepted if r["target"]["decision"] == "finding"]
    repos_by_split: dict[str, set[str]] = defaultdict(set)
    for record in accepted:
        repos_by_split[record["split"]].add(record["provenance"]["repository_key"])
    return {
        "dataset_id": config.dataset_id,
        "pipeline_version": PIPELINE_VERSION,
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
            "finding_span_kind": count(accepted_findings, lambda r: r["localization"]["span_kind"]),
            "complete_pairs": sum(
                1 for pair, kinds in _pairs(accepted).items() if kinds == {"vulnerable", "post_fix"}
            ),
            "license_detected": count(accepted, lambda r: r["provenance"]["license"]["detected"]),
            "license_category": count(accepted, lambda r: r["provenance"]["license"]["category"]),
            "flags": dict(sorted(Counter(f for r in accepted for f in r["quality"]["flags"]).items())),
            "fix_year": count(accepted, lambda r: r["provenance"]["fix_committed_at"][:4]),
        },
        "governance": {
            "by_approval_status": count(records, lambda r: r["governance"]["approval_status"]),
            "accepted_blockers": dict(sorted(blockers.items())),
            "training_approved": sum(1 for r in records if r["governance"]["approval_status"] == "training_approved"),
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
    "staging.jsonl",
    "holdout_reserved.jsonl",
    "holdout_reserved.eval_cases.jsonl",
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
    _write_jsonl(workdir / "train.chat.jsonl", [render.to_chat_example(r) for r in approved if r["split"] == "train"])
    _write_jsonl(
        workdir / "staging.jsonl",
        select(lambda r: r["quality"]["status"] == "accepted" and r["governance"]["approval_status"] == "pending_review"),
    )
    holdout = select(lambda r: r["quality"]["status"] == "accepted" and r["split"] == "holdout_reserved")
    _write_jsonl(workdir / "holdout_reserved.jsonl", holdout)
    holdout_cases = [render.to_eval_case(r) for r in holdout]
    validate_cases(holdout_cases)
    _write_jsonl(workdir / "holdout_reserved.eval_cases.jsonl", holdout_cases)
    _write_jsonl(workdir / "rejected.jsonl", select(lambda r: r["quality"]["status"] == "rejected"))
    sums = "".join(f"{file_sha256(workdir / name)}  {name}\n" for name in EXPORT_FILES)
    (workdir / "SHA256SUMS").write_text(sums, encoding="utf-8")


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
    _write_jsonl(committed / "manifest.jsonl", [manifest_row(r) for r in records])
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


def audit(config: GardenConfig, log: Any = print) -> dict[str, Any]:
    """Automated quality audit of the curated records (network: OSV API, upstream git)."""
    committed = config.committed_dir
    records = _read_jsonl(config.workdir / "records.jsonl")
    candidates = _read_jsonl(committed / "candidates.jsonl")
    result = run_audit(
        records, candidates, config.workdir / "cache" / "git", config.workdir / "cache" / "osv", OsvQuery(config.workdir / "cache" / "osv_query")
    )
    audit_dir = committed / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(audit_dir / "automated.jsonl", result["groups"])
    _write_jsonl(audit_dir / "sample.jsonl", result["sample"])
    exclusions_path = audit_dir / "exclusions.yaml"
    manual = load_manual_review(audit_dir / "manual_review.yaml")
    document = {
        "audit_version": result["audit_version"],
        "automated": result["exclusions"],
        "manual": manual_exclusions(manual),
    }
    exclusions_path.write_text(
        "# Generated by `python -m sentinel.garden audit` (automated section) and the manual review (manual section).\n"
        "# Applied by `curate` as failed checks `audit:<reason>`. Records are only ever excluded, never relabelled.\n"
        + yaml.safe_dump(document, sort_keys=False, width=200),
        encoding="utf-8",
    )
    summary = summarize(result, manual)
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
    status_drift = sorted(
        rid for rid in curated if rid in manifest
        and (curated[rid]["quality"]["status"], curated[rid]["split"]) != (manifest[rid]["quality"]["status"], manifest[rid]["split"])
    )
    write_exports(config.workdir, records)
    report = {
        "manifest_records": len(manifest),
        "rebuilt_records": len(rebuilt),
        "missing_records": missing,
        "missing_because_unreachable": sorted({manifest[r]["provenance"]["repository_key"] for r in missing} & set(unreachable)),
        "unexpected_records": unexpected,
        "content_hash_mismatches": mismatched,
        "status_or_split_drift": status_drift,
        "unreachable_repositories": unreachable,
        "exports": (config.workdir / "SHA256SUMS").read_text(encoding="utf-8").splitlines(),
        "ok": not mismatched and not unexpected and not status_drift,
    }
    _write_json(config.workdir / "reconstruction_report.json", report)
    log(
        f"reconstruct: {len(rebuilt)}/{len(manifest)} records rebuilt, {len(missing)} missing, "
        f"{len(mismatched)} hash mismatches -> {config.workdir} (verify with `sha256sum -c SHA256SUMS`)"
    )
    return report


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
    policy_path = PROJECT_ROOT / config.raw["split"]["policy"]
    if stats["split_policy"] != {"policy_id": config.split.policy_id, "sha256": file_sha256(policy_path)}:
        raise ValueError("split policy changed since the manifest was generated")
    rows = _read_jsonl(committed / "manifest.jsonl")
    if any("input" in row or "target" in row for row in rows):
        raise ValueError("manifest must not contain code, context or targets")
    registry = load_registry()
    sources = {s["source_id"]: s for s in registry["sources"]}
    approvals = load_approvals()
    for row in rows:
        status, _ = gate(row, sources, approvals)
        recorded = row["governance"]["approval_status"]
        if recorded == "training_approved" and status != "training_approved":
            raise ValueError(f"{row['record_id']} is marked training_approved but fails the gate now")
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
    review_summary = committed / "review" / "summary.json"
    if review_summary.is_file():
        if json.loads(review_summary.read_text(encoding="utf-8"))["manifest_sha256"] != stats["manifest_sha256"]:
            raise ValueError("review queue is stale: rerun `python -m sentinel.garden review`")
    return {"records": len(rows), "accepted": accepted, "training_approved": approved}
