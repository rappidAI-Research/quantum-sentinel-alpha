"""Stage 5 - quality checks, deduplication, label-conflict and evaluation-contamination checks.

Record-level checks never repair a record: a failed check rejects it and is reported by name.
Dataset-level checks compare records with each other (exact and near duplicates, contradictory
labels for near-identical code) and with every evaluation case and coding task in the repository.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from sentinel.garden import code

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "garden_record.schema.json"

CONTAINMENT_THRESHOLD = 0.8
CONTAINMENT_MIN_SHINGLES = 20

KIND_DECISION = {"vulnerable": "finding", "post_fix": "no_finding", "caller_abstention": "abstain"}

# High-confidence secret patterns only; a hit rejects the record (it is never redacted in place).
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "stripe_live_key": re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b"),
}


@dataclass(frozen=True)
class QualitySettings:
    near_duplicate_jaccard: float
    label_conflict_jaccard: float
    contamination_jaccard: float
    shingle_size: int


def load_record_schema() -> dict[str, Any]:
    schema: dict[str, Any] = json.loads(RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


# --------------------------------------------------------------------------------------------
# Patch application (used to verify that the remediation patch turns vulnerable into fixed code)
# --------------------------------------------------------------------------------------------

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_patch(original: str, patch: str) -> str | None:
    """Strictly apply a unified diff; return None if any context or removed line does not match."""
    source = original.splitlines()
    result: list[str] = []
    position = 0  # index into source
    lines = patch.splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    while i < len(lines):
        match = _HUNK_HEADER.match(lines[i])
        if not match:
            return None
        old_start = int(match.group(1))
        old_count = 1 if match.group(2) is None else int(match.group(2))
        start_index = old_start - 1 if old_count else old_start
        if start_index < position:
            return None
        result.extend(source[position:start_index])
        position = start_index
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            tag, text = line[:1], line[1:]
            if tag == " ":
                if position >= len(source) or source[position] != text:
                    return None
                result.append(text)
                position += 1
            elif tag == "-":
                if position >= len(source) or source[position] != text:
                    return None
                position += 1
            elif tag == "+":
                result.append(text)
            elif line.startswith("\\"):
                pass
            else:
                return None
            i += 1
    result.extend(source[position:])
    return "\n".join(result) + "\n"


# --------------------------------------------------------------------------------------------
# Record-level checks
# --------------------------------------------------------------------------------------------


def check_record(record: dict[str, Any], validator: Draft202012Validator, rejected_cwes: set[str], coarse_cwes: set[str], max_context_chars: int, min_lines: int) -> tuple[list[str], list[str]]:
    failed: list[str] = []
    flags: list[str] = []
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
    if errors:
        return ["schema"], flags

    inp, target, prov, loc = record["input"], record["target"], record["provenance"], record["localization"]
    language, excerpt = inp["language"], inp["code"]
    lines = excerpt.splitlines()

    if KIND_DECISION[record["kind"]] != target["decision"]:
        failed.append("kind_decision_consistent")
    if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != prov["content_sha256"]:
        failed.append("content_hash")
    if code.normalized_sha256(language, excerpt) != prov["normalized_sha256"]:
        failed.append("normalized_hash")
    if language not in code.ECOSYSTEM_LANGUAGES[prov["ecosystem"]]:
        failed.append("language_matches_ecosystem")
    if code.language_of(inp["file"]) != language or inp["file"] != prov["file_path"]:
        failed.append("file_language")
    if code.path_role(inp["file"]) != "source":
        failed.append("non_test_source_file")
    if len(lines) < min_lines:
        failed.append("context_complete:min_lines")
    if not inp["context"].strip() or len(inp["context"]) > max_context_chars:
        failed.append("context_complete:context")
    if not code.excerpt_is_well_formed(language, excerpt):
        failed.append("syntax_well_formed")
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(excerpt) or pattern.search(target.get("patch") or ""):
            failed.append(f"no_secrets:{name}")

    evidence_fields = [prov["advisory_id"], prov["vulnerable_revision"], prov["fixed_revision"], prov["fix_committed_at"], target["evidence"]]
    if not all(evidence_fields) or prov["vulnerable_revision"] == prov["fixed_revision"]:
        failed.append("evidence_complete")
    if not prov["license"]["files"]:
        flags.append("no_root_license_file")
    if prov["license"]["category"] in {"unknown", "none"}:
        flags.append(f"license_{prov['license']['category']}")

    if target["decision"] == "finding":
        cwe = target["cwe"]
        if not cwe or cwe in rejected_cwes or any(c in rejected_cwes for c in target["alternate_cwes"]):
            failed.append("cwe_valid")
        if cwe in target["alternate_cwes"]:
            failed.append("cwe_valid:alternate_repeats_primary")
        if cwe in coarse_cwes:
            flags.append("coarse_cwe")
        start, end = target["start_line"], target["end_line"]
        if start is None or end is None or not 1 <= start <= end <= len(lines):
            failed.append("localization_valid:span")
        elif all(code.is_import_or_trivial(language, line) for line in lines[start - 1 : end]):
            failed.append("localization_valid:trivial_span")
        if loc.get("method") != "upstream_fix_diff" or not loc.get("diff_consistent"):
            failed.append("localization_valid:diff_consistent")
        if prov["excerpt_revision"] != prov["vulnerable_revision"]:
            failed.append("revision_consistent")
        if not target["patch"]:
            failed.append("remediation_present")
        if len(target["alternate_cwes"]) > 1:
            flags.append("multi_cwe")
    else:
        if any(target[k] is not None for k in ("cwe", "start_line", "end_line")) or target["alternate_cwes"]:
            failed.append("localization_absent")
        if not target.get("label_scope"):
            failed.append("label_scope_present")
        expected_revision = prov["fixed_revision"] if record["kind"] == "post_fix" else prov["vulnerable_revision"]
        if prov["excerpt_revision"] != expected_revision:
            failed.append("revision_consistent")
    if record["kind"] == "caller_abstention":
        callee = loc.get("withheld_callee", "")
        masked = excerpt if language == "python" else code.mask_c_like(excerpt, language)
        if not callee or not re.search(rf"(?<![\w$]){re.escape(callee)}\s*\(", masked):
            failed.append("abstention_calls_withheld_callee")
        if re.search(rf"(?:def|function|func)\s+(?:\([^)]*\)\s*)?{re.escape(callee)}\s*\(", excerpt):
            failed.append("abstention_callee_not_shown")
    return failed, flags


def check_pairs(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Pair checks: the remediation patch must transform the vulnerable excerpt into the fixed one."""
    failures: dict[str, list[str]] = defaultdict(list)
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["pair_id"]:
            by_pair[record["pair_id"]][record["kind"]] = record
    for pair in by_pair.values():
        vulnerable, fixed = pair.get("vulnerable"), pair.get("post_fix")
        if vulnerable is None or fixed is None:
            for record in pair.values():
                failures[record["record_id"]].append("pair_complete")
            continue
        applied = apply_unified_patch(vulnerable["input"]["code"], vulnerable["target"]["patch"] or "")
        if applied != fixed["input"]["code"]:
            failures[vulnerable["record_id"]].append("patch_applies")
            failures[fixed["record_id"]].append("patch_applies")
        same = ("advisory_id", "repository_key", "vulnerable_revision", "fixed_revision", "file_path")
        if any(vulnerable["provenance"][k] != fixed["provenance"][k] for k in same):
            failures[vulnerable["record_id"]].append("pair_provenance_consistent")
            failures[fixed["record_id"]].append("pair_provenance_consistent")
    return failures


# --------------------------------------------------------------------------------------------
# Dataset-level: duplicates, conflicts, contamination
# --------------------------------------------------------------------------------------------


def _similar_pairs(fingerprints: dict[str, set[int]], threshold: float, max_postings: int = 64) -> list[tuple[str, str, float]]:
    """Record-id pairs with Jaccard >= threshold, via an inverted shingle index."""
    index: dict[int, list[str]] = defaultdict(list)
    for record_id in sorted(fingerprints):
        for shingle in fingerprints[record_id]:
            index[shingle].append(record_id)
    pairs = []
    for record_id in sorted(fingerprints):
        overlap: Counter[str] = Counter()
        for shingle in fingerprints[record_id]:
            postings = index[shingle]
            if len(postings) > max_postings:
                continue
            overlap.update(other for other in postings if other > record_id)
        for other, shared in overlap.items():
            score = code.jaccard(fingerprints[record_id], fingerprints[other])
            if score >= threshold:
                pairs.append((record_id, other, score))
    return sorted(pairs)


def deduplicate(records: list[dict[str, Any]], settings: QualitySettings) -> dict[str, tuple[str, str | None]]:
    """Return record_id -> (reason, duplicate_of) for records that must be rejected.

    Order is deterministic: records are compared in record_id order and the first one is kept.
    """
    rejected: dict[str, tuple[str, str | None]] = {}
    by_id = {r["record_id"]: r for r in records}

    # exact (normalized) duplicates
    first_by_hash: dict[tuple[str, str], str] = {}
    decisions_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in sorted(records, key=lambda r: r["record_id"]):
        normalized = record["provenance"]["normalized_sha256"]
        decisions_by_hash[normalized].add(record["target"]["decision"])
        key = (normalized, record["target"]["decision"])
        if key in first_by_hash:
            rejected[record["record_id"]] = ("exact_duplicate", first_by_hash[key])
        else:
            first_by_hash[key] = record["record_id"]
    for record in records:
        if len(decisions_by_hash[record["provenance"]["normalized_sha256"]]) > 1:
            rejected[record["record_id"]] = ("label_conflict_exact", None)

    fingerprints = {
        r["record_id"]: code.shingles(code.normalized_tokens(r["input"]["language"], r["input"]["code"]), settings.shingle_size)
        for r in records
    }
    threshold = min(settings.near_duplicate_jaccard, settings.label_conflict_jaccard)
    for a, b, score in _similar_pairs(fingerprints, threshold):
        ra, rb = by_id[a], by_id[b]
        if ra["group_id"] == rb["group_id"]:
            continue  # a vulnerable/fixed pair is near-identical by design
        if ra["target"]["decision"] == rb["target"]["decision"]:
            if score >= settings.near_duplicate_jaccard and b not in rejected and a not in rejected:
                rejected[b] = ("near_duplicate", a)
        elif score >= settings.label_conflict_jaccard:
            kinds = {ra["kind"]: a, rb["kind"]: b}
            if set(kinds) == {"vulnerable", "post_fix"}:
                # A "fixed" function that is near-identical to a function later reported vulnerable
                # is not a trustworthy hard negative.
                rejected[kinds["post_fix"]] = ("label_conflict_near", kinds["vulnerable"])
            else:
                rejected.setdefault(a, ("label_conflict_near", b))
                rejected.setdefault(b, ("label_conflict_near", a))
    return rejected


@dataclass(frozen=True)
class EvalReference:
    reference_id: str
    language: str
    text: str
    repository: str | None


def load_eval_references(case_dirs: Iterable[Path], coding_suites: Iterable[Path]) -> list[EvalReference]:
    """Every evaluation case (public and Git-ignored private) and every coding task text."""
    references: list[EvalReference] = []
    for directory in case_dirs:
        for path in sorted(directory.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                case = json.loads(line)
                upstream = (case.get("provenance") or {}).get("upstream") or {}
                references.append(EvalReference(case["case_id"], case["language"], case["code"], upstream.get("repository")))
    for path in coding_suites:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            text = task["prompt"] + task["canonical_solution"] + "\n" + task["tests"]
            references.append(EvalReference(f"coding:{task['task_id']}", "python", text, None))
    return references


def _repo_key(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", url)
    return f"{match.group(1)}/{match.group(2)}".lower() if match else url.lower()


def contamination(records: list[dict[str, Any]], references: list[EvalReference], settings: QualitySettings) -> dict[str, str]:
    """record_id -> reason for every record that overlaps an evaluation case or task."""
    hits: dict[str, str] = {}
    ref_hashes = {code.normalized_sha256(r.language, r.text): r.reference_id for r in references}
    ref_prints = {r.reference_id: code.shingles(code.normalized_tokens(r.language, r.text), settings.shingle_size) for r in references}
    ref_repos = {_repo_key(r.repository): r.reference_id for r in references if r.repository}
    for record in records:
        rid = record["record_id"]
        if record["provenance"]["normalized_sha256"] in ref_hashes:
            hits[rid] = f"eval_contamination_exact:{ref_hashes[record['provenance']['normalized_sha256']]}"
            continue
        if record["provenance"]["repository_key"] in ref_repos:
            hits[rid] = f"eval_repository_overlap:{ref_repos[record['provenance']['repository_key']]}"
            continue
        prints = code.shingles(code.normalized_tokens(record["input"]["language"], record["input"]["code"]), settings.shingle_size)
        for ref_id, ref_print in ref_prints.items():
            if code.jaccard(prints, ref_print) >= settings.contamination_jaccard:
                hits[rid] = f"eval_contamination_near:{ref_id}"
                break
            # an evaluation snippet embedded in a longer training function
            if len(ref_print) >= CONTAINMENT_MIN_SHINGLES and len(prints & ref_print) / len(ref_print) >= CONTAINMENT_THRESHOLD:
                hits[rid] = f"eval_contamination_contained:{ref_id}"
                break
    return hits
