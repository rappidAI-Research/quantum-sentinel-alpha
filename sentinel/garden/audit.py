"""Independent data-quality audit of curated Sentinel Garden records.

Two parts:

1. Automated, evidence-based checks per fix group (vulnerable + post_fix + abstention):
   - advisory <-> code correspondence: is the changed function, file or a changed identifier named in
     the advisory text, and does the upstream commit message reference the advisory?
   - fix-pattern consistency: do the added/removed lines contain a remediation typical for the CWE
     family? (a heuristic signal for manual review, never an automatic exclusion)
   - hard-negative validity: does a LATER upstream fix (another advisory of the same package) change
     the same function, or does another advisory call this fix incomplete? Then the post-fix code is
     not a trustworthy "no_finding" and the post_fix record is excluded.
   - abstention validity: the call to the withheld function must pass non-constant data, and the
     caller must not duplicate the vulnerable code.
   - split leakage: the same advisory/CVE or fix commit in two splits, or near-identical code across
     splits. The train side is excluded; holdout/validation records are never moved.
   - localization precision: span length and share of the function.
2. A deterministic, stratified sample (language x kind x CWE family) for manual review. Manual
   verdicts are stored in `audit/manual_review.yaml` with the reviewer's identity; error rates are
   reported with Wilson 95% intervals.

No exploit or test execution happens here; the report says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from sentinel.garden import code
from sentinel.garden.diff import parse_unified_diff
from sentinel.garden.gitsource import GitError, UpstreamRepo
from sentinel.garden.osv import fix_commit_refs
from sentinel.garden.split import parse_time

AUDIT_VERSION = "2"  # 2: broad localization spans are excluded
LATER_FIX_SIMILARITY = 0.5  # Jaccard of 5-token shingles between our post-fix code and the later-fixed function
OSV_API = "https://api.osv.dev/v1"

# CWE family -> (CWEs, regex of remediation indicators in added or removed lines)
CWE_FAMILIES: dict[str, tuple[set[str], re.Pattern[str]]] = {
    "xss_output_encoding": (
        {"CWE-79", "CWE-80", "CWE-83", "CWE-116", "CWE-1336"},
        re.compile(r"escape|sanitiz|encode|DOMPurify|textContent|innerText|bleach|markupsafe|HTMLEscape|autoescape|xss|purify|Sandbox", re.I),
    ),
    "path_traversal": (
        {"CWE-22", "CWE-23", "CWE-24", "CWE-36", "CWE-73", "CWE-59", "CWE-61", "CWE-706"},
        re.compile(r"\.\.|Clean|Rel\(|IsLocal|realpath|abspath|normpath|commonpath|normalize|resolve|relative|isAbsolute|startswith|startsWith|HasPrefix|secure_filename|EvalSymlinks|Lstat|is_relative_to|basename|Base\(", re.I),
    ),
    "command_injection": (
        {"CWE-78", "CWE-77", "CWE-88"},
        re.compile(r"shell\s*=\s*False|execFile|spawn|shlex|quote|exec\.Command|['\"]--['\"]|escape|argv|\[\s*['\"]|startsWith\(['\"]-|HasPrefix\([^)]*\"-", re.I),
    ),
    "sql_injection": (
        {"CWE-89", "CWE-943"},
        re.compile(r"\?|%s|\$\d|placeholder|bind|param|escape|quote|prepare|Named|text\(", re.I),
    ),
    "code_injection": (
        {"CWE-94", "CWE-95", "CWE-96", "CWE-913"},
        re.compile(r"literal_eval|safe_load|sandbox|JSON\.parse|isolat|allow|deny|block|escape|\beval\b|Function\(|vm\.", re.I),
    ),
    "prototype_pollution": (
        {"CWE-1321", "CWE-915"},
        re.compile(r"__proto__|prototype|constructor|hasOwnProperty|Object\.create\(null\)|hasOwn", re.I),
    ),
    "redirect_ssrf": (
        {"CWE-601", "CWE-918", "CWE-441"},
        re.compile(r"url|host|origin|scheme|protocol|redirect|allow|IsAbs|startsWith|HasPrefix|//|\\\\|parse", re.I),
    ),
    "resource_limits_redos": (
        {"CWE-400", "CWE-1333", "CWE-770", "CWE-674", "CWE-835", "CWE-407", "CWE-789", "CWE-834", "CWE-1284", "CWE-190", "CWE-191", "CWE-369", "CWE-129", "CWE-125", "CWE-787", "CWE-476", "CWE-248", "CWE-754", "CWE-755", "CWE-772", "CWE-401", "CWE-404"},
        re.compile(r"max|limit|len\(|length|size|depth|timeout|count|\{\d|\?\)|\*\?|break|return|nil|null|None|throw|raise|err|catch|try|recover|[<>]=?", re.I),
    ),
    "authn_authz": (
        {"CWE-862", "CWE-863", "CWE-285", "CWE-287", "CWE-306", "CWE-639", "CWE-269", "CWE-266", "CWE-280", "CWE-290", "CWE-294", "CWE-384", "CWE-352", "CWE-346", "CWE-348", "CWE-1188", "CWE-732", "CWE-749"},
        re.compile(r"auth|permission|role|admin|owner|user|token|session|csrf|check|verify|forbid|deny|401|403|allow|origin|scope|access|privilege|valid", re.I),
    ),
    "information_exposure": (
        {"CWE-200", "CWE-201", "CWE-202", "CWE-203", "CWE-208", "CWE-209", "CWE-532", "CWE-312", "CWE-359", "CWE-524", "CWE-489"},
        re.compile(r"redact|mask|omit|delete|remove|strip|\*\*\*|sanitiz|constant.?time|compare_digest|ConstantTimeCompare|timingSafeEqual|hmac|log|secret|password|token|header", re.I),
    ),
    "crypto_tls_integrity": (
        {"CWE-295", "CWE-347", "CWE-327", "CWE-328", "CWE-330", "CWE-338", "CWE-311", "CWE-326", "CWE-345", "CWE-354", "CWE-303", "CWE-377"},
        re.compile(r"verify|InsecureSkipVerify|signature|alg|crypto|random|secrets|hash|hmac|cert|tls|secure|key|nonce|mode|perm|0o?[67]00", re.I),
    ),
    "deserialization_input_validation": (
        {"CWE-502", "CWE-20", "CWE-74", "CWE-184", "CWE-178", "CWE-1287", "CWE-1289", "CWE-436", "CWE-444"},
        re.compile(r"safe_load|SafeLoader|allow|deny|block|literal_eval|json|valid|check|reject|match|regex|re\.|RegExp|test\(|lower|upper|normalize|strip|trim|type", re.I),
    ),
}

INCOMPLETE_WORDS = re.compile(r"incomplete|bypass|insufficient|regression|still vulnerable|did not fully|partial fix|not fully", re.I)
STOP_IDENTIFIERS = {
    "return", "const", "let", "var", "func", "function", "string", "import", "from", "self", "this", "true", "false",
    "None", "null", "nil", "error", "async", "await", "else", "elif", "range", "value", "values", "data", "result",
    "request", "response", "params", "options", "config", "object", "class", "static", "public", "private",
}


def family_of(cwe: str | None) -> str:
    for family, (cwes, _) in CWE_FAMILIES.items():
        if cwe in cwes:
            return family
    return "other"


def wilson_interval(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = errors / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


# --------------------------------------------------------------------------------------------
# Evidence sources
# --------------------------------------------------------------------------------------------


def _post_json(url: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "User-Agent": "quantum-sentinel-alpha-sentinel-garden/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


class OsvQuery:
    """All OSV advisories of a package (cached), used to find later fixes and incomplete-fix reports."""

    def __init__(self, cache_dir: Path, post: Callable[[str, dict[str, Any]], Any] = _post_json) -> None:
        self.cache_dir = cache_dir
        self.post = post

    def package_vulns(self, ecosystem: str, package: str) -> list[dict[str, Any]] | None:
        key = hashlib.sha256(f"{ecosystem}\x00{package}".encode()).hexdigest()[:24]
        cache = self.cache_dir / f"{key}.json"
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        vulns: list[dict[str, Any]] = []
        token = None
        try:
            for _ in range(20):
                payload: dict[str, Any] = {"package": {"name": package, "ecosystem": ecosystem}}
                if token:
                    payload["page_token"] = token
                page = self.post(f"{OSV_API}/query", payload)
                vulns.extend(page.get("vulns") or [])
                token = page.get("next_page_token")
                if not token:
                    break
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(vulns), encoding="utf-8")
        return vulns


# --------------------------------------------------------------------------------------------
# Automated checks
# --------------------------------------------------------------------------------------------


def _identifiers(lines: list[str]) -> set[str]:
    found = set()
    for line in lines:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", line):
            if token not in STOP_IDENTIFIERS:
                found.add(token)
    return found


def _changed_lines(patch: str) -> tuple[list[str], list[str]]:
    body = [line for line in patch.splitlines()[2:] if not line.startswith("@@")]
    removed = [line[1:] for line in body if line.startswith("-")]
    added = [line[1:] for line in body if line.startswith("+")]
    return removed, added


def correspondence(vulnerable: dict[str, Any], advisory_text: str, commit_message: str) -> dict[str, Any]:
    provenance = vulnerable["provenance"]
    function = provenance["excerpt_function"]
    removed, added = _changed_lines(vulnerable["target"]["patch"])
    signals = []
    if not function.startswith("<") and re.search(rf"(?<![\w$]){re.escape(function)}(?![\w$])", advisory_text):
        signals.append("function_named_in_advisory")
    path = provenance["file_path"]
    stem = Path(path).stem
    if path in advisory_text or Path(path).name in advisory_text or (len(stem) >= 5 and re.search(rf"\b{re.escape(stem)}\b", advisory_text)):
        signals.append("file_named_in_advisory")
    overlap = sorted(i for i in _identifiers(removed + added) if re.search(rf"(?<![\w$]){re.escape(i)}(?![\w$])", advisory_text))
    if overlap:
        signals.append("changed_identifier_named_in_advisory")
    ids = [provenance["advisory_id"], *provenance["aliases"]]
    if any(i in commit_message for i in ids):
        signals.append("commit_message_cites_advisory")
    elif re.search(r"secur|vulnerab|CVE-|GHSA-|exploit|inject|travers|xss|csrf|ssrf|redos|sanitiz|escape|bypass|unauthori", commit_message, re.I):
        signals.append("commit_message_security_related")
    strong = {"function_named_in_advisory", "file_named_in_advisory", "commit_message_cites_advisory"}
    strength = "strong" if strong & set(signals) else "moderate" if signals else "weak"
    return {"strength": strength, "signals": signals, "identifier_overlap": overlap[:8]}


def fix_pattern(vulnerable: dict[str, Any]) -> dict[str, Any]:
    family = family_of(vulnerable["target"]["cwe"])
    if family == "other":
        return {"family": family, "consistent": None}
    removed, added = _changed_lines(vulnerable["target"]["patch"])
    pattern = CWE_FAMILIES[family][1]
    hits = sorted({m.group(0) for line in added + removed for m in pattern.finditer(line)})
    return {"family": family, "consistent": bool(hits), "indicators": hits[:6]}


def span_precision(vulnerable: dict[str, Any]) -> dict[str, Any]:
    target = vulnerable["target"]
    lines = len(vulnerable["input"]["code"].splitlines())
    span = target["end_line"] - target["start_line"] + 1
    return {"span_lines": span, "function_lines": lines, "share": round(span / lines, 3), "broad": span > 12 and span / lines > 0.4}


def _call_arguments(line: str, callee: str) -> str | None:
    match = re.search(rf"(?<![\w$]){re.escape(callee)}\s*\(", line)
    if not match:
        return None
    depth, start = 1, match.end()
    for i in range(start, len(line)):
        if line[i] == "(":
            depth += 1
        elif line[i] == ")":
            depth -= 1
            if depth == 0:
                return line[start:i]
    return None  # the call continues on the next line: its arguments are not known to be constant


_LITERAL_ARGS = re.compile(r"^\s*(?:(?:-?\d+(?:\.\d+)?|true|false|True|False|None|null|nil|undefined|\"[^\"]*\"|'[^']*'|`[^`$]*`)\s*,?\s*)*$")


def abstention_validity(abstention: dict[str, Any], vulnerable: dict[str, Any] | None) -> dict[str, Any]:
    callee = abstention["localization"]["withheld_callee"]
    excerpt = abstention["input"]["code"]
    lines = excerpt.splitlines()
    line = lines[abstention["localization"]["call_line"] - 1] if abstention["localization"]["call_line"] <= len(lines) else ""
    arguments = _call_arguments(line, callee)
    problems = []
    # Only a call whose arguments are all literals cannot carry attacker data. A call without
    # arguments is not flagged: a method's receiver or module state may carry the data.
    if arguments is not None and arguments.strip() and _LITERAL_ARGS.match(arguments):
        problems.append("constant_arguments")
    if vulnerable is not None:
        language = abstention["input"]["language"]
        a = code.shingles(code.normalized_tokens(language, excerpt))
        b = code.shingles(code.normalized_tokens(language, vulnerable["input"]["code"]))
        if code.jaccard(a, b) >= 0.5:
            problems.append("caller_duplicates_vulnerable_code")
    return {"call_arguments": (arguments or "")[:120], "problems": problems}


@dataclass
class LaterFix:
    advisory_id: str
    reason: str
    commit: str | None = None


def later_fixes(
    group: dict[str, dict[str, Any]],
    package_vulns: list[dict[str, Any]],
    repo: UpstreamRepo,
) -> list[LaterFix]:
    """Other advisories that show the post-fix function is still (or again) vulnerable."""
    vulnerable = group["vulnerable"]
    provenance = vulnerable["provenance"]
    own_ids = {provenance["advisory_id"], *provenance["aliases"]}
    fixed_at = parse_time(provenance["fix_committed_at"])
    function, path = provenance["excerpt_function"], provenance["file_path"]
    post_fix_code = group["post_fix"]["input"]["code"]
    found: list[LaterFix] = []
    for vuln in package_vulns:
        ids = {vuln.get("id"), *(vuln.get("aliases") or [])}
        if ids & own_ids:
            continue
        text = f"{vuln.get('summary', '')}\n{vuln.get('details', '')}"
        if any(i in text for i in own_ids) and INCOMPLETE_WORDS.search(text):
            found.append(LaterFix(vuln["id"], "reported_incomplete_fix"))
            continue
        for owner, name, sha in fix_commit_refs(vuln):
            if f"{owner}/{name}".lower() != provenance["repository_key"] or len(sha) != 40 or sha == provenance["fixed_revision"]:
                continue
            try:
                info = repo.ensure_commit(sha)
                if not info.parents or parse_time(info.committed_at) <= fixed_at:
                    continue
                if path not in {p for _, p in repo.changed_files(info.parents[0], sha)}:
                    continue
                old_source = repo.text(info.parents[0], path)
                new_source = repo.text(sha, path)
                language = vulnerable["input"]["language"]
                functions = code.find_functions(language, old_source)
                # noise only: a later fix that just moves a brace still changes the function's logic
                old_trivial = code.noise_lines(language, old_source)
                new_trivial = code.noise_lines(language, new_source)
                for hunk in parse_unified_diff(repo.file_diff(info.parents[0], sha, path)):
                    if all(n in old_trivial for n, _ in hunk.removed) and all(n in new_trivial for n, _ in hunk.added):
                        continue
                    anchor = max(hunk.old_start, 1)
                    fn = code.innermost(functions, anchor, anchor + max(hunk.old_count, 1) - 1)
                    if fn is None or fn.name != function:
                        continue
                    # Same name is not enough (anonymous callbacks, same-named methods of other
                    # classes): the function the later fix changed must resemble our post-fix code.
                    changed = "\n".join(old_source.splitlines()[fn.start - 1 : fn.end]) + "\n"
                    similarity = code.jaccard(
                        code.shingles(code.normalized_tokens(language, changed)),
                        code.shingles(code.normalized_tokens(language, post_fix_code)),
                    )
                    if similarity >= LATER_FIX_SIMILARITY:
                        found.append(LaterFix(vuln["id"], "later_fix_changes_same_function", sha))
                        break
            except GitError:
                continue
            if found and found[-1].advisory_id == vuln["id"]:
                break
    return found


# --------------------------------------------------------------------------------------------
# Leakage across splits
# --------------------------------------------------------------------------------------------


def audit_population(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records accepted apart from earlier audit exclusions, so re-running the audit is stable."""
    return [r for r in records if all(c.startswith("audit:") for c in r["quality"]["failed_checks"])]


def split_leakage(records: list[dict[str, Any]], threshold: float = 0.7) -> dict[str, str]:
    """record_id (train side only) -> reason, for advisories/commits/code shared with another split."""
    accepted = audit_population(records)
    leaks: dict[str, str] = {}
    by_key: dict[str, set[str]] = defaultdict(set)
    for record in accepted:
        provenance = record["provenance"]
        for key in [provenance["advisory_id"], *provenance["aliases"], provenance["fixed_revision"]]:
            by_key[key].add(record["split"])
    for record in accepted:
        if record["split"] != "train":
            continue
        provenance = record["provenance"]
        for key in [provenance["advisory_id"], *provenance["aliases"], provenance["fixed_revision"]]:
            others = by_key[key] - {"train"}
            if others:
                leaks[record["record_id"]] = f"shared_{'commit' if key == provenance['fixed_revision'] else 'advisory'}_with_{sorted(others)[0]}"
                break
    prints = {r["record_id"]: code.shingles(code.normalized_tokens(r["input"]["language"], r["input"]["code"])) for r in accepted}
    held = [r for r in accepted if r["split"] != "train"]
    for record in accepted:
        if record["split"] != "train" or record["record_id"] in leaks:
            continue
        for other in held:
            if other["input"]["language"] == record["input"]["language"] and code.jaccard(prints[record["record_id"]], prints[other["record_id"]]) >= threshold:
                leaks[record["record_id"]] = f"near_duplicate_of_{other['split']}"
                break
    return leaks


# --------------------------------------------------------------------------------------------
# Stratified manual-review sample
# --------------------------------------------------------------------------------------------


def stratified_sample(groups: dict[str, dict[str, dict[str, Any]]], pairs_per_language: int, abstentions_per_language: int, salt: str) -> list[dict[str, str]]:
    """Deterministic sample: per language, pairs round-robin over CWE families, plus abstentions."""
    def order(group_id: str) -> str:
        return hashlib.sha256(f"{salt}\x00{group_id}".encode()).hexdigest()

    sample: list[dict[str, str]] = []
    by_language: dict[str, list[str]] = defaultdict(list)
    for group_id, group in groups.items():
        if "vulnerable" in group and "post_fix" in group:
            by_language[group["vulnerable"]["input"]["language"]].append(group_id)
    for language in sorted(by_language):
        families: dict[str, list[str]] = defaultdict(list)
        for group_id in sorted(by_language[language], key=order):
            families[family_of(groups[group_id]["vulnerable"]["target"]["cwe"])].append(group_id)
        chosen: list[str] = []
        while len(chosen) < pairs_per_language and any(families.values()):
            for family in sorted(families):
                if families[family] and len(chosen) < pairs_per_language:
                    chosen.append(families[family].pop(0))
        sample += [{"group_id": g, "language": language, "item": "pair"} for g in chosen]
        abstentions = sorted((g for g in by_language[language] if "caller_abstention" in groups[g]), key=order)
        sample += [{"group_id": g, "language": language, "item": "abstention"} for g in abstentions[:abstentions_per_language]]
    return sample


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def audit_exclusions(results: list[dict[str, Any]], leakage: dict[str, str]) -> list[dict[str, str]]:
    """Automated exclusions from the per-group audit results and the split-leakage check."""
    exclusions = []
    for result in results:
        records_by_kind = result["records"]
        for fix in result.get("later_fixes", []):
            exclusions.append({"record_id": records_by_kind["post_fix"], "reason": f"{fix['reason']}:{fix['advisory_id']}"})
        for problem in result.get("abstention", {}).get("problems", []):
            exclusions.append({"record_id": records_by_kind["caller_abstention"], "reason": f"abstention_{problem}"})
        if result.get("span", {}).get("broad") and "vulnerable" in records_by_kind:
            # In the manual sample every broad span was imprecise (it covers unrelated edits made in
            # the same commit, such as renames or docstrings); a localization target that cannot be
            # trusted is not trained on. curate rejects the group's siblings with it.
            exclusions.append({"record_id": records_by_kind["vulnerable"], "reason": "broad_localization_span"})
    for record_id, reason in sorted(leakage.items()):
        exclusions.append({"record_id": record_id, "reason": f"split_leakage_{reason}"})
    return exclusions


def run_audit(
    records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    git_cache: Path,
    osv_record_dir: Path,
    osv_query: OsvQuery,
    workers: int = 8,
) -> dict[str, Any]:
    accepted = audit_population(records)
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in accepted:
        groups[record["group_id"]][record["kind"]] = record
    by_advisory = {c["advisory_id"]: c for c in candidates}

    def advisory_text(advisory_id: str, ecosystem: str) -> str:
        path = osv_record_dir / ecosystem / f"{advisory_id}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            return f"{record.get('summary', '')}\n{record.get('details', '')}"
        return by_advisory.get(advisory_id, {}).get("summary", "")

    def audit_group(group_id: str) -> dict[str, Any]:
        group = groups[group_id]
        anchor = group.get("vulnerable") or next(iter(group.values()))
        provenance = anchor["provenance"]
        result: dict[str, Any] = {
            "group_id": group_id,
            "records": {kind: r["record_id"] for kind, r in sorted(group.items())},
            "language": anchor["input"]["language"],
            "split": anchor["split"],
            "advisory_id": provenance["advisory_id"],
            "repository_key": provenance["repository_key"],
        }
        repo = UpstreamRepo(git_cache, provenance["repository"], provenance["repository_key"])
        if "vulnerable" in group:
            vulnerable = group["vulnerable"]
            try:
                message = repo.commit_message(provenance["fixed_revision"])
            except GitError:
                message = ""
            result["cwe"] = vulnerable["target"]["cwe"]
            result["correspondence"] = correspondence(vulnerable, advisory_text(provenance["advisory_id"], provenance["ecosystem"]), message)
            result["fix_pattern"] = fix_pattern(vulnerable)
            result["span"] = span_precision(vulnerable)
            if "post_fix" in group:
                packages = by_advisory.get(provenance["advisory_id"], {}).get("packages", provenance["packages"])
                vulns: list[dict[str, Any]] = []
                unavailable = False
                for package in packages:
                    found = osv_query.package_vulns(provenance["ecosystem"], package)
                    if found is None:
                        unavailable = True
                    else:
                        vulns.extend(found)
                result["later_fixes"] = [vars(f) for f in later_fixes(group, vulns, repo)]
                result["later_fix_check"] = "incomplete" if unavailable else "done"
        if "caller_abstention" in group:
            result["abstention"] = abstention_validity(group["caller_abstention"], group.get("vulnerable"))
        return result

    # one worker per repository: git fetches into the same cache must not run concurrently
    by_repo: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(groups):
        by_repo[next(iter(groups[group_id].values()))["provenance"]["repository_key"]].append(group_id)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_repo = list(pool.map(lambda key: [audit_group(g) for g in by_repo[key]], sorted(by_repo)))
    results = sorted((r for chunk in per_repo for r in chunk), key=lambda r: r["group_id"])
    leakage = split_leakage(records)
    exclusions = audit_exclusions(results, leakage)
    unique = {(e["record_id"], e["reason"]): e for e in exclusions}
    return {
        "audit_version": AUDIT_VERSION,
        "groups": results,
        "exclusions": sorted(unique.values(), key=lambda e: (e["record_id"], e["reason"])),
        "sample": stratified_sample(groups, pairs_per_language=8, abstentions_per_language=3, salt="sentinel-garden-audit-v1"),
    }


def load_exclusions(path: Path) -> dict[str, list[str]]:
    """record_id -> reasons, from automated and manual audit exclusions."""
    if not path.is_file():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result: dict[str, list[str]] = defaultdict(list)
    for section in ("automated", "manual"):
        for entry in document.get(section) or []:
            result[entry["record_id"]].append(entry["reason"])
    return dict(result)


MANUAL_FIELDS = {
    "advisory_matches_code": {"yes", "partly", "no"},
    "fix_addresses_weakness": {"yes", "partly", "no"},
    "localization": {"precise", "acceptable", "imprecise", "wrong"},
    "cwe": {"defensible", "questionable", "wrong"},
    "hard_negative_safe": {"yes", "unclear", "no"},
    "abstention_valid": {"yes", "unclear", "no"},
}


def load_manual_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"reviewer": None, "items": []}
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in document["items"]:
        for key, allowed in MANUAL_FIELDS.items():
            if key in item and item[key] not in allowed:
                raise ValueError(f"{item['group_id']}: {key}={item[key]!r} not in {sorted(allowed)}")
    return document


def summarize(audit: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
    groups = audit["groups"]
    with_vulnerable = [g for g in groups if "correspondence" in g]
    patterns = [g for g in with_vulnerable if g["fix_pattern"]["consistent"] is not None]
    summary: dict[str, Any] = {
        "groups": len(groups),
        "correspondence": dict(sorted(Counter(g["correspondence"]["strength"] for g in with_vulnerable).items())),
        "correspondence_by_language": {
            lang: dict(sorted(Counter(g["correspondence"]["strength"] for g in with_vulnerable if g["language"] == lang).items()))
            for lang in sorted({g["language"] for g in with_vulnerable})
        },
        "fix_pattern_consistent": f"{sum(1 for g in patterns if g['fix_pattern']['consistent'])}/{len(patterns)}",
        "fix_pattern_by_family": {
            fam: f"{sum(1 for g in patterns if g['fix_pattern']['family'] == fam and g['fix_pattern']['consistent'])}/{sum(1 for g in patterns if g['fix_pattern']['family'] == fam)}"
            for fam in sorted({g["fix_pattern"]["family"] for g in patterns})
        },
        "broad_spans": sum(1 for g in with_vulnerable if g["span"]["broad"]),
        "span_lines_median": sorted(g["span"]["span_lines"] for g in with_vulnerable)[len(with_vulnerable) // 2] if with_vulnerable else None,
        "later_fix_checked": sum(1 for g in groups if g.get("later_fix_check") == "done"),
        "later_fix_incomplete_lookups": sum(1 for g in groups if g.get("later_fix_check") == "incomplete"),
        "post_fix_invalidated": sum(1 for g in groups if g.get("later_fixes")),
        "abstentions_checked": sum(1 for g in groups if "abstention" in g),
        "abstentions_invalid": sum(1 for g in groups if g.get("abstention", {}).get("problems")),
        "exclusions_by_reason": dict(sorted(Counter(e["reason"].split(":")[0] for e in audit["exclusions"]).items())),
    }
    # Error rates describe the current sample only: a verdict on a group that is no longer sampled
    # (the population changed) is kept for its exclusions but not counted, and sampled items without a
    # verdict are reported as pending rather than silently missing from the denominator.
    sample_keys = [(s["group_id"], s.get("item", "pair")) for s in audit.get("sample") or []]
    by_key = {(i["group_id"], i.get("item", "pair")): i for i in manual.get("items") or []}
    items = [by_key[k] for k in sample_keys if k in by_key]
    pending = [f"{g}:{kind}" for g, kind in sample_keys if (g, kind) not in by_key]
    outside = sorted(f"{g}:{kind}" for g, kind in by_key if (g, kind) not in set(sample_keys))
    rates = {}
    for key, bad in (
        ("advisory_matches_code", {"no"}),
        ("fix_addresses_weakness", {"no"}),
        ("localization", {"imprecise", "wrong"}),
        ("cwe", {"wrong"}),
        ("hard_negative_safe", {"no"}),
        ("abstention_valid", {"no"}),
    ):
        judged = [i for i in items if key in i]
        errors = sum(1 for i in judged if i[key] in bad)
        uncertain = sum(1 for i in judged if i[key] in {"unclear", "partly", "questionable", "acceptable"})
        low, high = wilson_interval(errors, len(judged))
        rates[key] = {"judged": len(judged), "errors": errors, "uncertain_or_partial": uncertain, "error_rate": round(errors / len(judged), 3) if judged else None, "wilson95": [round(low, 3), round(high, 3)]}
    # The same rates over the sampled items that survive the automated exclusions: the error estimate
    # for the records that remain (manual findings still count - they are evidence, not removals).
    auto_excluded = {e["record_id"] for e in audit.get("exclusions") or []}
    group_records = {g["group_id"]: g.get("records", {}) for g in groups}
    kept = [
        i for i in items
        if group_records.get(i["group_id"], {}).get("vulnerable" if i.get("item", "pair") == "pair" else "caller_abstention") not in auto_excluded
    ]
    kept_rates = {}
    for key, rate in rates.items():
        bad = {"advisory_matches_code": {"no"}, "fix_addresses_weakness": {"no"}, "localization": {"imprecise", "wrong"}, "cwe": {"wrong"}, "hard_negative_safe": {"no"}, "abstention_valid": {"no"}}[key]
        judged = [i for i in kept if key in i]
        errors = sum(1 for i in judged if i[key] in bad)
        low, high = wilson_interval(errors, len(judged))
        kept_rates[key] = {"judged": len(judged), "errors": errors, "error_rate": round(errors / len(judged), 3) if judged else None, "wilson95": [round(low, 3), round(high, 3)]}
    summary["manual"] = {
        "reviewer": manual.get("reviewer"),
        "rates_after_automated_exclusions": kept_rates,
        "sample_items": len(sample_keys),
        "items": len(items),
        "pending": pending,
        "outside_sample": outside,
        "rates": rates,
    }
    return summary


def report_markdown(summary: dict[str, Any], audit: dict[str, Any], manual: dict[str, Any]) -> str:
    """Audit report; every number is computed from the audit outputs."""
    groups = audit["groups"]
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exclusion in audit["exclusions"]:
        by_reason[exclusion["reason"].split(":")[0]].append(exclusion)
    later = [g for g in groups if g.get("later_fixes")]
    weak = [g for g in groups if g.get("correspondence", {}).get("strength") == "weak"]
    unconfirmed = [g for g in groups if g.get("fix_pattern", {}).get("consistent") is False]
    rates = summary["manual"]["rates"]
    lines = [
        "# Sentinel Garden data-quality audit",
        "",
        f"Generated by `python -m sentinel.garden audit` (audit v{audit['audit_version']}). Population: {summary['groups']} fix groups "
        "accepted by the pipeline before audit exclusions. **No exploit, test or patch was executed**: findings rest on advisory "
        "text, upstream commits and diffs, OSV records of the same packages and manual reading.",
        "",
        "## Automated checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Advisory ↔ code correspondence (strong / moderate / weak) | {summary['correspondence'].get('strong', 0)} / {summary['correspondence'].get('moderate', 0)} / {summary['correspondence'].get('weak', 0)} |",
        f"| Fix pattern consistent with CWE family (heuristic) | {summary['fix_pattern_consistent']} |",
        f"| Broad localization spans (>12 lines and >40 % of the function) | {summary['broad_spans']} (median span {summary['span_lines_median']} lines) |",
        f"| Hard negatives checked against later advisories of the same package | {summary['later_fix_checked']} (lookups incomplete: {summary['later_fix_incomplete_lookups']}) |",
        f"| Hard negatives invalidated (later fix of the same function / reported incomplete) | {summary['post_fix_invalidated']} |",
        f"| Abstentions checked / invalid | {summary['abstentions_checked']} / {summary['abstentions_invalid']} |",
        "",
        "Correspondence by language: " + "; ".join(f"{lang}: {dict(v)}" for lang, v in summary["correspondence_by_language"].items()) + ".",
        "",
        "Fix-pattern consistency by CWE family: " + "; ".join(f"{fam} {v}" for fam, v in summary["fix_pattern_by_family"].items()) + ".",
        "",
        "## Exclusions applied by `curate`",
        "",
        "| Reason | Records |",
        "| --- | ---: |",
        *[f"| {reason} | {len(items)} |" for reason, items in sorted(by_reason.items())],
        *([f"| manual review | {len(manual_exclusions(manual))} |"] if manual_exclusions(manual) else []),
        "",
    ]
    if later:
        lines += ["### Invalidated hard negatives", "", "| Group | Advisory | Function | Later evidence |", "| --- | --- | --- | --- |"]
        for g in later[:40]:
            lines.append(f"| `{g['group_id']}` | {g['advisory_id']} | {g['repository_key']} | " + ", ".join(f"{f['reason']} {f['advisory_id']}" for f in g["later_fixes"]) + " |")
        lines.append("")
    lines += [
        "## Manual stratified review",
        "",
        f"Reviewer: {manual.get('reviewer') or 'not yet performed'}. Reviewed {summary['manual']['items']} of "
        f"{summary['manual']['sample_items']} sampled items"
        + (f" (**{len(summary['manual']['pending'])} pending**)" if summary["manual"]["pending"] else "")
        + " (sample definition: `audit/sample.jsonl`; verdicts: `audit/manual_review.yaml`). Rates count sampled items only"
        + (f"; {len(summary['manual']['outside_sample'])} earlier verdicts on groups no longer sampled still apply their exclusions." if summary["manual"]["outside_sample"] else "."),
        "",
        "| Question | Judged | Errors | Uncertain / partial | Error rate | Wilson 95 % |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, rate in rates.items():
        error_rate = "-" if rate["error_rate"] is None else f"{rate['error_rate']:.1%}"
        lines.append(f"| {key} | {rate['judged']} | {rate['errors']} | {rate['uncertain_or_partial']} | {error_rate} | {rate['wilson95'][0]:.1%} - {rate['wilson95'][1]:.1%} |")
    kept = summary["manual"].get("rates_after_automated_exclusions") or {}
    if kept:
        lines += [
            "",
            "The same questions over the sampled items whose records survive the automated exclusions (the estimate for the exported data; errors found manually still count):",
            "",
            "| Question | Judged | Errors | Error rate | Wilson 95 % |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for key, rate in kept.items():
            error_rate = "-" if rate["error_rate"] is None else f"{rate['error_rate']:.1%}"
            lines.append(f"| {key} | {rate['judged']} | {rate['errors']} | {error_rate} | {rate['wilson95'][0]:.1%} - {rate['wilson95'][1]:.1%} |")
    notes = [i for i in manual.get("items") or [] if i.get("notes")]
    if notes:
        lines += ["", "Findings from the manual review:", ""]
        lines += [f"- `{i['group_id']}` ({i.get('item', 'pair')}): {i['notes']}" for i in notes]
    lines += [
        "",
        "## Uncertainties",
        "",
        f"- {len(weak)} groups have only weak textual correspondence between advisory and changed code; they are kept (the fix commit is referenced by a GitHub-reviewed advisory) but are the first candidates for review.",
        f"- {len(unconfirmed)} groups show no remediation pattern typical for their CWE family; this is a heuristic and is not an exclusion reason on its own.",
        "- A post-fix record is a hard negative for the advisory's weakness only. Later fixes are found only when OSV links them to the same package with a commit reference.",
        "- Localization is diff-derived: the changed lines of the fix or the insertion point of an added check.",
        "",
    ]
    return "\n".join(lines)


def manual_exclusions(manual: dict[str, Any]) -> list[dict[str, str]]:
    """Records a manual verdict marks as wrong; these are excluded, never relabelled."""
    out = []
    for item in manual.get("items") or []:
        reasons = [key for key, bad in (("advisory_matches_code", "no"), ("fix_addresses_weakness", "no"), ("localization", "wrong"), ("cwe", "wrong")) if item.get(key) == bad]
        for record_id in item.get("exclude_records") or []:
            out.append({"record_id": record_id, "reason": "manual_" + ("_".join(reasons) or item.get("exclude_reason", "review"))})
    return out
