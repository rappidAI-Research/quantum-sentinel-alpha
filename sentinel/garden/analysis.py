"""Dataset composition, shortcut probes and development-run error analysis (Garden v4 weakness analysis).

    python -m sentinel.garden analyze --config configs/garden/seed_v3.yaml --output data/garden/seed_v4/analysis/sentinel-garden-v3.json
    python -m sentinel.garden analyze --config configs/garden/seed_v4.yaml --output data/garden/seed_v4/analysis/sentinel-garden-v4.json \\
        [--runs runs/eval/dev/*.json --cases runs/devset/dev.eval_cases.jsonl]

Reads the curated records with code (`<workdir>/records.jsonl`) and writes aggregate numbers only: no
code, context, evidence or patch text, so the output can be committed. It measures; it does not label
or exclude anything.

Shortcut probes ask whether a label can be predicted without understanding the code. Each probe is
fitted on the train split and scored on the validation split (repository-disjoint):
- `abstention_phrase`: the abstention-only context sentence of Garden v1-v3;
- `excerpt_length`: the best single line-count threshold between finding and no_finding;
- `multi_block_excerpt`: excerpts made of several file blocks (a fix that added a helper);
- `code_tokens_nb` / `context_tokens_nb`: a multinomial naive Bayes model over identifier tokens of the
  code or of the context only, with its pair accuracy (does it score a vulnerable excerpt above its own
  post-fix version?).
A probe far above chance means the label is partly predictable from surface features: a model may
learn the cue instead of the weakness.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sentinel.garden.audit import family_of

ANALYSIS_VERSION = "1"
ABSTENTION_PHRASE = "its body is not included"  # Garden v1-v3 abstention context (sentinel/garden/build.py)
CALLEE_NOTE = "and not shown:"  # Garden v4 note, on every record kind
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Import names that identify a framework or a security-relevant library ("imports mention", a heuristic).
FRAMEWORKS = {
    "python": ["django", "flask", "fastapi", "starlette", "tornado", "aiohttp", "jinja2", "sqlalchemy", "requests", "urllib", "yaml", "pickle", "subprocess", "lxml", "werkzeug"],
    "javascript": ["express", "koa", "fastify", "react", "vue", "angular", "next", "lodash", "axios", "jsonwebtoken", "child_process", "fs", "path", "url", "crypto"],
    "typescript": ["express", "koa", "fastify", "react", "vue", "@angular", "next", "lodash", "axios", "jsonwebtoken", "child_process", "fs", "path", "url", "crypto"],
    "go": ["net/http", "gin-gonic", "labstack/echo", "gorilla", "grpc", "os/exec", "path/filepath", "crypto", "html/template", "database/sql", "k8s.io", "encoding/json"],
}


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": ordered[-1]}


def _frameworks(record: dict[str, Any]) -> list[str]:
    context = record["input"]["context"]
    imports = context.split("File imports:", 1)[1] if "File imports:" in context else ""
    return [name for name in FRAMEWORKS[record["input"]["language"]] if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", imports)]


def composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate composition of a record population (accepted records with code)."""
    findings = [r for r in records if r["target"]["decision"] == "finding"]
    repos = Counter(r["provenance"]["repository_key"] for r in records)
    top10 = sum(count for _, count in repos.most_common(10))
    frameworks: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for name in _frameworks(record):
            frameworks[record["input"]["language"]][name] += 1
    return {
        "records": len(records),
        "by_decision": dict(sorted(Counter(r["target"]["decision"] for r in records).items())),
        "by_language": dict(sorted(Counter(r["input"]["language"] for r in records).items())),
        "findings_by_family": dict(Counter(family_of(r["target"]["cwe"]) for r in findings).most_common()),
        "findings_top_cwes": dict(Counter(r["target"]["cwe"] for r in findings).most_common(15)),
        "distinct_finding_cwes": len({r["target"]["cwe"] for r in findings}),
        "findings_by_language_and_family": {
            language: dict(Counter(family_of(r["target"]["cwe"]) for r in findings if r["input"]["language"] == language).most_common())
            for language in sorted({r["input"]["language"] for r in findings})
        },
        "repositories": len(repos),
        "top10_repository_share": round(top10 / len(records), 3) if records else None,
        "records_per_repository": _quantiles(list(repos.values())),
        "excerpt_lines": {
            decision: _quantiles([len(r["input"]["code"].splitlines()) for r in records if r["target"]["decision"] == decision])
            for decision in ("finding", "no_finding", "abstain")
        },
        "span_lines": _quantiles([r["target"]["end_line"] - r["target"]["start_line"] + 1 for r in findings]),
        "span_kind": dict(sorted(Counter(r["localization"].get("span_kind") for r in findings).items())),
        "multi_function_findings": sum(1 for r in findings if r["provenance"].get("excerpt_functions")),
        "commit_chain_findings": sum(1 for r in findings if r["provenance"].get("fix_commit_chain")),
        "coarse_or_multi_cwe_findings": sum(1 for r in findings if "coarse_cwe" in r.get("quality", {}).get("flags", []) or r["target"]["alternate_cwes"]),
        "context_mentions_unshown_callee": {
            decision: round(sum(1 for r in group if ABSTENTION_PHRASE in r["input"]["context"] or CALLEE_NOTE in r["input"]["context"]) / len(group), 3)
            for decision, group in _by_decision(records).items()
        },
        "frameworks_in_imports": {language: dict(counter.most_common(8)) for language, counter in sorted(frameworks.items())},
    }


def _by_decision(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["target"]["decision"]].append(record)
    return dict(sorted(groups.items()))


class NaiveBayes:
    """Multinomial naive Bayes over identifier tokens (lower-cased), Laplace smoothing."""

    def __init__(self, documents: list[tuple[list[str], str]]) -> None:
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.docs: Counter[str] = Counter()
        for tokens, label in documents:
            self.counts[label].update(tokens)
            self.docs[label] += 1
        self.vocabulary = {t for c in self.counts.values() for t in c}
        self.totals = {label: sum(c.values()) for label, c in self.counts.items()}

    def score(self, tokens: list[str], label: str) -> float:
        prior = math.log(self.docs[label] / sum(self.docs.values()))
        size = len(self.vocabulary) + 1
        return prior + sum(math.log((self.counts[label][t] + 1) / (self.totals[label] + size)) for t in tokens)

    def predict(self, tokens: list[str]) -> str:
        return max(sorted(self.docs), key=lambda label: self.score(tokens, label))


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def shortcut_probes(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    """Surface-feature probes fitted on `train` and scored on `test` (see module docstring)."""
    result: dict[str, Any] = {"train_records": len(train), "test_records": len(test)}

    def phrase_rule(records: list[dict[str, Any]]) -> dict[str, Any]:
        hits = [r for r in records if ABSTENTION_PHRASE in r["input"]["context"]]
        abstain = [r for r in records if r["target"]["decision"] == "abstain"]
        return {
            "records_with_phrase": len(hits),
            "abstain_share_among_them": round(sum(1 for r in hits if r["target"]["decision"] == "abstain") / len(hits), 3) if hits else None,
            "abstain_recall_of_rule": round(sum(1 for r in abstain if ABSTENTION_PHRASE in r["input"]["context"]) / len(abstain), 3) if abstain else None,
        }

    result["abstention_phrase"] = {"train": phrase_rule(train), "test": phrase_rule(test)}

    def binary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in records if r["target"]["decision"] in {"finding", "no_finding"}]

    result["surface_features_nb_all_decisions"] = surface_probe(train, test)
    result["security_comment"] = {
        decision: round(sum(1 for r in group if security_comment(r["input"]["code"])) / len(group), 3)
        for decision, group in _by_decision(test).items()
    }
    fit, score = binary(train), binary(test)
    result["surface_features_nb_finding_vs_no_finding"] = surface_probe(fit, score)
    base = Counter(r["target"]["decision"] for r in score)
    result["finding_vs_no_finding_test_majority_rate"] = round(max(base.values()) / len(score), 3) if score else None

    if fit and score:
        lines = lambda r: len(r["input"]["code"].splitlines())
        best = max(
            ((threshold, direction) for threshold in sorted({lines(r) for r in fit}) for direction in ("finding_if_longer", "finding_if_shorter")),
            key=lambda option: _threshold_accuracy(fit, lines, *option),
        )
        result["excerpt_length"] = {"threshold": best[0], "rule": best[1], "test_accuracy": round(_threshold_accuracy(score, lines, *best), 3)}

        multi = [r for r in score if len(r["provenance"]["excerpt_file_lines"]) > 1]
        result["multi_block_excerpt"] = {
            "test_records": len(multi),
            "no_finding_share": round(sum(1 for r in multi if r["target"]["decision"] == "no_finding") / len(multi), 3) if multi else None,
        }

        for name, field in (("code_tokens_nb", "code"), ("context_tokens_nb", "context")):
            model = NaiveBayes([(_tokens(r["input"][field]), r["target"]["decision"]) for r in fit])
            correct = sum(1 for r in score if model.predict(_tokens(r["input"][field])) == r["target"]["decision"])
            pairs = _pairs(score)
            margins = [(_margin(model, _tokens(v["input"][field])), _margin(model, _tokens(f["input"][field]))) for v, f in pairs]
            differing = [(a, b) for a, b in margins if a != b]  # identical inputs (e.g. the same context) cannot be ranked
            result[name] = {
                "test_accuracy": round(correct / len(score), 3),
                "test_pairs": len(pairs),
                "pairs_with_different_input": len(differing),
                "pair_accuracy": round(sum(1 for a, b in differing if a > b) / len(differing), 3) if differing else None,
            }
    return result


_COMMENT = re.compile(r"^\s*(#|//|/\*|\*)")
_SECURITY_WORD = re.compile(r"(?i)secur|vulnerab|sanitiz|escap|inject|travers|xss|csrf|ssrf|prevent|attack|malicious|untrusted|cve-|ghsa-|exploit|bypass|validat")


def security_comment(code: str) -> bool:
    """A comment line that talks about security (fixes often add one, e.g. "// prevent path traversal")."""
    return any(_COMMENT.match(line) and _SECURITY_WORD.search(line) for line in code.splitlines())


def _bucket(value: int, edges: tuple[int, ...]) -> int:
    return sum(value > edge for edge in edges)


def surface_features(record: dict[str, Any]) -> list[str]:
    """Features that say nothing about the code's semantics: if they predict the label, it is a shortcut."""
    code_text, context = record["input"]["code"], record["input"]["context"]
    lines = code_text.splitlines()
    first = next((line for line in lines if line.strip()), "")
    return [
        f"lines:{_bucket(len(lines), (10, 20, 40, 80))}",
        f"blocks:{min(len(record['provenance']['excerpt_file_lines']), 3)}",
        f"indent:{_bucket(len(first) - len(first.lstrip()), (0, 4))}",
        f"callee_note:{CALLEE_NOTE in context or ABSTENTION_PHRASE in context}",
        f"caller_note:{'Called in this file by' in context}",
        f"security_comment:{security_comment(code_text)}",
        f"context_chars:{_bucket(len(context), (200, 400, 700))}",
        f"blank_lines:{_bucket(sum(1 for line in lines if not line.strip()), (0, 3, 8))}",
    ]


def source_of(record: dict[str, Any]) -> str:
    """Garden v6: where a record's label comes from (GHSA/OSV advisory, CVE via MoreFixes, CodeQL tool test)."""
    provenance = record["provenance"]
    if provenance.get("source_kind") == "tool_test":
        return "codeql_tool_test"
    return {"nvd_reference": "morefixes_nvd", "ghsa_reference": "morefixes_ghsa", "commit_message_reference": "morefixes_commit_message"}.get(
        provenance.get("fix_evidence") or "", "ghsa_osv"
    )


def source_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Can surface features tell a tool test from real-world code (two folds by record id)? If they can and the
    sources' decision rates differ, the source identity is a label cue."""
    tool = [r for r in records if source_of(r) == "codeql_tool_test"]
    if not tool:
        return {}
    labelled = [(r, "tool" if source_of(r) == "codeql_tool_test" else "real") for r in records]
    folds = ([x for x in labelled if x[0]["record_id"] < "sg-m"], [x for x in labelled if x[0]["record_id"] >= "sg-m"])
    correct = total = 0
    for fit, score in (folds, folds[::-1]):
        if not fit or not score:
            continue
        model = NaiveBayes([(surface_features(r) + [f"lang:{r['input']['language']}"], y) for r, y in fit])
        correct += sum(1 for r, y in score if model.predict(surface_features(r) + [f"lang:{r['input']['language']}"]) == y)
        total += len(score)
    rates = {
        source: round(sum(1 for r in group if r["target"]["decision"] == "finding") / len(group), 3)
        for source, group in sorted(_group(records, source_of).items())
    }
    return {"tool_vs_real_accuracy": round(correct / total, 3) if total else None, "tool_share": round(len(tool) / len(records), 3), "finding_rate_by_source": rates}


def _group(records: list[dict[str, Any]], key: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[key(record)].append(record)
    return out


def surface_probe(fit: list[dict[str, Any]], score: list[dict[str, Any]]) -> dict[str, Any]:
    if not fit or not score:
        return {}
    model = NaiveBayes([(surface_features(r), r["target"]["decision"]) for r in fit])
    correct = sum(1 for r in score if model.predict(surface_features(r)) == r["target"]["decision"])
    majority = Counter(r["target"]["decision"] for r in score).most_common(1)[0][1]
    by_decision = {}
    for decision in sorted({r["target"]["decision"] for r in score}):
        group = [r for r in score if r["target"]["decision"] == decision]
        by_decision[decision] = round(sum(1 for r in group if model.predict(surface_features(r)) == decision) / len(group), 3)
    return {"test_accuracy": round(correct / len(score), 3), "majority_rate": round(majority / len(score), 3), "recall_by_decision": by_decision}


def _threshold_accuracy(records: list[dict[str, Any]], measure: Any, threshold: int, direction: str) -> float:
    def predict(record: dict[str, Any]) -> str:
        longer = measure(record) > threshold
        return "finding" if longer == (direction == "finding_if_longer") else "no_finding"

    return sum(1 for r in records if predict(r) == r["target"]["decision"]) / len(records)


def _margin(model: NaiveBayes, tokens: list[str]) -> float:
    return model.score(tokens, "finding") - model.score(tokens, "no_finding")


def _pairs(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("pair_id"):
            by_pair[record["pair_id"]][record["kind"]] = record
    return [(p["vulnerable"], p["post_fix"]) for _, p in sorted(by_pair.items()) if "vulnerable" in p and "post_fix" in p]


def run_errors(artifact: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Error analysis of one `run_model` artifact on the cases it was run on."""
    confusion: Counter[tuple[str, str]] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    for row in artifact["predictions"]:
        case = cases.get(row["case_id"])
        expected, predicted = row["expected_decision"], row["prediction"]["decision"]
        confusion[(expected, predicted)] += 1
        language = case["language"] if case else "unknown"
        by_language[language]["cases"] += 1
        by_language[language]["correct"] += expected == predicted
        if expected == "finding":
            family = family_of((case or {}).get("ground_truth", {}).get("cwe"))
            by_family[family]["findings"] += 1
            by_family[family]["detected"] += predicted == "finding"
            by_family[family]["abstained"] += predicted == "abstain"
        elif expected == "no_finding" and predicted == "finding":
            by_language[language]["false_positives"] += 1
    return {
        "run": artifact.get("run_id") or artifact.get("model", {}).get("id"),
        "adapter": (artifact["model"].get("adapter") or {}).get("path"),
        "confusion_expected_to_predicted": {f"{e}->{p}": n for (e, p), n in sorted(confusion.items())},
        "recall_by_family": {f: {**dict(c), "recall": round(c["detected"] / c["findings"], 3)} for f, c in sorted(by_family.items())},
        "by_language": {k: {**dict(c), "accuracy": round(c["correct"] / c["cases"], 3)} for k, c in sorted(by_language.items())},
    }


def analyze(records: list[dict[str, Any]], runs: list[dict[str, Any]] | None = None, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    accepted = [r for r in records if r["quality"]["status"] == "accepted"]
    approved = [r for r in accepted if r["governance"]["approval_status"] == "training_approved"]
    train = [r for r in accepted if r["split"] == "train"]
    validation = [r for r in accepted if r["split"] == "validation"]
    roles = sorted({r.get("evaluation_role") for r in accepted if r.get("evaluation_role")})
    result: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset_id": records[0]["dataset_id"] if records else None,
        "accepted": composition(accepted),
        "training_approved": composition(approved),
        "by_split": {split: composition([r for r in accepted if r["split"] == split]) for split in sorted({r["split"] for r in accepted})},
        "shortcut_probes": shortcut_probes(train, validation),
        "rejections": dict(Counter(c.split(":")[0] for r in records if r["quality"]["status"] == "rejected" for c in r["quality"]["failed_checks"]).most_common(25)),
    }
    sources = _group(accepted, source_of)
    if set(sources) - {"ghsa_osv"}:  # Garden v6: several label sources
        result["by_source"] = {source: composition(group) for source, group in sorted(sources.items())}
        result["source_probe"] = source_probe(train)
    if roles:
        result["by_evaluation_role"] = {role: composition([r for r in accepted if r.get("evaluation_role") == role]) for role in roles}
    if runs:
        by_id = {c["case_id"]: c for c in cases or []}
        result["development_runs"] = [run_errors(run, by_id) for run in runs]
    else:
        result["development_runs"] = "not available: no run artifacts were given (they live on the GPU machine, not in this repository)"
    return result


def write(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


EVALUATION_ROLES = {"retired_development", "development", "previous_holdout", "final_test", "temporal_quarantine"}


def contamination_report(
    records: list[dict[str, Any]],
    earlier_manifests: dict[str, list[dict[str, Any]]],
    earlier_records: dict[str, list[dict[str, Any]]],
    near_duplicate: float = 0.7,
    containment: float = 0.8,
) -> dict[str, Any]:
    """Cross-version contamination audit of the accepted training-role records of a v4 dataset.

    Compared with every evaluation set any version exported (earlier development set, holdout
    reservations, the v4 development and final test sets, temporal quarantine): repository and split
    history, advisory and CVE identity, fix and vulnerable revisions (forks share commits), exact
    normalized code, fix fingerprint (the same fix in another repository), shared packages (moved,
    renamed or forked repositories), near-duplicate and contained code. Counts are after curation:
    every entry other than the history is expected to be zero."""
    from sentinel.garden import code
    from sentinel.garden.audit import patch_fingerprint
    from sentinel.garden.quality import CONTAINMENT_MIN_SHINGLES

    accepted = [r for r in records if r["quality"]["status"] == "accepted"]
    training = [r for r in accepted if r.get("evaluation_role") == "training"]
    held = [r for r in accepted if r.get("evaluation_role") in EVALUATION_ROLES]
    for version, rows in sorted(earlier_manifests.items()):
        held_splits = {"validation", "holdout_reserved", "temporal_quarantine"}
        held += [{**row, "_version": version} for row in rows if row["quality"]["status"] == "accepted" and row["split"] in held_splits]

    def identities(row: dict[str, Any]) -> dict[str, set[str]]:
        p = row["provenance"]
        return {
            "advisory_or_alias": {p["advisory_id"], *p["aliases"]},
            "revision": {p["fixed_revision"], p["vulnerable_revision"]},
            "normalized_code": {p["normalized_sha256"]},
            "package": {f"{p['ecosystem']}:{name}" for name in p["packages"]},
            "repository": {p["repository_key"]},
        }

    held_index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in held:
        quarantine = row.get("evaluation_role", row.get("split")) == "temporal_quarantine"
        for kind, values in identities(row).items():
            if quarantine and kind == "repository":
                continue  # post-cutoff fixes of training repositories, by design (reported by the audit)
            # the audit's package rule protects validation and holdout; the quarantine is not repository-disjoint by design
            key = "package_with_temporal_quarantine" if quarantine and kind == "package" else kind
            for value in values:
                held_index[key][value].add(row["provenance"]["repository_key"])
    shared: Counter[str] = Counter()
    for row in training:
        own = row["provenance"]["repository_key"]
        found = identities(row)
        found["package_with_temporal_quarantine"] = found["package"]
        for kind, values in found.items():
            # a package or revision shared within the same repository is that repository's own history
            if any(held_index[kind][value] - ({own} if kind in {"package", "package_with_temporal_quarantine", "revision"} else set()) for value in values):
                shared[kind] += 1
    fingerprints = {f for f in (patch_fingerprint(r) for r in held if "target" in r) if f}
    shared["fix_fingerprint"] = sum(1 for r in training if patch_fingerprint(r) in fingerprints)

    with_code = [r for r in held if "input" in r] + [r for rows in earlier_records.values() for r in rows if r["quality"]["status"] == "accepted" and r["split"] != "train"]
    prints = [(r["input"]["language"], code.shingles(code.normalized_tokens(r["input"]["language"], r["input"]["code"]))) for r in with_code]
    near = contained = 0
    for row in training:
        mine = code.shingles(code.normalized_tokens(row["input"]["language"], row["input"]["code"]))
        same = [p for language, p in prints if language == row["input"]["language"]]
        if any(code.jaccard(mine, p) >= near_duplicate for p in same):
            near += 1
        elif any(len(p) >= CONTAINMENT_MIN_SHINGLES and len(mine & p) / len(p) >= containment for p in same):
            contained += 1
    shared["near_duplicate_code"], shared["contains_held_out_code"] = near, contained

    history: dict[str, set[str]] = defaultdict(set)
    for version, rows in earlier_manifests.items():
        for row in rows:
            history[row["provenance"]["repository_key"]].add(row["split"])
    moved = sorted({r["provenance"]["repository_key"] for r in training} & {k for k, splits in history.items() if splits & {"validation", "holdout_reserved"}})
    return {
        "training_records": len(training),
        "evaluation_records_compared": len(held),
        "evaluation_records_with_code": len(with_code),
        "training_records_sharing": dict(sorted(shared.items())),
        "training_repositories_previously_in_validation_or_holdout": moved,
        "eval_case_rejections": dict(Counter(c.split(":")[0] for r in records for c in r["quality"]["failed_checks"] if c.startswith(("eval_contamination", "eval_repository_overlap"))).most_common()),
        "audit_leakage_exclusions": dict(Counter(c.split(":")[1].split("_with_")[0].split("_of_")[0] for r in records for c in r["quality"]["failed_checks"] if c.startswith("audit:split_leakage")).most_common()),
    }
