"""Supervision audit of the training, validation and development records (CPU, deterministic).

    python -m sentinel.train.supervision_audit --workdir artifacts/garden/seed_v6 --output report.json

Measures what the labels teach and where the sets differ (docs/TRAINING_OPTIMIZATION_V2.md § 3):
- length shortcut: how well "shorter code => finding" separates the classes (ROC-AUC), with and without
  tool-test findings, and how often the fixed half of a pair is longer than its vulnerable half;
- localization targets: span size, insertion points (the fix only adds lines, so the target is where a
  check is missing), findings whose fix also adds code outside the function;
- evidence supervision: where the trained explanation comes from (advisory summary, tool message);
- CWE labels: coarse labels, CWEs of the validation/development findings never or rarely trained on;
- contradictory labels: identical normalized code with different decisions, within and across sets;
- composition: kinds, sources, languages, decisions, publication years.

It reads train.jsonl, validation.jsonl and development/records.jsonl only: never a final test or holdout.
Composition and label properties are reported; nothing here is fitted to the development set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

SETS = {"train": "train.jsonl", "validation": "validation.jsonl", "development": "development/records.jsonl"}
RARE = 5


def _lines(record: dict[str, Any]) -> int:
    return record["input"]["code"].count("\n") + 1


def _decision(record: dict[str, Any]) -> str:
    return str(record["target"]["decision"])


def auc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def complete_pairs(records: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    halves: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        if r.get("pair_id") and _decision(r) in {"finding", "no_finding"}:
            halves[r["pair_id"]][_decision(r)] = r
    return [p for _, p in sorted(halves.items()) if len(p) == 2]


def length_shortcut(records: list[dict[str, Any]]) -> dict[str, Any]:
    findings = [r for r in records if _decision(r) == "finding"]
    real = [r for r in findings if r.get("kind") != "query_test_finding"]
    negatives = [-_lines(r) for r in records if _decision(r) == "no_finding"]
    pairs = complete_pairs(records)
    return {
        "auc_shorter_is_finding": auc([-_lines(r) for r in findings], negatives),
        "auc_shorter_is_finding_without_tool_tests": auc([-_lines(r) for r in real], negatives),
        "median_lines": {
            "finding": _median([_lines(r) for r in real]),
            "tool_test_finding": _median([_lines(r) for r in findings if r.get("kind") == "query_test_finding"]),
            "no_finding": _median([-x for x in negatives]),
        },
        "pairs": len(pairs),
        "fixed_half_longer_in_chars": round(sum(1 for p in pairs if len(p["no_finding"]["input"]["code"]) > len(p["finding"]["input"]["code"])) / len(pairs), 4) if pairs else None,
    }


def supervision_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    findings = [r for r in records if _decision(r) == "finding"]
    localization = [r.get("localization") or {} for r in findings]
    spans = [r["target"]["end_line"] - r["target"]["start_line"] + 1 for r in findings if r["target"].get("start_line")]
    n = len(findings)
    return {
        "findings": n,
        "span_kind": dict(sorted(Counter(str(loc.get("span_kind")) for loc in localization).items())),
        "median_target_span_lines": _median(spans),
        "median_excerpt_lines": _median([_lines(r) for r in findings]),
        "fix_adds_code_outside_the_function": round(sum(1 for loc in localization if (loc.get("fix_added_top_level_lines") or 0) > 0) / n, 4) if n else None,
        "context_names_code_defined_elsewhere": round(sum(1 for r in findings if "Defined elsewhere" in r["input"].get("context", "")) / n, 4) if n else None,
        "evidence_source": dict(sorted(Counter(str(r["target"].get("evidence_source")) for r in findings).items())),
        "coarse_cwe": round(sum(1 for r in findings if "coarse_cwe" in r["quality"].get("flags", [])) / n, 4) if n else None,
        "with_alternate_cwes": round(sum(1 for r in findings if r["target"].get("alternate_cwes")) / n, 4) if n else None,
    }


def _cwes(record: dict[str, Any]) -> set[str]:
    return {c for c in [record["target"].get("cwe"), *record["target"].get("alternate_cwes", [])] if c}


def cwe_coverage(train: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    trained = Counter(c for r in train if _decision(r) == "finding" for c in _cwes(r))
    findings = [r for r in other if _decision(r) == "finding"]
    unseen = [r for r in findings if not (_cwes(r) & set(trained))]
    rare = [r for r in findings if _cwes(r) & set(trained) and max(trained[c] for c in _cwes(r) if c in trained) < RARE]
    return {
        "findings": len(findings),
        "never_trained": len(unseen),
        f"trained_fewer_than_{RARE}_times": len(rare),
        "distinct_cwes": len({r["target"].get("cwe") for r in findings}),
    }


def label_conflicts(sets: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for name, records in sets.items():
        for r in records:
            seen[r["provenance"]["normalized_sha256"]].add((name, _decision(r)))
    conflicting = [v for v in seen.values() if len({d for _, d in v}) > 1]
    return {
        "within_a_set": sum(1 for v in conflicting if len({n for n, _ in v}) == 1),
        "across_sets": sum(1 for v in conflicting if len({n for n, _ in v}) > 1),
        "development_code_also_in_train": sum(1 for v in seen.values() if {"development", "train"} <= {n for n, _ in v}),
    }


def composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    findings = [r for r in records if _decision(r) == "finding"]
    return {
        "records": len(records),
        "kind": dict(sorted(Counter(str(r.get("kind")) for r in records).items())),
        "source": dict(sorted(Counter(r["provenance"]["code_source_id"] for r in records).items())),
        "language_decision": dict(sorted(Counter(f"{r['input']['language']}/{_decision(r)}" for r in records).items())),
        "finding_publication_year": dict(sorted(Counter(str(r["provenance"].get("published", ""))[:4] for r in findings).items())),
    }


def report(workdir: Path) -> dict[str, Any]:
    sets = {name: [json.loads(line) for line in (workdir / path).read_text(encoding="utf-8").splitlines() if line.strip()] for name, path in SETS.items()}
    return {
        "workdir": str(workdir),
        "sets": {name: len(records) for name, records in sets.items()},
        "length_shortcut": {name: length_shortcut(records) for name, records in sets.items()},
        "supervision": {name: supervision_profile(records) for name, records in sets.items()},
        "cwe_coverage": {name: cwe_coverage(sets["train"], sets[name]) for name in ("validation", "development")},
        "label_conflicts": label_conflicts(sets),
        "composition": {name: composition(records) for name, records in sets.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True, help="the Garden work directory, e.g. artifacts/garden/seed_v6")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = report(args.workdir)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    for name, values in result["length_shortcut"].items():
        print(f"{name}: AUC(shorter => finding) {values['auc_shorter_is_finding']} (without tool tests {values['auc_shorter_is_finding_without_tool_tests']}), fixed half longer in {values['fixed_half_longer_in_chars']} of {values['pairs']} pairs")
    for name, values in result["supervision"].items():
        print(f"{name}: {values['findings']} findings, span kinds {values['span_kind']}, evidence {values['evidence_source']}, coarse CWE {values['coarse_cwe']}")
    print(f"CWE coverage: {result['cwe_coverage']}; label conflicts: {result['label_conflicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
