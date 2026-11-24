"""Compare adapter checkpoints on the development set and apply the pre-registered selection rule.

    python -m sentinel.eval.select_checkpoint --devset runs/devset/manifest.json \\
        runs/eval/dev/*.json [--patch runs/eval/dev-patch/*.json] [--output runs/eval/dev/selection.json]

Only runs on the development set are accepted: every detection artifact must have been evaluated on
exactly the case set in the devset manifest (`cases.sha256`), so a holdout or QuickEval run is refused
and can never influence the choice. All runs must share model, revision, prompt, parser, scorer and
generation settings. The foundation run (no adapter) is shown as the reference; it is not a candidate.

The rule was fixed before any development result existed (SELECTION_RULE). Patch results are shown
for information and do not enter the rule. Exit codes: 0 a checkpoint was selected, 1 no eligible
checkpoint, 2 runs refused (wrong case set, mismatched contract or model).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from sentinel.eval.runner import load_artifact

SELECTION_RULE_VERSION = "1"
SELECTION_RULE = (
    "eligible: reportable adapter runs on the development set with invalid_rate <= 0.02; "
    "select the highest decision_accuracy; ties: lower false_positive_rate, then higher strict_recall, "
    "then the earlier training step"
)
MAX_INVALID_RATE = 0.02
SHARED = (("model", "id"), ("model", "revision"), ("contract",), ("generation",), ("cases", "sha256"))
_STEP = re.compile(r"step-(\d{6})")

COLUMNS = (
    ("decision acc.", "decision_accuracy"),
    ("FPR", "false_positive_rate"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("F1", "f1"),
    ("strict recall", "strict_recall"),
    ("CWE acc.", "cwe_accuracy"),
    ("loc. top-1", "localization_top1"),
    ("strict pair", "strict_pair_accuracy"),
    ("abstain recall", "explicit_abstention_recall"),
    ("invalid", "invalid_rate"),
)


class Refused(ValueError):
    pass


def _get(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        document = document.get(key) if isinstance(document, dict) else None  # type: ignore[assignment]
    return document


def label(artifact: dict[str, Any]) -> tuple[str, int | None]:
    """'foundation', 'step-000150' (a checkpoint) or 'final' (the run's final adapter), and its step."""
    adapter = artifact["model"].get("adapter")
    if not adapter:
        return "foundation", None
    match = _STEP.search(adapter["path"])
    if match:
        return f"step-{match.group(1)}", int(match.group(1))
    return "final", None


def _correct(artifact: dict[str, Any]) -> dict[str, bool]:
    return {r["case_id"]: r["prediction"]["decision"] == r["expected_decision"] for r in artifact["predictions"]}


def false_positive_analysis(artifact: dict[str, Any], language: dict[str, str]) -> dict[str, Any]:
    """Findings on hard negatives (the false positives) and on abstention cases, by language."""
    fps, on_abstain, by_language = [], [], {}
    for record in artifact["predictions"]:
        if record["prediction"]["decision"] != "finding":
            continue
        if record["expected_decision"] == "no_finding":
            fps.append(record["case_id"])
            key = language.get(record["case_id"]) or "unknown"
            by_language[key] = by_language.get(key, 0) + 1
        elif record["expected_decision"] == "abstain":
            on_abstain.append(record["case_id"])
    return {"false_positives": sorted(fps), "findings_on_abstention_cases": sorted(on_abstain), "false_positives_by_language": by_language}


def select(devset: dict[str, Any], artifacts: list[dict[str, Any]], language: dict[str, str] | None = None) -> dict[str, Any]:
    if not artifacts:
        raise Refused("no run artifacts given")
    expected_cases = devset["detection"]["cases_sha256"]
    for artifact in artifacts:
        if artifact["cases"]["sha256"] != expected_cases:
            raise Refused(
                f"{artifact['cases']['path']} is not the development set (cases sha256 {artifact['cases']['sha256'][:12]}); "
                "holdout and QuickEval runs never enter checkpoint selection"
            )
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        for path in SHARED:
            if _get(artifact, path) != _get(reference, path):
                raise Refused(f"runs differ in {'.'.join(path)}; compare only runs of one model with one contract")
    runs = []
    seen: set[str] = set()
    for artifact in artifacts:
        name, step = label(artifact)
        if name in seen:
            raise Refused(f"two runs for {name}")
        seen.add(name)
        metrics = artifact["metrics"]
        eligible = name != "foundation" and artifact["reportable"] and metrics["invalid_rate"] <= MAX_INVALID_RATE
        runs.append(
            {
                "label": name,
                "step": step,
                "adapter": (artifact["model"].get("adapter") or {}).get("path"),
                "reportable": artifact["reportable"],
                "reportable_blockers": artifact["reportable_blockers"],
                "eligible": eligible,
                "metrics": {key: metrics[key] for _, key in COLUMNS},
                "counts": {key: metrics[key] for key in ("true_positive", "false_positive", "false_negative", "true_negative", "negative_cases", "positive_cases", "abstain_cases")},
                **false_positive_analysis(artifact, language or {}),
                "_correct": _correct(artifact),
            }
        )
    candidates = [r for r in runs if r["eligible"]]

    def key(run: dict[str, Any]) -> tuple[float, float, float, float]:
        m = run["metrics"]
        step = run["step"] if run["step"] is not None else float("inf")  # the final adapter is the latest step
        return (-m["decision_accuracy"], m["false_positive_rate"], -(m["strict_recall"] or 0.0), step)

    selected = min(candidates, key=key) if candidates else None
    # the run's final weights: the final adapter, else the latest checkpoint
    adapters = [r for r in runs if r["label"] != "foundation"]
    final = next((r for r in runs if r["label"] == "final"), None) or (
        max(adapters, key=lambda r: r["step"]) if adapters else None
    )
    foundation = next((r for r in runs if r["label"] == "foundation"), None)
    comparison = None
    if selected and final and selected is not final:
        a, b = selected["_correct"], final["_correct"]
        comparison = {
            "final": final["label"],
            "selected_right_final_wrong": sum(1 for c in a if a[c] and not b[c]),
            "selected_wrong_final_right": sum(1 for c in a if not a[c] and b[c]),
        }
    for run in runs:
        del run["_correct"]
    return {
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_rule": SELECTION_RULE,
        "devset": {"cases_sha256": expected_cases, **{k: devset["detection"][k] for k in ("count", "by_expected_decision")}},
        "model": {"id": reference["model"]["id"], "revision": reference["model"]["revision"]},
        "runs": runs,
        "selected": selected["label"] if selected else None,
        "selected_adapter": selected["adapter"] if selected else None,
        "selected_beats_foundation_decision_accuracy": (
            selected["metrics"]["decision_accuracy"] > foundation["metrics"]["decision_accuracy"] if selected and foundation else None
        ),
        "discordant_cases_vs_final": comparison,
    }


def _patch_rows(paths: list[Path], devset: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result.get("cases_identity") or {}).get("sha256") != devset["patch"]["cases_sha256"]:
            raise Refused(f"{path} is not a patch run on the development patch cases")
        name = "reference" if result.get("backend") == "reference" else label({"model": result.get("model") or {}})[0]
        rows.append({"label": name, **{k: result.get(k) for k in ("applies_rate", "normalized_match_rate", "touches_span_rate", "syntax_ok_rate")}})
    return rows


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devset", type=Path, required=True, help="manifest.json written by sentinel.eval.devset")
    parser.add_argument("runs", nargs="+", type=Path, help="run_model artifacts on the development detection cases")
    parser.add_argument("--patch", nargs="*", type=Path, default=[], help="sentinel.eval.patch results on the development patch cases")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    devset = json.loads(args.devset.read_text(encoding="utf-8"))
    try:
        cases_file = args.devset.parent / "dev.eval_cases.jsonl"
        language = {}
        if cases_file.is_file():
            rows = [json.loads(line) for line in cases_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            language = {row["case_id"]: row["language"] for row in rows}
        result = select(devset, [load_artifact(path) for path in args.runs], language)
        result["patch"] = _patch_rows(args.patch, devset)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    runs = sorted(result["runs"], key=lambda r: (r["label"] != "foundation", r["step"] if r["step"] is not None else float("inf")))
    print(f"development set: {result['devset']['count']} cases {result['devset']['by_expected_decision']}")
    print("run".ljust(14) + "".join(title.rjust(15) for title, _ in COLUMNS) + "  FP/negatives  eligible")
    for run in runs:
        m, c = run["metrics"], run["counts"]
        print(
            run["label"].ljust(14)
            + "".join(_fmt(m[key]).rjust(15) for _, key in COLUMNS)
            + f"  {c['false_positive']}/{c['negative_cases']}".ljust(14)
            + f"  {run['eligible']}"
        )
    for row in result["patch"]:
        print(f"patch {row['label']}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in row.items() if k != "label"))
    print(f"rule v{SELECTION_RULE_VERSION}: {SELECTION_RULE}")
    print(f"selected: {result['selected']} ({result['selected_adapter']})")
    if result["discordant_cases_vs_final"]:
        print(f"case-level vs final adapter: {result['discordant_cases_vs_final']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
