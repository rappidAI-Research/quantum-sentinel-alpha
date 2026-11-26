"""Compare adapter checkpoints on the development set and apply the pre-registered selection rule.

    python -m sentinel.eval.select_checkpoint --devset runs/devset/manifest.json \\
        runs/eval/dev/*.json [--patch runs/eval/dev-patch/*.json] [--output runs/eval/dev/selection.json]

Only runs on the development set are accepted: every detection artifact must have been evaluated on
exactly the case set in the devset manifest (`cases.sha256`), so a holdout or QuickEval run is refused
and can never influence the choice. All runs must share model, revision, prompt, parser, scorer and
generation settings, except a calibrated finding bias (`run_model --calibration`, chosen per checkpoint on the
training validation split; a bias without that provenance is refused). The foundation run (no adapter) is
shown as the reference; it is not a candidate.

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

SELECTION_RULE_VERSION = "2"
# Rule v1 (Garden v3/v4/v5 runs): decision accuracy alone among valid runs. Rule v2 (fixed on 2026-11-25,
# before any Garden v6 result): decision accuracy may not hide a recall or false-positive failure, so a
# checkpoint is eligible only above a recall floor and below a false-positive ceiling. The floors are the
# first adapter's development results rounded against it (recall 0.349 -> 0.30, FPR 0.148 -> 0.20): a new
# adapter worse than both is not selected, whatever its accuracy.
SELECTION_RULES = {
    "1": {
        "text": (
            "eligible: reportable adapter runs on the development set with invalid_rate <= 0.02; "
            "select the highest decision_accuracy; ties: lower false_positive_rate, then higher strict_recall, "
            "then the earlier training step"
        ),
        "min_recall": None,
        "max_false_positive_rate": None,
    },
    "2": {
        "text": (
            "eligible: reportable adapter runs on the development set with invalid_rate <= 0.02, recall >= 0.30 and "
            "false_positive_rate <= 0.20; select the highest decision_accuracy; ties: lower false_positive_rate, then higher "
            "strict_recall, then the earlier training step; no eligible run: nothing is selected"
        ),
        "min_recall": 0.30,
        "max_false_positive_rate": 0.20,
    },
}
SELECTION_RULE = SELECTION_RULES[SELECTION_RULE_VERSION]["text"]
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


_RUN = re.compile(r"runs/train/([^/]+)/")


def _comparable(value: Any, path: tuple[str, ...]) -> Any:
    """Generation settings must match, except the calibrated finding bias (Training Optimization v1): it is
    chosen per checkpoint on the training validation split, so each checkpoint may carry its own."""
    if path == ("generation",) and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "finding_logit_bias"}
    return value


def label(artifact: dict[str, Any]) -> tuple[str, int | None]:
    """'foundation', 'step-000150' (a checkpoint) or 'final' (the run's final adapter), and its step."""
    adapter = artifact["model"].get("adapter")
    if not adapter:
        return "foundation", None
    match = _STEP.search(adapter["path"])
    if match:
        return f"step-{match.group(1)}", int(match.group(1))
    return "final", None


def training_run(artifact: dict[str, Any]) -> str | None:
    """The training run an adapter comes from (`runs/train/<run>/...`), or None."""
    adapter = artifact["model"].get("adapter")
    match = _RUN.search(adapter["path"]) if adapter else None
    return match.group(1) if match else None


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


def _eligible(metrics: dict[str, Any], rule: dict[str, Any]) -> list[str]:
    """The rule's conditions a run fails (empty = eligible)."""
    failed = []
    if metrics["invalid_rate"] > MAX_INVALID_RATE:
        failed.append("invalid_rate")
    if rule["min_recall"] is not None and (metrics["recall"] is None or metrics["recall"] < rule["min_recall"]):
        failed.append("recall_below_floor")
    if rule["max_false_positive_rate"] is not None and (metrics["false_positive_rate"] is None or metrics["false_positive_rate"] > rule["max_false_positive_rate"]):
        failed.append("false_positive_rate_above_ceiling")
    return failed


def select(devset: dict[str, Any], artifacts: list[dict[str, Any]], language: dict[str, str] | None = None, rule_version: str = SELECTION_RULE_VERSION) -> dict[str, Any]:
    if rule_version not in SELECTION_RULES:
        raise Refused(f"unknown selection rule {rule_version!r}; known: {sorted(SELECTION_RULES)}")
    rule = SELECTION_RULES[rule_version]
    if not artifacts:
        raise Refused("no run artifacts given")
    expected_cases = devset["detection"]["cases_sha256"]
    for artifact in artifacts:
        if artifact["cases"]["sha256"] != expected_cases:
            raise Refused(
                f"{artifact['cases']['path']} is not the development set (cases sha256 {artifact['cases']['sha256'][:12]}); "
                "holdout and QuickEval runs never enter checkpoint selection"
            )
    for artifact in artifacts:
        bias = artifact["generation"].get("finding_logit_bias", 0.0)
        calibration = artifact.get("calibration") or {}
        if bias and (calibration.get("bias") != bias or (calibration.get("cases") or {}).get("kind") != "calibration"):
            raise Refused(
                f"{label(artifact)[0]}: finding_logit_bias {bias} has no calibration on the training validation split "
                "(run_model --calibration); a bias chosen elsewhere never enters checkpoint selection"
            )
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        for path in SHARED:
            if _comparable(_get(artifact, path), path) != _comparable(_get(reference, path), path):
                raise Refused(f"runs differ in {'.'.join(path)}; compare only runs of one model with one contract")
    runs = []
    seen: set[str] = set()
    # checkpoints of several training runs (Garden v4: approaches A and B) are told apart by run name
    several = len({training_run(a) for a in artifacts if a["model"].get("adapter")}) > 1
    for artifact in artifacts:
        name, step = label(artifact)
        if several and name != "foundation":
            name = f"{training_run(artifact)}/{name}"
        bias = artifact["generation"].get("finding_logit_bias", 0.0)
        if bias:
            name = f"{name}+calibrated"
        if name in seen:
            raise Refused(f"two runs for {name}")
        seen.add(name)
        metrics = artifact["metrics"]
        failed = _eligible(metrics, rule)
        eligible = name != "foundation" and artifact["reportable"] and not failed
        runs.append(
            {
                "label": name,
                "run": training_run(artifact),
                "step": step,
                "adapter": (artifact["model"].get("adapter") or {}).get("path"),
                "finding_logit_bias": bias,
                "reportable": artifact["reportable"],
                "reportable_blockers": artifact["reportable_blockers"],
                "eligible": eligible,
                "failed_conditions": failed if name != "foundation" else [],
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
    # the final weights of the selected checkpoint's run: its final adapter, else its latest checkpoint
    adapters = [r for r in runs if r["label"] != "foundation" and (selected is None or r["run"] == selected["run"])]
    final = next((r for r in adapters if r["step"] is None), None) or (
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
        "selection_rule_version": rule_version,
        "selection_rule": rule["text"],
        "devset": {"cases_sha256": expected_cases, **{k: devset["detection"][k] for k in ("count", "by_expected_decision")}},
        "model": {"id": reference["model"]["id"], "revision": reference["model"]["revision"]},
        "runs": runs,
        "selected": selected["label"] if selected else None,
        "selected_adapter": selected["adapter"] if selected else None,
        # calibrated decoding: the final test must be run with this same bias (run_model --calibration)
        "selected_finding_logit_bias": selected["finding_logit_bias"] if selected else None,
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
    parser.add_argument("--rule", default=SELECTION_RULE_VERSION, choices=sorted(SELECTION_RULES), help="selection rule version (default: the current one; 1 reproduces earlier selections)")
    args = parser.parse_args(argv)
    devset = json.loads(args.devset.read_text(encoding="utf-8"))
    try:
        cases_file = args.devset.parent / "dev.eval_cases.jsonl"
        language = {}
        if cases_file.is_file():
            rows = [json.loads(line) for line in cases_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            language = {row["case_id"]: row["language"] for row in rows}
        result = select(devset, [load_artifact(path) for path in args.runs], language, args.rule)
        result["patch"] = _patch_rows(args.patch, devset)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    runs = sorted(result["runs"], key=lambda r: (r["label"] != "foundation", r["run"] or "", r["step"] if r["step"] is not None else float("inf")))
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
    print(f"rule v{result['selection_rule_version']}: {result['selection_rule']}")
    bias = result["selected_finding_logit_bias"]
    print(f"selected: {result['selected']} ({result['selected_adapter']})" + (f", calibrated finding bias {bias}" if bias else ""))
    if result["discordant_cases_vs_final"]:
        print(f"case-level vs final adapter: {result['discordant_cases_vs_final']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
