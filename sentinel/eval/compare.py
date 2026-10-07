"""Side-by-side comparison of two QuickEval run artifacts.

    python -m sentinel.eval.compare runs/qwen35-4b.json runs/qwen35-9b.json

Deliberately produces no combined score or winner: the foundation decision weighs security
precision, false positives, recall, localization, CWE quality, abstention, coding regression,
VRAM, latency and throughput by human judgement. Exit code 2 means the runs are not comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sentinel.eval.runner import load_artifact

# (section, label, artifact path, preferred direction) - direction is a reading aid only.
ROWS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("Security", "precision", ("metrics", "precision"), "higher"),
    ("Security", "false positive rate", ("metrics", "false_positive_rate"), "lower"),
    ("Security", "hard-negative accuracy", ("metrics", "hard_negative_accuracy"), "higher"),
    ("Security", "recall", ("metrics", "recall"), "higher"),
    ("Security", "F1", ("metrics", "f1"), "higher"),
    ("Security", "strict recall (loc+CWE)", ("metrics", "strict_recall"), "higher"),
    ("Security", "vuln/fix pair accuracy", ("metrics", "pair_accuracy"), "higher"),
    ("Security", "strict pair accuracy", ("metrics", "strict_pair_accuracy"), "higher"),
    ("Security", "localization top-1", ("metrics", "localization_top1"), "higher"),
    ("Security", "localization mean overlap", ("metrics", "localization_mean_overlap"), "higher"),
    ("Security", "CWE accuracy", ("metrics", "cwe_accuracy"), "higher"),
    ("Abstention", "abstention recall", ("metrics", "explicit_abstention_recall"), "higher"),
    ("Abstention", "abstention precision", ("metrics", "abstention_precision"), "higher"),
    ("Abstention", "over-abstention rate", ("metrics", "over_abstention_rate"), "lower"),
    ("Abstention", "findings on abstain cases", ("metrics", "abstention_case_finding_rate"), "lower"),
    ("Validity", "invalid output rate", ("metrics", "invalid_rate"), "lower"),
    ("Validity", "decision accuracy", ("metrics", "decision_accuracy"), "higher"),
    ("Counts", "TP", ("metrics", "true_positive"), "higher"),
    ("Counts", "FP", ("metrics", "false_positive"), "lower"),
    ("Counts", "FN", ("metrics", "false_negative"), "lower"),
    ("Counts", "TN", ("metrics", "true_negative"), "higher"),
    ("Efficiency", "peak VRAM (GiB)", ("efficiency", "peak_vram_gib"), "lower"),
    ("Efficiency", "model load (s)", ("efficiency", "model_load_seconds"), "lower"),
    ("Efficiency", "total eval (s)", ("efficiency", "total_eval_seconds"), "lower"),
    ("Efficiency", "mean latency/case (s)", ("efficiency", "mean_case_latency_seconds"), "lower"),
    ("Efficiency", "generated tokens", ("efficiency", "generated_tokens"), "-"),
    ("Efficiency", "tokens/s", ("efficiency", "tokens_per_second"), "higher"),
    ("Coding regression", "status", ("coding_regression", "status"), "-"),
    ("Coding regression", "pass rate", ("coding_regression", "pass_rate"), "higher"),
)

# Must be identical for a fair comparison.
CONTRACT_FIELDS: tuple[tuple[str, ...], ...] = (
    ("cases", "sha256"),
    ("contract", "prompt_sha256"),
    ("contract", "parser_version"),
    ("contract", "scorer_version"),
    ("generation",),
)
# May legitimately differ (hardware/model constraints) but must be visible.
RUNTIME_FIELDS = (
    "device",
    "dtype",
    "quantization",
    "gpu_name",
    "torch_version",
    "transformers_version",
    "fla_kernels",
    "causal_conv1d_kernels",
)


def _get(artifact: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = artifact
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def comparability(a: dict[str, Any], b: dict[str, Any]) -> tuple[list[str], list[str]]:
    problems = [
        f"{'.'.join(path)} differs: {_get(a, path)!r} vs {_get(b, path)!r}"
        for path in CONTRACT_FIELDS
        if _get(a, path) != _get(b, path)
    ]
    notes = [
        f"runtime.{key} differs: {a['runtime'].get(key)!r} vs {b['runtime'].get(key)!r}"
        for key in RUNTIME_FIELDS
        if a["runtime"].get(key) != b["runtime"].get(key)
    ]
    coding_a, coding_b = a["coding_regression"], b["coding_regression"]
    if coding_a["status"] != coding_b["status"]:
        problems.append(f"coding_regression.status differs: {coding_a['status']} vs {coding_b['status']}")
    elif coding_a["status"] == "completed" and coding_a["suite_sha256"] != coding_b["suite_sha256"]:
        problems.append("coding_regression.suite_sha256 differs")
    for label, artifact in (("A", a), ("B", b)):
        if not artifact["reportable"]:
            notes.append(f"run {label} is not reportable: {'; '.join(artifact['reportable_blockers'])}")
    return problems, notes


def disagreements(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, str]]:
    """Cases where exactly one run reached the expected decision."""
    decisions_b = {record["case_id"]: record["prediction"]["decision"] for record in b["predictions"]}
    rows = []
    for record in a["predictions"]:
        case_id, expected = record["case_id"], record["expected_decision"]
        decision_a, decision_b = record["prediction"]["decision"], decisions_b.get(case_id)
        if (decision_a == expected) != (decision_b == expected):
            rows.append({"case_id": case_id, "expected": expected, "a": decision_a, "b": str(decision_b)})
    return rows


def build_report(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    problems, notes = comparability(a, b)
    rows = []
    for section, label, path, direction in ROWS:
        value_a, value_b = _get(a, path), _get(b, path)
        delta = value_b - value_a if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)) else None
        rows.append({"section": section, "metric": label, "a": value_a, "b": value_b, "delta": delta, "prefer": direction})
    language_rows = {
        language: {"a": a["metrics"]["by_language"].get(language), "b": b["metrics"]["by_language"].get(language)}
        for language in sorted(set(a["metrics"].get("by_language", {})) | set(b["metrics"].get("by_language", {})))
    }
    return {
        "a": {"model": a["model"], "created_at": a["created_at"], "reportable": a["reportable"]},
        "b": {"model": b["model"], "created_at": b["created_at"], "reportable": b["reportable"]},
        "comparable": not problems,
        "problems": problems,
        "notes": notes,
        "rows": rows,
        "by_language": language_rows,
        "disagreements": disagreements(a, b) if not problems else [],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None:
        return ""
    return f"{value:+d}" if isinstance(value, int) else f"{value:+.3f}"


def render(report: dict[str, Any]) -> str:
    out: list[str] = []
    for label in ("a", "b"):
        model = report[label]["model"]
        out.append(
            f"{label.upper()}: {model['id']} @ {model['revision'] or 'UNPINNED'} "
            f"[{model['backend']}] reportable={report[label]['reportable']}"
        )
    if report["problems"]:
        out.append("\nNOT COMPARABLE - the evaluation contract differs:")
        out.extend(f"  - {problem}" for problem in report["problems"])
    if report["notes"]:
        out.append("\nNotes:")
        out.extend(f"  - {note}" for note in report["notes"])

    section = None
    out.append(f"\n{'metric':<30}{'A':>12}{'B':>12}{'B-A':>10}  prefer")
    for row in report["rows"]:
        if row["section"] != section:
            section = row["section"]
            out.append(f"[{section}]")
        out.append(
            f"  {row['metric']:<28}{_fmt(row['a']):>12}{_fmt(row['b']):>12}{_fmt_delta(row['delta']):>10}  {row['prefer']}"
        )
    if report["by_language"]:
        out.append("[Decision accuracy by language]")
        for language, values in report["by_language"].items():
            cells = [f"{v['correct']}/{v['cases']}" if v else "n/a" for v in (values["a"], values["b"])]
            out.append(f"  {language:<28}{cells[0]:>12}{cells[1]:>12}")
    if report["disagreements"]:
        out.append(f"\nCases only one run decided correctly ({len(report['disagreements'])}):")
        for row in report["disagreements"]:
            out.append(f"  {row['case_id']:<48} expected={row['expected']:<10} A={row['a']:<10} B={row['b']}")
    out.append("\nNo winner is computed; weigh security, abstention, coding regression and cost explicitly.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two QuickEval run artifacts.")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    report = build_report(load_artifact(args.run_a), load_artifact(args.run_b))
    print(json.dumps(report, indent=2, sort_keys=True) if args.as_json else render(report))
    return 0 if report["comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
