from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Bump when a metric definition changes; compare refuses to mix scorer versions.
SCORER_VERSION = "3"


@dataclass(frozen=True)
class QuickEvalMetrics:
    """QuickEval security metrics.

    Counting rules:
    - Only `no_finding` cases can produce false positives; findings on abstention cases are
      reported separately (`abstention_case_finding_rate`) because their ground truth is unknown.
    - `invalid` predictions are never correct: they are false negatives on vulnerable cases and
      count against `hard_negative_accuracy`, abstention recall and `invalid_rate`.
    - Localization and CWE accuracy are measured over true positives; `strict_recall` requires
      decision, overlapping location and an accepted CWE at once.
    - `pair_accuracy` credits a vulnerable/secure pair when both decisions are right;
      `strict_pair_accuracy` additionally requires the vulnerable side's CWE and location.
    """

    cases: int
    positive_cases: int
    negative_cases: int
    abstain_cases: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    invalid_predictions: int
    precision: float
    false_positive_rate: float
    hard_negative_accuracy: float | None
    recall: float
    f1: float
    strict_recall: float | None
    localization_top1: float | None
    localization_mean_overlap: float | None
    cwe_accuracy: float | None
    pair_accuracy: float | None
    strict_pair_accuracy: float | None
    explicit_abstention_recall: float | None
    abstention_precision: float | None
    over_abstention_rate: float | None
    abstention_case_finding_rate: float | None
    invalid_rate: float
    decision_accuracy: float
    patch_apply_rate: float | None
    functional_test_pass_rate: float | None
    security_test_pass_rate: float | None
    regression_free_rate: float | None
    by_language: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _line_overlap(expected_start: int, expected_end: int, predicted_start: int, predicted_end: int) -> float:
    intersection = max(0, min(expected_end, predicted_end) - max(expected_start, predicted_start) + 1)
    union = max(expected_end, predicted_end) - min(expected_start, predicted_start) + 1
    return intersection / union if union else 0.0


def accepted_cwes(truth: dict[str, Any]) -> set[str]:
    return {truth["cwe"], *truth.get("alternate_cwes", [])}


def score(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> QuickEvalMetrics:
    by_id = {prediction["case_id"]: prediction for prediction in predictions}
    case_ids = {case["case_id"] for case in cases}
    unknown = set(by_id) - case_ids
    if unknown:
        raise ValueError(f"predictions contain unknown case_ids: {sorted(unknown)}")
    missing = case_ids - set(by_id)
    if missing:
        raise ValueError(f"missing predictions for case_ids: {sorted(missing)}")

    positive = negative = abstain = 0
    tp = fp = fn = tn = invalid = strict_hits = 0
    localization_hits = 0
    localization_overlap_total = 0.0
    cwe_hits = 0
    abstain_hits = abstain_decisions = over_abstain = abstain_case_findings = 0
    patch_attempts = patch_applied = functional_pass = security_pass = regression_free = 0
    # pair_id -> [(decision correct, strictly correct)]; strict adds CWE + location for the vulnerable side.
    pair_results: dict[str, list[tuple[bool, bool]]] = {}
    by_language: dict[str, dict[str, int]] = {}

    for case in cases:
        prediction = by_id[case["case_id"]]
        expected = case["expected_decision"]
        decision = prediction["decision"]
        invalid += int(decision == "invalid")
        abstain_decisions += int(decision == "abstain")
        correct = decision == expected
        strict = correct

        if expected == "finding":
            positive += 1
            over_abstain += int(decision == "abstain")
            if decision == "finding":
                tp += 1
                truth = case["ground_truth"]
                finding = prediction["finding"]
                assert finding is not None
                overlap = _line_overlap(
                    truth["start_line"], truth["end_line"], finding["start_line"], finding["end_line"]
                )
                cwe_ok = finding["cwe"] in accepted_cwes(truth)
                localization_overlap_total += overlap
                localization_hits += int(overlap > 0.0)
                cwe_hits += int(cwe_ok)
                strict = overlap > 0.0 and cwe_ok
                strict_hits += int(strict)
            else:
                fn += 1
        elif expected == "no_finding":
            negative += 1
            over_abstain += int(decision == "abstain")
            if decision == "finding":
                fp += 1
            elif decision == "no_finding":
                tn += 1
        else:
            abstain += 1
            abstain_hits += int(decision == "abstain")
            abstain_case_findings += int(decision == "finding")

        if "pair_id" in case:
            pair_results.setdefault(case["pair_id"], []).append((correct, strict))

        language = by_language.setdefault(case.get("language", "unknown"), {"cases": 0, "correct": 0})
        language["cases"] += 1
        language["correct"] += int(correct)

        verification = prediction["patch_verification"]
        if verification is not None:
            patch_attempts += 1
            patch_applied += int(verification["applied"])
            functional_pass += int(verification["functional_tests_passed"])
            security_pass += int(verification["security_tests_passed"])
            regression_free += int(verification["regression_free"])

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, positive)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    answerable = positive + negative
    complete_pairs = [results for results in pair_results.values() if len(results) == 2]

    return QuickEvalMetrics(
        cases=len(cases),
        positive_cases=positive,
        negative_cases=negative,
        abstain_cases=abstain,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        invalid_predictions=invalid,
        precision=precision,
        false_positive_rate=_safe_ratio(fp, negative),
        hard_negative_accuracy=_optional_ratio(tn, negative),
        recall=recall,
        f1=f1,
        strict_recall=_optional_ratio(strict_hits, positive),
        localization_top1=_optional_ratio(localization_hits, tp),
        localization_mean_overlap=(localization_overlap_total / tp if tp else None),
        cwe_accuracy=_optional_ratio(cwe_hits, tp),
        pair_accuracy=_optional_ratio(sum(all(c for c, _ in pair) for pair in complete_pairs), len(complete_pairs)),
        strict_pair_accuracy=_optional_ratio(
            sum(all(s for _, s in pair) for pair in complete_pairs), len(complete_pairs)
        ),
        explicit_abstention_recall=_optional_ratio(abstain_hits, abstain),
        abstention_precision=_optional_ratio(abstain_hits, abstain_decisions),
        over_abstention_rate=_optional_ratio(over_abstain, answerable),
        abstention_case_finding_rate=_optional_ratio(abstain_case_findings, abstain),
        invalid_rate=_safe_ratio(invalid, len(cases)),
        decision_accuracy=_safe_ratio(sum(lang["correct"] for lang in by_language.values()), len(cases)),
        patch_apply_rate=_optional_ratio(patch_applied, patch_attempts),
        functional_test_pass_rate=_optional_ratio(functional_pass, patch_attempts),
        security_test_pass_rate=_optional_ratio(security_pass, patch_attempts),
        regression_free_rate=_optional_ratio(regression_free, patch_attempts),
        by_language=dict(sorted(by_language.items())),
    )
