"""Calibration metrics for detection confidence measured outside the generated text.

Sentinel's training data cannot supervise a confidence value (labels are upstream-fix facts, not
probabilities), so the model does not write one (prompt `quickeval-review-v2`). A confidence is
instead *measured* from the model and checked against outcomes on data the model never trained on:

1. Score: for each case, the probability the model assigns to each decision, from the log-likelihood
   of the three reply prefixes ``{"decision": "finding"`` / ``"no_finding"`` / ``"abstain"`` after the
   evaluation prompt (`decision_distribution`). No sampling, no text parsing.
2. Event: ``expected_decision == "finding"`` for P(vulnerable); other events (correct decision,
   correct localization) use the same functions with a different boolean.
3. Fit any recalibration (temperature, isotonic) on the Garden validation split only; report Brier
   score, expected calibration error and the reliability table on a separate held-out set
   (holdout_reserved or the private holdout), never on the data used to fit.

See docs/CALIBRATION.md. This module computes metrics; it never produces a confidence by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

DECISIONS = ("finding", "no_finding", "abstain")


def decision_prefixes() -> dict[str, str]:
    """The reply prefixes whose log-likelihoods are compared (the JSON contract's first field)."""
    return {decision: f'{{"decision": "{decision}"' for decision in DECISIONS}


def decision_distribution(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Normalise the three prefix log-likelihoods into a probability distribution (softmax)."""
    if set(logprobs) != set(DECISIONS):
        raise ValueError(f"need log-probabilities for exactly {DECISIONS}, got {sorted(logprobs)}")
    values = [float(logprobs[d]) for d in DECISIONS]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("log-probabilities must be finite")
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    total = sum(weights)
    return {d: w / total for d, w in zip(DECISIONS, weights)}


def _checked(pairs: Iterable[tuple[float, bool]]) -> list[tuple[float, bool]]:
    out = []
    for confidence, outcome in pairs:
        if confidence is None:
            raise ValueError("calibration needs a measured confidence for every case; got None")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
            raise ValueError(f"confidence {confidence} is outside [0, 1]")
        out.append((confidence, bool(outcome)))
    return out


def brier_score(pairs: Iterable[tuple[float, bool]]) -> float | None:
    data = _checked(pairs)
    if not data:
        return None
    return sum((c - float(o)) ** 2 for c, o in data) / len(data)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    observed_rate: float | None


def reliability_bins(pairs: Iterable[tuple[float, bool]], bins: int = 10) -> list[ReliabilityBin]:
    """Equal-width bins over [0, 1]; the last bin includes 1.0."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    data = _checked(pairs)
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, outcome in data:
        groups[min(int(confidence * bins), bins - 1)].append((confidence, outcome))
    result = []
    for index, group in enumerate(groups):
        result.append(
            ReliabilityBin(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=len(group),
                mean_confidence=sum(c for c, _ in group) / len(group) if group else None,
                observed_rate=sum(1 for _, o in group if o) / len(group) if group else None,
            )
        )
    return result


def expected_calibration_error(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> float | None:
    """Count-weighted mean |confidence - observed rate| over the reliability bins."""
    table = reliability_bins(pairs, bins)
    total = sum(b.count for b in table)
    if total == 0:
        return None
    return sum(b.count * abs(b.mean_confidence - b.observed_rate) for b in table if b.count) / total


def calibration_report(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> dict[str, object]:
    data = _checked(pairs)
    return {
        "cases": len(data),
        "positives": sum(1 for _, o in data if o),
        "brier": brier_score(data),
        "ece": expected_calibration_error(data, bins),
        "bins": [vars(b) for b in reliability_bins(data, bins)],
    }
