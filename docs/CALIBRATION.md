# Detection confidence and calibration

## Problem

The first output contract (`quickeval-review-v1`) asked every finding for a `confidence` between 0
and 1, and the Garden training renderer filled it with the constant `0.9` for every vulnerable
record. That number was a formatting convention, not a measurement. Training on it would teach the
model that every reported vulnerability deserves 90 % confidence, whatever the evidence, and a
model-written number of that kind cannot be interpreted as a probability.

Garden labels are facts about upstream fixes (this function was fixed for this CWE). They carry no
per-example probability, and inventing one per example would be worse than a constant.

## Decision

1. **The model does not write a confidence.** Prompt `quickeval-review-v2` (current) asks for the same
   JSON finding as v1 without the `confidence` field. Garden detection replies are rendered without it
   (`sentinel/garden/render.py`, render version `detection-v2`), and `curate` rejects a record whose
   rendered reply would carry one.
2. **The contract stays compatible.** Parser version 3 accepts findings with or without `confidence`.
   A v1-style output with a number parses exactly as before; the value is stored as self-reported and
   uncalibrated, and `null` means "not reported". Predictions and the engine finding schema allow
   `null`. `quickeval-review-v1` is kept byte for byte (its `prompt_sha256` is unchanged and pinned by
   a test), so any artifact produced with it can be reproduced. Runs with different prompts are never
   compared (`sentinel.eval.compare` checks `prompt_sha256`). No reportable model run exists with v1.
3. **Confidence is measured, then calibrated, on independent data** (below). Until that measurement
   exists, Sentinel reports no confidence, and downstream consumers must not substitute a default.

The QuickEval scorer never used `confidence`, so security metrics of earlier and later runs are
computed identically.

## How to measure calibration

`sentinel/eval/calibration.py` provides the metrics; it never produces a confidence by itself.

1. **Score without generation.** For each case, compute the log-likelihood of the three reply
   prefixes `{"decision": "finding"`, `{"decision": "no_finding"` and `{"decision": "abstain"` after
   the evaluation prompt (teacher forcing, same chat template and `enable_thinking: false`), and
   normalise them with `decision_distribution`. P(finding) is the detection confidence. It costs one
   forward pass per prefix and needs no text parsing.
2. **Choose the event.** P(vulnerable) is checked against `expected_decision == "finding"`. Other
   events (decision correct, localization correct given a finding) use the same functions with a
   different boolean outcome.
3. **Fit on validation, report on held-out data.** Fit any recalibration (one temperature on the
   logits, or isotonic regression) on the Garden **validation** split (governance-approved, never
   trained on). Report Brier score, expected calibration error (10 equal-width bins) and the
   reliability table on a **different** held-out set: `holdout_reserved` (repository-disjoint) or the
   private holdout once it exists. Never fit on the set you report, and never fit or report on
   QuickEval v1 alone (86 public cases are too few and are used for model comparison).
4. **Report uncertainty.** With a few hundred held-out cases the reliability of sparse bins is noisy;
   report bin counts and a bootstrap interval for ECE and Brier, and do not interpret bins with fewer
   than 20 cases.
5. **Only then expose it.** A calibrated P(vulnerable) may populate the engine's `confidence`
   (`finding.schema.json`) with the calibration set, method and date recorded alongside it. Without
   that record the field stays `null`.

Status: the metrics and the prefix definition are implemented and tested; no model has been scored,
so no calibration result exists.
