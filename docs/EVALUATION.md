# Evaluation after training

The first QLoRA run (Ornith-1.5-9B, 1,871 examples, 2 epochs, 234 optimizer steps) completed on 2026-11-24. This page fixes how its checkpoints are compared and how the result is tested. It was written before any development-set result existed. Commands: `docs/GPU_RUNBOOK.md` § 12.

## Results so far (QuickEval v1, reported by the GPU session)

| Metric | Foundation | Final adapter |
| --- | ---: | ---: |
| Precision | 0.939 | 0.841 |
| Recall | 0.838 | 1.000 |
| F1 | 0.886 | 0.914 |
| Strict recall | 0.595 | 0.811 |
| CWE accuracy | 0.710 | 0.811 |
| Strict pair accuracy | 0.467 | 0.733 |
| Abstention recall | 0.400 | 0.800 |
| Decision accuracy | 0.779 | 0.884 |
| False positive rate | 0.059 | 0.206 |
| Invalid output rate | 0.012 | 0.000 |
| Coding regression (coding-v1) | 9/16 | 11/16 |

These artifacts are not part of this repository. The adapter run is recorded as `reportable=False`, because the evaluator counted the adapter's FP32 LoRA parameters (40,108,032, exactly the configured LoRA size) as foundation parameters. That check is now split (below). The recorded artifact is not reinterpreted or rewritten. A reportable result needs a new run with the current code.

**QuickEval v1 has been inspected** and the false-positive increase was found on it. It is therefore a regression check from now on, not a basis for selection and not the final independent test.

## Roles of the data

| Set | Source | Size | Used for |
| --- | --- | --- | --- |
| Training | Garden v3 `train` split, approved | 1,331 detection + 540 patch | gradient updates |
| **Development** | Garden v3 `validation` split, approved (`python -m sentinel.eval.devset`) | 133 detection cases (63 finding, 54 hard negative, 16 abstention) + 54 patch cases, 57 repositories | checkpoint selection and any later threshold or decoding decision |
| **Final independent test** | Garden v3 `holdout_reserved` (`holdout_reserved.eval_cases.jsonl`, written by `reconstruct`) | 399 accepted records | evaluated **once**, after the choice is fixed |
| Regression check | QuickEval v1 (86 synthetic cases) and `coding-v1` | 86 / 16 | detect a regression; never used to choose |

- **Separation:** splits are repository-level and frozen (`sentinel-garden-split-v1`). No repository is in two of train, validation and holdout. The development set is built only from exports the training loader has verified (approvals, supervision hashes, `SHA256SUMS`). During training it served only for the validation loss, never for gradient updates or any choice.
- **Holdout caveats:** holdout identities are public, so it is a reservation, not a private test. Its upstream fixes predate Ornith's publication (2026-08-18), so the foundation may have seen them. Adapter-versus-foundation differences on it are still a test our training never touched.
- **Repository level:** "repository-level" means the holdout repositories are disjoint from training. Whole-repository scanning (the Sentinel Engine) does not exist yet and cannot be evaluated.

## What the existing infrastructure measures

All of these come from the unchanged QuickEval scorer (`sentinel/eval/metrics.py`, scorer version 3) on whichever case file is passed.

| Question | Metric |
| --- | --- |
| Secure-code discrimination, hard negatives | `false_positive_rate`, `hard_negative_accuracy`, `pair_accuracy`. Garden hard negatives are the same functions after the verified upstream fix |
| False-positive analysis | per run: the false-positive case IDs and the findings on abstention cases, by language (`select_checkpoint`) |
| CWE classification and localization | `cwe_accuracy`, `localization_top1`, `localization_mean_overlap`, `strict_recall`, `strict_pair_accuracy` |
| Context-dependent abstention | `explicit_abstention_recall`, `abstention_precision`, `abstention_case_finding_rate`, `over_abstention_rate` |
| Output validity | `invalid_rate` |
| Patch generation | `sentinel.eval.patch` on the development patch cases: `applies_rate`, `normalized_match_rate`, `touches_span_rate`, `syntax_ok_rate`. This is offline scoring only; security is not verified |

## Checkpoint selection (rule v1, fixed in advance)

`python -m sentinel.eval.select_checkpoint` accepts only runs on the development detection cases: a holdout or QuickEval run is refused by its case-set hash. It also requires identical model, revision, prompt, parser, scorer and generation settings. Then it applies:

> **Eligible:** reportable adapter runs with `invalid_rate ≤ 0.02`.
> **Select:** the highest `decision_accuracy` (finding, hard negative and abstention cases count alike).
> **Ties:** the lower `false_positive_rate`, then the higher `strict_recall`, then the earlier step.

The foundation run is the reference, not a candidate. The tool also reports:
- whether the selected checkpoint beats the foundation on decision accuracy;
- a case-level comparison with the latest weights (cases only one of the two gets right);
- patch results, for information only.

With 54 hard negatives, one false positive moves the FPR by 1.9 points. Differences of a few cases are within noise, and the case-level counts show how many cases a difference rests on. If the rule selects an earlier checkpoint, it reduces false positives only if that is visible on the development set. Whether it also does so on unseen code is answered by the holdout run, once.

## Adapter evaluation integrity

An adapter evaluation records three separate checks:
- **Foundation parameters:** `runtime.foundation_parameter_dtypes` must be exactly the requested dtype (bfloat16). This check is as strict as before.
- **Adapter parameters:** `runtime.adapter_parameter_dtypes` and `runtime.adapter_verification` record the comparison of every loaded LoRA tensor with `adapter_model.safetensors`: same name, same shape, and the file's dtype (or float32 when PEFT upcasts a half-precision adapter). No tensor may be missing or extra.
- **Checkpoint files:** a checkpoint adapter must match its checkpoint's `checksums.json`.

Any failure keeps the run non-reportable. The combined `parameter_dtypes` is still recorded.

## Garden v4: fresh development and final test sets (2026-11-25)

Results so far (reported by the GPU session):
- Rule v1 selected `step-000200` on the v3 development set: decision accuracy 63.2 %, precision 73.3 %, recall 34.9 %, F1 47.3 %, strict recall 12.7 %, FPR 14.8 %, abstention recall 100 %, invalid 0 %.
- With 63 finding, 54 no_finding and 16 abstain cases, these rates are exactly: 22 of 63 findings detected (8 of them strictly), 8 false positives, 46 hard negatives correct and 16 of 16 abstentions.
- How the 41 missed findings split between `no_finding` and `abstain` is not known without the run artifacts (`runs/eval/dev/step-000200.json`).

That set has now been used for a choice, and its failures inform Garden v4. It is therefore **retired**: from v4 on it serves only for the validation loss, never for a selection. Garden v4 gives every record an `evaluation_role` (`sentinel/garden/pipeline.py`; frozen split policy unchanged, no repository moves):

| Role | What | v4 accepted records | Used for |
| --- | --- | ---: | --- |
| `training` | train split | 2,811 | training (after approval) |
| `retired_development` | validation repositories of the v3 development set | 142 | validation loss only |
| `development` | the other validation repositories (never evaluated) | 176 (59 repositories: finding 87, no_finding 66, abstain 23) | **v4 checkpoint selection**; excluded from training exports whatever their approvals |
| `previous_holdout` | holdout repositories whose cases v1-v3 exported | 431 | final test, reported separately; untouched only if § 12's holdout runs were never made |
| `final_test` | holdout repositories no version exported before | 45 (23 repositories: finding 24, no_finding 21) | **final test**, once |
| `temporal_quarantine` | post-cutoff fixes of train/validation repositories | 442 | reported separately; not repository-disjoint |

- The development set is written in the format of `sentinel.eval.devset` (`artifacts/garden/seed_v4/development/manifest.json`). `select_checkpoint` therefore accepts it unchanged, and still refuses every other case set.
- Checkpoints of several training runs are compared under rule v1 together. They are labelled `<run>/step-NNNNNN`. The final weights for the case-level comparison are those of the selected checkpoint's run.
- The training loader refuses any record whose role is not `training` or `retired_development`.
- `check` recomputes every role from the split and the earlier manifests.

**Limits:**
- With 45 cases, `final_test` is small: one case moves a rate by 2 to 5 points.
- If the v3 holdout was never evaluated, it is the stronger final test (431 cases). Please confirm whether § 12's holdout runs were made.
- Both sets are public reservations like every Garden split. Their upstream fixes may predate the foundation's training data.

## Garden v5 (2026-11-25)

The roles are unchanged and are recomputed for v5.

| Role | Cases |
| --- | --- |
| development (selection) | 163 (47 repositories; finding 78, no_finding 62, abstain 23; no SQL injection finding) |
| `final_test` | 9 (4 repositories) |
| `previous_holdout` | 453 (109 repositories) |
| `temporal_quarantine` | 441 |

**Final test.** `final_test` is too small for a conclusion: v4's 45 cases came mostly from combined excerpts that v5 rejects for lack of evidence. If the Garden v3 holdout was never evaluated, `previous_holdout` is the final test (all four languages, 12 CWE families, abstentions, patch cases). Until that is confirmed, report both separately.

**Difficult negatives.** v5's defended siblings are split like every record. They are training data only after their kind is validated; evaluation-role siblings are evaluation cases either way.
