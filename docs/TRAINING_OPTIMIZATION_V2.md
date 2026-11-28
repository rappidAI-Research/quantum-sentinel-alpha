# Training Optimization v2: bottlenecks and a sequential plan

Prepared on 2026-11-28. Nothing was trained, no GPU or AWS resource was started, no weights were published and nothing was merged. Final test and reserved holdout are untouched.

The E0, E2 and E3 artifacts are on the GPU machine and were not available here. Everything this document says about those runs comes from the reported aggregates; § 6 (F0) lists the commands that produce the case-level diagnostics.

Labels used throughout:
- **Measured:** computed in this repository from the committed data and code.
- **Derived:** arithmetic on the reported aggregates (113 finding, 99 no_finding and 27 abstain development cases).
- **Verified:** read in the code and covered by a test.
- **Hypothesis:** consistent with the evidence, not tested.
- **Assumption:** taken as true and to be checked.

## 1. Summary

1. **The main bottleneck is discrimination, not the threshold** (derived; hypothesis until F0 measures real AUCs). Under an equal-variance binormal model, d′ was:
   - 0.78–0.84 for every E2 checkpoint reported;
   - 0.75 for V3 step 200 and 0.78 for V6 step 346;
   - 0.53–0.68 for E3 and 0.50 for E0 step 275 calibrated.

   That is a ROC-AUC of about 0.64–0.72. Recall rose from 9.7 % (foundation) to 41.6 % (E2), but mostly because the operating point moved from FPR 4 % to FPR 15 %. The ranking of vulnerable above non-vulnerable code improved little.
2. **The second bottleneck is explaining a finding** (derived). The share of true positives with an accepted CWE *and* overlapping lines is 29–47 % for every model, the foundation included, with no trend. Training raised how often the model flags vulnerable code, not how often it explains it correctly.

   The supervision explains part of this (measured):
   - All 848 real training findings are explained by the advisory summary, never by the code.
   - 24–28 % of the line targets are insertion points: a 2-line span where a check is missing.
   - 10–12 % of the CWE labels are flagged as coarse.
3. **Calibration moves the threshold; it cannot create discrimination.** E2 step 200 with its calibrated bias (−0.1249) has exactly the aggregates of the uncalibrated checkpoint.

   Hypothesis: no development case lies within 0.125 of the decision boundary, because decision weight 10 makes margins large. Whether the predictions are identical case by case is the first F0 check.
4. **No evidence that patch supervision hurts detection** (derived). E3 (detection only) reaches d′ 0.53–0.68 against 0.50–0.78 for V6/E0. Removing patch supervision alone did not help, and E3's validation recall still fell from 70 % to 26 % between steps 160 and 170.
5. **E2 vs E0 is suggestive, not established** (derived). The gaps are +5.9 points accuracy (z ≈ 1.3, unpaired) and +8.9 points recall (z ≈ 1.4). E2 also had 35 checkpoints to E0's 14, i.e. more chances to pass rule v2. Only E1, the dense V6 rerun, would isolate the effect of decision weight 10 from cadence and run-to-run variance.
6. **The release targets are far out of reach of objective tuning.**
   - Recall 80 % at FPR 10 % on this development set needs d′ ≈ 2.1, i.e. AUC ≈ 0.93, against about 0.72 now.
   - Precision 90 % at recall 80 % allows at most 10 false positives on the 99 negatives.
   - PrimeVul ([Ding et al. 2024](https://arxiv.org/abs/2403.18624)) found code language models, including far larger ones, poor at telling a vulnerable function from its patched version. Our development negatives are almost all such patched twins (93 of 99).

   Assumption: the targets are not reachable with this 9B foundation and function-level data by training changes alone. v2 aims at a measurable gain in discrimination (pre-registered: plateau AUC +0.03) and in stability.
7. **The v2 plan:**
   - **F0:** no training. Re-analyse the v1 artifacts, score E2 on its training precision, average E2 checkpoints exactly.
   - **F1:** E2 plus a pairwise ranking loss on the verified vulnerable/fixed train pairs.
   - **F2:** E2 with LR 5e-5.
   - **F3:** both, only if F1 and F2 each pass.

   Each step has a pre-registered decision rule. F0–F2 take about 12–15 GPU hours, roughly USD 25–30.

## 2. Reported results and what follows from them

Derived from the reported precision, recall and FPR (Wilson 95 % intervals). d′ = z(recall) − z(FPR) under the binormal assumption; strict hits = strict recall × 113.

| Model | TP | FP | Recall [95 % CI] | FPR [95 % CI] | d′ | binormal AUC | strict hits | strict / TP |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Foundation | 11 | 4 | 0.097 [0.06, 0.17] | 0.040 [0.02, 0.10] | 0.45 | 0.62 | 4 | 36 % |
| V3 step 200 | 29 | 8 | 0.257 [0.19, 0.34] | 0.081 [0.04, 0.15] | 0.75 | 0.70 | 11 | 38 % |
| V3 calibrated | 48 | 19 | 0.425 [0.34, 0.52] | 0.192 [0.13, 0.28] | 0.68 | 0.68 | 14 | 29 % |
| V6 step 346 | 19 | 4 | 0.168 [0.11, 0.25] | 0.040 [0.02, 0.10] | 0.78 | 0.71 | 9 | 47 % |
| E0 step 275 cal. | 37 | 17 | 0.327 [0.25, 0.42] | 0.172 [0.11, 0.26] | 0.50 | 0.64 | 13 | 35 % |
| E2 step 160 cal. | 54 | 20 | 0.478 [0.39, 0.57] | 0.202 [0.13, 0.29] | 0.78 | 0.71 | | |
| **E2 step 200 cal.** | **47** | **15** | **0.416 [0.33, 0.51]** | **0.152 [0.09, 0.24]** | **0.82** | **0.72** | **16** | **34 %** |
| E2 step 240 cal. | 49 | 17 | 0.434 [0.35, 0.53] | 0.172 [0.11, 0.26] | 0.78 | 0.71 | | |
| E2 step 280 cal. | 48 | 16 | 0.425 [0.34, 0.52] | 0.162 [0.10, 0.25] | 0.80 | 0.71 | | |
| E2 step 330 cal. | 48 | 15 | 0.425 [0.34, 0.52] | 0.152 [0.09, 0.24] | 0.84 | 0.72 | | |
| E3 step 200 cal. | 34 | 14 | 0.301 [0.22, 0.39] | 0.141 [0.09, 0.22] | 0.55 | 0.65 | 14 | 41 % |
| E3 step 230 / 254 | 31 | 10 | 0.274 [0.20, 0.36] | 0.101 [0.06, 0.18] | 0.68 | 0.68 | | |

- **E2 step 200 calibrated** decides 144 of 239 cases correctly:
  - findings: 47 found, 66 missed;
  - negatives: 78 correct, 15 false positives, 6 abstained;
  - abstention cases: 19 of 27 correct.
- **E2 holds a stable operating region.** From step 200 to step 330, recall is 41.6–43.4 % at FPR 15.2–17.2 %. Nothing in the reported points suggests that more steps or another threshold would move it much.
- **What the targets need** under the same model:

  | Target | d′ | AUC |
  | --- | ---: | ---: |
  | Recall 0.80 at FPR 0.10 | 2.12 | 0.93 |
  | Recall 0.50 at FPR 0.10 | 1.28 | 0.82 |
  | Recall 0.60 at FPR 0.15 | 1.29 | 0.82 |
- **The binormal model is an assumption.** LLM decision scores need not be normal or of equal variance. F0 replaces these estimates with ROC-AUCs from the teacher-forced score files, which the v1 sequence already wrote for E0, E2 and E3.

## 3. Data and supervision audit

Measured with `python -m sentinel.train.supervision_audit --workdir artifacts/garden/seed_v6`. It reads train, validation and development only.

| | Train | Validation | Development |
| --- | --- | --- | --- |
| Findings (real + tool tests) | 848 + 199 | 70 | 113 |
| Complete vulnerable/fixed pairs | 734 | 58 | 92 |
| AUC of "shorter code ⇒ finding" | **0.62** (0.55 without tool tests) | 0.53 | 0.53 |
| Fixed half longer than its vulnerable half | 88 % | 84 % | 86 % |
| Line target is an insertion point | 237 (28 % of real findings) | 17 (24 %) | 32 (28 %) |
| Median target span / excerpt length | 2 / 21 lines | 2 / 29 | 2 / 32 |
| Evidence text from the advisory summary | 848 of 848 real (tool tests: query message) | 70 of 70 | 113 of 113 |
| Coarse CWE label | 11 % | 10 % | 12 % |
| Fix also adds code outside the function | 10 % | 14 % | 8 % |
| Context names code defined elsewhere | 30 % | 31 % | 44 % |
| CWE never / rarely (< 5) trained | | 2 / 5 of 70 | 7 / 18 of 113 |
| Source: GHSA/OSV · MoreFixes · CodeQL tests | 79 % · 11 % · 10 % | 95 % · 5 % · 0 | 64 % · 36 % · 0 |
| Identical code with different labels | 0 within or across the three sets | | |

**What this establishes (measured):**
- **No contradictory labels.** No development code appears in train.
- **A length shortcut exists in train only.** The 199 CodeQL tool-test findings are short (median 7 lines) and have no negatives, so in train "short code ⇒ finding" separates the classes better than chance (AUC 0.62). In validation and development it does not (0.53). Whether the model uses this cue is not known.
- **Within a pair, length is not usable as a cue.** The fixed half is longer in 84–88 % of pairs in every set, but a single prompt never shows its twin.
- **The explanation the model learns is the advisory text, not the code.** Every real finding's `evidence` is the advisory summary. That teaches what the vulnerability is called, not where in the code it is.
- **The development set is harder on context and from another source mix.**
  - 44 % of development findings depend on callees outside the excerpt, against 30 % in train.
  - 36 % of the development cases come from MoreFixes repositories, against 5 % of the calibration set.
- **Localization targets are partly ambiguous.** For insertion points the target is where a check is missing.

## 4. Root cause, question by question

| Question | Status | What answers it |
| --- | --- | --- |
| ROC-AUC, precision-recall | derived ≈ 0.64–0.72 (binormal) | F0: `analyze` on the E0/E2/E3 score files |
| Score distributions, calibration transfer | E2-200 calibrated = uncalibrated in aggregate (reported) | F0: case-level `compare` (identical predictions?), mean scores per class, per-source results |
| Margin stability | E3 validation recall 70 % → 26 % within 10 steps (reported) | F0: flips between consecutive checkpoints; F2 tests a lower LR |
| Vulnerable/fixed pair discrimination | 93 of 99 development negatives are fixed twins (measured), so FPR ≈ "flags the fix" | F0: pair outcomes (discriminated, both flagged, both missed, inverted) on development and calibration pairs; F1 targets it |
| Per language, per CWE, per source | composition measured (§ 3) | F0: `analyze` breakdowns |
| False-negative patterns | not available | F0: run-artifact analysis (missed findings by language, family, source) |
| Correct decision vs correct CWE and lines | strict / TP 29–47 %, no trend (derived) | F0: true-positive quality (accepted CWE, same family, lines) |
| NF4 vs BF16 | E0 compared them for V6 only | F0: E2 on `bnb-4bit-dq`, the exact training precision |
| Gradient clipping, optimizer | not available | F0: `diagnostics --log` over all 346 steps of E2 and E3 |
| Teacher-forced vs generated decisions | tools exist (`greedy_matches_generation_conditions`) | F0: score files vs run artifacts of the same checkpoints |
| Train / validation / development mismatch | measured (§ 3): sources, context, length shortcut | per-source results in F0 |

**Conclusion, with confidence:**
1. **Discrimination.** High confidence: every reported point sits on a similar low curve.
2. **Explanation quality, mainly a supervision issue.** Medium confidence; the advisory-summary evidence and insertion-point spans are measured facts.
3. **Calibration transfer.** Low impact: it moves along the curve.
4. **Optimization stability.** Open: real, but its effect on the plateau is unknown.
5. **Foundation capability.** Likely the ceiling for the release targets, not yet for incremental gains.

## 5. What v2 implements, and what it leaves out

**A. Pairwise ranking objective** (`objective.pair_rank_weight`, `sentinel/train/trainer.py`), verified by tests.
- **The loss.** For each verified vulnerable/fixed pair it adds `softplus(−(s_vulnerable − s_fixed))`, where s = logit(finding) − logit(no) at the decision position. It rewards ranking the vulnerable half above its own fix, independently of the global threshold, which calibration already handles. It targets exactly what the FPR measures on this development set.
- **Which pairs.** Only the 734 complete pairs whose halves are both governance-checked train records (`TrainingData.pairs`). Development, final-test and holdout pairs cannot enter: they are not in the training data.
- **Batching.** A pair forms one micro-batch, and an optimizer step holds at least 16 examples, so a pair is never split. F1 has 341 steps against E2's 346.
- **The weight.** At 1.0, a pair's ranking loss is of the order of its two halves' own decision-weighted losses. That is strong enough to act without dominating. One value is tested, not a sweep.
- **The default is unchanged.** With the weight at 0 (every v1 config) the training path is the previous one, and a test pins the loss to V6's.
- **New validation metric.** Validation now reports `pair_ranking`: the share of its 58 pairs ranked correctly, and the mean gap.

**B. Exact checkpoint averaging** (`python -m sentinel.train.average`).
- **The method.** The mean of k LoRA updates is itself a LoRA update of rank k·r: A stacked, B concatenated and divided by k, lora_alpha × k. Nothing is approximated.
- **The windows** are pre-registered: 5 checkpoints (50 steps) ending at steps 200, 250, 300 and 346 of E2.
- **Integrity.** Sources are checksum-verified and recorded, and the result is labelled `avg5-step-NNNNNN`.
- **Why.** It damps the step-to-step swings without any training, and it can be applied to F1/F2 too.

**C. Analysis:**
- `compare` reports how many decisions differ between two runs, whether the predictions are identical, and the exact McNemar p-value of the discordant cases.
- `decision_scores analyze` adds per-source results and, for run artifacts, true-positive quality. It also computes the pre-registered plateau metrics of § 6 per run (`plateau`), with checkpoint averages kept separate.
- `decision_scores cases` writes the calibration set as eval cases, so its pairs can be analysed.

**D. The supervision audit** (§ 3).

**Deliberately not done:**
- **Class-sensitive loss weights.** They move the threshold, which calibration already does. The evidence says the threshold is not the bottleneck.
- **A sweep of decision-token weights.** Each point costs about 5 GPU hours with evaluation. Weight 10 is not known to be optimal, but no evidence points to another value.
- **Separate optimization of classification and structured generation, staged or curriculum training, a reduced patch weight.** Strict/TP is flat across all models, including the foundation, so the limit is what the explanation is trained on, not how the losses are balanced. E3 shows no detection gain without patch supervision. These stay options for after F1/F2.
- **New data, Garden v7.** No defect was found that only new data could fix. The supervision recommendations in § 7 need a dataset change and the owner's approval.

## 6. Sequential plan with decision points

| Step | What | Change against E2 | GPU estimate | Decision |
| --- | --- | --- | --- | --- |
| **F0** | v1 artifacts re-analysed (CPU); E2 on NF4 training precision; exact averages of E2; selection | none (no training) | ≈ 1.5–2.5 h | D0 |
| **F1** | `configs/train/qlora_ornith15_9b_v6_to2_pair_rank.yaml` | pair ranking weight 1.0 | ≈ 2.9–3.2 h train + 2–3 h evaluation | D1 |
| **F2** | `configs/train/qlora_ornith15_9b_v6_to2_lr5e-5.yaml` | LR 5e-5 | ≈ 2.8 h train + 2–3 h evaluation | D2 |
| **F3** (conditional) | `configs/train/qlora_ornith15_9b_v6_to2_pair_rank_lr5e-5.yaml` | both | ≈ 5–6 h | D3 |
| E1 (optional) | `configs/train/qlora_ornith15_9b_v6_to1_baseline_dense.yaml` | the V6 control | ≈ 5–6 h | only to claim the effect of decision weight 10 |

E2 is the reference: same data, seed, schedule and checkpoint cadence, so F1–F3 compare with it directly.

**Pre-registered metrics, fixed on 2026-11-28 before any v2 result:**
- **Plateau AUC:** the mean teacher-forced development ROC-AUC (finding vs no_finding, bf16) over all checkpoints from step 180 on, i.e. the second epoch. Averaging over the plateau removes most of the checkpoint-selection noise.
- **Pair discrimination:** development pairs with the vulnerable half flagged and the fix not, at each checkpoint's greedy decision, averaged over the same checkpoints.
- **Swing:** the mean number of changed development decisions between consecutive checkpoints from step 100 on.
- **Selection:** unchanged rule v2 (recall ≥ 0.30, FPR ≤ 0.20, invalid ≤ 0.02, reportable), greedy and calibrated candidates.

**Decision points:**
- **D0, after F0:**
  - **True plateau AUC of E2.** At or below 0.80 confirms the discrimination bottleneck: run F1, then F2. At or above 0.85 points to transfer or threshold instead; re-examine the calibration set per source first.
  - **NF4 vs bf16.** If E2's AUC differs by ≥ 0.02, or ≥ 5 % of decisions differ, propose evaluating on `bnb-4bit-dq`. That is a protocol change for the owner.
  - **Averages.** An averaged E2 checkpoint is a rule-v2 candidate like any other. If the averages' plateau is ≥ the checkpoints', average F1 and F2 the same way.
- **D1, F1 vs E2:**
  - Adopt the pair objective if plateau AUC ≥ E2 + 0.03, pair discrimination ≥ E2 + 4 pairs (of 92), and F1's selected rule-v2 candidate is not worse than E2's by more than 2 points of accuracy.
  - Stop the pair line if plateau AUC < E2 + 0.01. No second weight is tried.
- **D2, F2 vs E2:** adopt LR 5e-5 if the swing falls by ≥ 30 % and plateau AUC ≥ E2 − 0.01.
- **D3:** run F3 only if D1 and D2 both adopt.

**Stopping criterion for v2.** If none of F1–F3 raises plateau AUC by ≥ 0.03 over E2, the ceiling is data and foundation, not the objective. Further GPU time on objective tuning is then not justified, and the next step is § 7.

**The final test** runs once, after the plan ends, on the checkpoint selected over V6, v1 and v2 runs: `scripts/to2_gpu.sh final`. It is refused if v1 already ran it.

## 7. Data and foundation: recommendations

None of these is implemented; each needs the owner's decision.

1. **Code-grounded evidence** (supported by § 2 and § 3).
   - **What:** replace the advisory-summary `evidence` of real findings with a deterministic, diff-derived statement, e.g. "line 2 calls `gcm.Open` without the nonce-length check the fix adds". No LLM is involved.
   - **Why:** the explanation quality (strict/TP) has not moved with any training so far, and the current evidence never refers to the code.
   - **Cost:** it changes the supervision hashes, so it means a new `curate` and the owner's approval of the changed targets.
   - **When:** after F1/F2, so that its effect is not confounded.
2. **Tool-test negatives** (supported by § 3). The CodeQL tool-test negatives that wait for the independent human review would remove the train-only length shortcut. A cheaper check is an ablation without tool tests, which takes a training run.
3. **Insertion-point targets.** Report strict recall separately for insertion points and changed lines before changing the scoring. Widening their span would be an evaluation-protocol change.
4. **Calibration-set mix.** If F0's per-source results show the MoreFixes part of the development set behaving differently, a calibration set with MoreFixes repositories needs a new split, i.e. a dataset change.
5. **Foundation.** If F1–F3 leave plateau AUC below 0.80, the remaining levers are larger than v2:
   - a larger foundation (a new bake-off);
   - callee context in the prompt (44 % of development findings depend on it);
   - substantially more verified pairs.

   Assumption: the release targets (recall 80 %, precision 90 %, FPR 10 %, strict recall 65 %) are not reachable with Ornith-1.5-9B at function level.

## 8. Compute and cost

Basis (measured in v1):
- E2: 2.78 h for 346 steps, peak 18.72 GiB.
- E3: peak 21.56 GiB.
- Rate: `g6e.xlarge` On-Demand USD 1.861 per hour. This is the public rate on 2026-11-28; check it before starting.

Everything else is an estimate.

| Work | Estimate |
| --- | --- |
| F0 CPU part (analyses, audit, averaging) | minutes |
| F0 GPU: NF4 scores of 4 E2 checkpoints, averages scored on development and calibration sets | ≈ 0.5 h |
| F0 GPU: generation runs of shortlisted averages (≤ 8) | ≈ 1–2 h |
| F1 training: same examples as E2 (pairs batched in twos, 341 steps); peak VRAM ≈ 19–26 GiB, from two sequences per pair micro-batch | ≈ 2.9–3.2 h |
| F2 training | ≈ 2.8 h |
| Evaluation per run: 35 checkpoints scored (dev + calibration), ≤ 8 generation runs | ≈ 2–3 h |
| **F0 + F1 + F2** | **≈ 12–15 GPU h ≈ USD 22–28** |
| F3 (conditional) | ≈ 5–6 h ≈ USD 10–11 |

## 9. Files

| File | Purpose |
| --- | --- |
| `sentinel/train/{data,config,trainer}.py` | Verified train/validation pairs, pair micro-batches, steps counted in examples, the ranking loss, validation pair ranking |
| `sentinel/train/average.py` | Exact average of LoRA checkpoints |
| `sentinel/train/supervision_audit.py` | The audit of § 3 |
| `sentinel/train/diagnostics.py` | Pair-mode step reconstruction |
| `sentinel/eval/compare.py` | Changed decisions, identical predictions, exact McNemar p |
| `sentinel/eval/decision_scores.py` | Per-source results, true-positive quality, `cases`, averaged-adapter labels |
| `sentinel/eval/select_checkpoint.py` | Averaged-adapter labels |
| `configs/train/qlora_ornith15_9b_v6_to2_*.yaml` | F1, F2, F3 |
| `scripts/to2_gpu.sh` | The GPU sequence, one stage at a time (`DRY_RUN=1` prints it) |
| `docs/GPU_RUNBOOK.md` § 17 | Commands and backup |
| `tests/test_training_optimization.py` | Tests for all of the above |
