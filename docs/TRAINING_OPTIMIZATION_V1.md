# Training Optimization v1: root cause analysis and controlled experiments

Prepared on 2026-11-26. Nothing was trained, no GPU and no AWS resource was started, no dataset was authorized and no weights were published. The V6 config (`configs/train/qlora_ornith15_9b_v6.yaml`) and every historical artifact are unchanged.

Every statement is labeled:
- **Measured:** computed in this repository from the committed data, configs and code, or from the reported V6 results.
- **Verified:** checked by reading the code and covered by a test.
- **Hypothesis:** consistent with the evidence, not tested yet. Each hypothesis names the experiment that tests it.
- **Assumption (unverified):** taken as true here and to be checked on the GPU machine.

## 1. Summary

1. **No implementation defect explains the swing between steps 150 and 175.** Sample order, shuffling, accumulation, loss normalization, masking, schedule, prompt and template, checkpoint files and labels were audited (§ 3). Every item was verified; nothing was found that differs around steps 126–175.
2. **Four verified limitations make the swing hard to see and impossible to correct.**
   - **The validation loss cannot see decisions.** Decision tokens are 0.78 % of the validation tokens; patch replies are 65 %.
   - **The objective barely weights the decision.** It is one token per detection reply: 0.87 % of the supervised training tokens.
   - **Checkpoints are 25 steps apart.** The swing lies between two checkpoints, and the epoch boundary (after step 173) falls between them too.
   - **There was no way to move the decision threshold** without retraining.

   A fifth limitation is that training ran on an NF4 base while evaluation used bf16. Its effect is not measured.
3. **Leading hypothesis (H1): the decision threshold moved; the ranking may not have.** From step 150 to step 175 the number of flagged cases fell from 97 to 16 of 212. True positives went from 65 to 14 and false positives from 32 to 2. Validation loss improved by 0.003 over the same steps. The four reported operating points are compatible with a threshold shift at unchanged or better discrimination. They cannot tell this apart from lost discrimination (§ 4.2).
4. **The first experiment, E0, needs no training.** It produces decision scores of all 14 V6 checkpoints, their ROC-AUC, pair outcomes and flips, and a bf16 versus NF4 comparison. It also calibrates each checkpoint's finding bias on the training validation split. It costs about 1.5–3 GPU hours and either confirms H1 or rules it out.
5. **Training experiments E1–E4 each change one variable** against V6: denser checkpoints, decision-token weight 10, detection only, or learning rate 5e-5. Each starts from a fresh adapter on the pinned foundation. All of E0–E4 and calibrated candidates were approved by the owner on 2026-11-26.

## 2. Reported V6 results and what follows arithmetically

The run parameters and results are as reported (Ornith-1.5-9B, 2,758 train / 204 validation examples, 346 steps, 2 epochs, LR 1e-4, rank 16, effective batch 16, 2.4 h, peak 21.56 GiB, 14 checkpoints, all reportable, 0 % invalid).

**Measured (from the reported precision, recall and FPR on 113 finding / 99 no_finding development cases).** Every reported triple is consistent with these counts. The confidence intervals are Wilson 95 %.

| Model | TP | FP | Flagged | Recall [95 % CI] | FPR [95 % CI] | Recall − FPR |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Ornith (foundation) | 11 | 4 | 15 | 0.097 [0.055, 0.166] | 0.040 [0.016, 0.099] | 0.057 |
| V3 step 200 (`qsa-qlora-ornith15-9b-v0`) | 29 | 8 | 37 | 0.257 [0.185, 0.344] | 0.081 [0.042, 0.151] | 0.176 |
| V6 step 25 | 58 | 42 | 100 | 0.513 [0.422, 0.603] | 0.424 [0.332, 0.523] | 0.089 |
| V6 step 125 | 31 | 17 | 48 | 0.274 [0.201, 0.363] | 0.172 [0.110, 0.258] | 0.103 |
| V6 step 150 | 65 | 32 | 97 | 0.575 [0.483, 0.662] | 0.323 [0.239, 0.420] | 0.252 |
| V6 step 175 | 14 | 2 | 16 | 0.124 [0.075, 0.197] | 0.020 [0.006, 0.071] | 0.104 |
| V6 step 200 | 19 | 2 | 21 | 0.168 [0.110, 0.248] | 0.020 [0.006, 0.071] | 0.148 |
| V6 step 346 | 19 | 4 | 23 | 0.168 [0.110, 0.248] | 0.040 [0.016, 0.099] | 0.128 |

- Step 125 misses rule v2 by 3 findings (31 instead of 34 of 113) at an FPR of 0.172. Step 150 meets the recall floor but is 12 negatives over the FPR ceiling (32 instead of at most 19 of 99).
- One false positive moves the FPR by 1.0 point. One finding moves recall by 0.9 points. Differences of a few cases between neighbouring checkpoints are within noise; the swing from 97 to 16 flagged cases is not.
- The brief does not list steps 50, 75, 100 and 225–325. The analysis tools read all 14 artifacts.

## 3. Implementation audit

| Item | Result | Evidence |
| --- | --- | --- |
| Data identity | Measured: the local Garden v6 work directory gives 2,758 train examples (2,024 detection + 734 patch) and 204 validation examples (146 + 58), as reported; `data_sha256 51ab4c9d…`, config `f9a0edf6…`. **Assumption (unverified):** the run used exactly this data. Check `run.json` → `data.sha256`. | `sentinel.train.diagnostics` |
| Ordering and shuffling | Verified. The order is recomputed per epoch by sorting on `sha256(seed, epoch, task, record_id)`, which is deterministic and independent of the input order. Measured: the reconstruction gives 173 steps per epoch, 346 in total, epoch boundary after step 173. | `optimizer_steps`, tests |
| Class mix per 25-step window | Measured: finding 34–42 %, no_finding 24–32 %, abstain 6–12 %, patch 23–31 % in every window. Steps 126–150: 42 / 26 / 7 / 26 %; steps 151–175: 39 / 24 / 8 / 29 %. **Nothing unusual before the swing**; steps 151–175 even carry slightly more findings than average. | `diagnostics --window 126-175` |
| Epoch boundary | Measured: after step 173. The last window of each epoch has 6 micro-batches; at step 173 these are 2 finding, 3 no_finding and 1 patch. **The boundary lies between checkpoints 150 and 175.** | same |
| Seeds | Verified: one seed for shuffling, LoRA initialization and dropout. **Assumption:** CUDA kernels are not bit-deterministic, so a rerun with the same seed is close but not identical. | config, trainer |
| Batches and accumulation | Verified: micro-batch 1, accumulation 16, windows aligned to each epoch's start. Each window divides by its own size, so the partial window of 6 is a correct mean, from fewer examples. | `accumulation_window`, tests |
| Loss normalization | Verified: each example's loss is its mean over supervised tokens, then averaged over the window, so every example weighs the same whatever its length. Measured consequence: the decisive token's summed weight is 19.7 for finding, 21.1 for no_finding and 5.4 for abstain (Σ 1/reply length), against 1,047 / 755 / 222 examples. Short no_finding replies give their decision token more weight per example; in total the two classes are nearly balanced. | `diagnostics --tokenizer-dir` |
| Supervised token weighting | Measured: patch replies hold 60.0 % of the supervised training tokens, finding 24.6 %, no_finding 11.6 %, abstain 3.9 %. The median reply is 52 / 36 / 41 tokens and 157.5 for patch. **The decision is one token: 0.87 % of the supervised tokens, a mean 2.28 % of its own example's loss.** | same |
| Assistant masking | Verified: the prompt is masked (`-100`); the reply and the end-of-turn token are supervised. Measured: all 2,024 train and 146 validation detection replies start with `{"decision": "`, and the next token is one id per class: `finding` 64919, `no` 2083, `ab` 365. | `encode_example`, tests |
| Class distribution | Measured. Train detection: 1,047 / 755 / 222. Validation: 70 / 58 / 18. Development: 113 / 99 / 27. 93 of the 99 development negatives are post-fix twins of development findings and 6 are defended siblings. | |
| Detection / patch interference | Not measurable without training. Patch replies are 27 % of the examples, 27 % of all tokens and 60 % of the supervised tokens. Tested by E3. | |
| Optimizer and scheduler | Verified: AdamW and cosine decay with `int(346 × 0.03) = 10` warmup steps. The reconstructed learning rate equals transformers' scheduler at all 346 steps (test). Measured: 6.34e-5 at step 150 and 5.19e-5 at step 175, a smooth decay. The rate at step 1 is 0, as in transformers. | `learning_rate`, test |
| Gradient clipping | Verified: max norm 1.0. **Not verified for steps 126–175:** `run.json` keeps only the last 50 steps. `diagnostics --log runs/train-v6.log` reads every step and counts clipped steps. | |
| Train and evaluation precision | Verified: training uses an NF4 base (double quantization) with bf16 compute; `run_model` evaluates on a bf16 base by default, as in the § 15 commands (**assumption:** the V6 evaluations did so; see § 4.4). The adapter is then applied to weights it was not trained on. **The effect on decisions is not measured**; E0 scores both precisions. | configs, `HFBackend` |
| Checkpoint files | Verified: checksums and `weights_only` loading. Every evaluation compares the loaded LoRA tensors with the adapter file, and all 14 runs were reportable, so the evaluated weights were the saved ones. | `adapter_verification` |
| Prompt and chat template | Measured: for all 385 development and validation records, the training prompt messages equal the evaluation prompt messages exactly (0 differences). Both are rendered with the same `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` and tokenized without special tokens. | check in this PR, `encode_example` |
| Contradictory supervision | Measured: 0 of 2,024 distinct training prompts carry two different decisions. There are 734 complete vulnerable/fixed pairs; the median pair differs in 4 of 25 lines and 41 % differ in at most 3 lines. These pairs are hard by design, not contradictory. | |
| Shortcuts | Measured: 199 tool-test findings (CodeQL) have no negatives in train. **Hypothesis:** tool-test style could be a finding cue; it cannot explain a swing toward no_finding. Earlier shortcut probes: `docs/GARDEN_V4.md`. | |

**Defects found:** none in the V6 training path. One defect in this PR's own interim code was found and fixed before any use. The run schema did not allow `generation.finding_logit_bias`, so a calibrated run would have failed validation after generating every case. A test now covers it.

## 4. Root cause analysis

### 4.1 Verified limitations

1. **The validation loss is blind to decisions.** Measured: patch replies are 65 % of the validation tokens and decision tokens 0.78 %. If the mean loss on the 128 finding/no_finding decision tokens rose by 0.2 nats, the validation loss would move by about 0.0016. That is less than the 0.003 improvement observed from step 150 to step 175. The validation loss could not have shown the swing, and the run logged nothing else.
2. **The objective gives the decision almost no weight** (§ 3). Most of the loss is spent on patch text, CWE, line numbers and evidence. The decision prior can drift as a side effect of those gradients at almost no cost in loss.
3. **The checkpoint cadence is too coarse** to locate the swing, or to tell whether it coincides with the epoch boundary.
4. **The decision threshold could not be moved without retraining.** Greedy decoding fixes the operating point. A checkpoint that ranks well but sits at the wrong threshold could not be used.

### 4.2 What the reported operating points allow, and what they do not

Under an equal-variance binormal model (an **assumption**, unverified for LLM scores), discrimination is d′ = z(recall) − z(FPR):

| | step 125 | step 150 | step 175 | step 200 | step 346 |
| --- | ---: | ---: | ---: | ---: | ---: |
| d′ | 0.35 | 0.65 | 0.89 | 1.09 | 0.78 |
| d′ range from the 95 % intervals | −0.19 to 0.88 | 0.16 to 1.13 | 0.03 to 1.69 | 0.25 to 1.86 | 0.06 to 1.47 |

Under that model the later checkpoints discriminate at least as well as step 150, at a far more conservative threshold. The intervals are wide, mostly because 2 false positives out of 99 is a very uncertain FPR. The numbers support taking **H1** seriously; they do not establish it. Only the scores of every case can (E0).

### 4.3 Hypotheses and how each is tested

| | Hypothesis | Prediction if true | Test |
| --- | --- | --- | --- |
| H1 | The decision threshold moved toward no_finding; the ranking is preserved | From 150 to 175: AUC stable or higher; mean finding score falls; pairs move from "both flagged" to "discriminated" or "both missed"; flips go mostly finding → no_finding, often both members of a pair together. A calibrated bias brings a later checkpoint into rule v2. | E0 |
| H2 | Discrimination collapsed | AUC falls at 175. Calibration does not help. | E0 |
| H3 | The NF4/bf16 precision mismatch shifts decisions | The same checkpoint scored on NF4 versus bf16 shows a systematic shift in mean finding score or decisions. | E0 (4 checkpoints) |
| H4 | Patch supervision interferes with detection | Detection only: higher or steadier AUC and decision accuracy. | E3 |
| H5 | The learning rate is too high in late epoch 1 | LR 5e-5: smaller swings between neighbouring checkpoints. | E4 |
| H6 | The second pass over the data (epoch 2 starts after step 173) shifts the decision prior | With checkpoints every 10 steps, the swing falls between 170 and 180, not between 150 and 170. | E1 |
| H7 | The decision drifts because it carries about 1–2 % of the loss | Decision weight 10: validation decision metrics steadier across checkpoints; AUC ≥ E1. | E2 |

H1 and H2 exclude each other. H3–H7 can hold together, which is why each experiment changes one variable.

### 4.4 Unverified assumptions

- The V6 run used `data_sha256 51ab4c9d…` and config `f9a0edf6…`. Check them in `run.json`.
- The V6 development runs used greedy decoding on a bf16 base without quantization, the `run_model` defaults. Check `runtime` and `generation` in the artifacts.
- A rerun with the same seed reproduces V6 closely but not bit for bit, because of CUDA nondeterminism. E1 measures how closely.
- The calibration set (the training validation split) transfers to the development set. Its source mix differs: 95 % GHSA/OSV against 64 % on the development set (36 % MoreFixes). That is why the development evaluation under rule v2 stays the gate (§ 6).

## 5. What this PR implements, and what it leaves out

**Diagnostics (CPU, no weights).** `python -m sentinel.train.diagnostics` provides:
- the examples of every optimizer step;
- the class mix of any window of steps and the learning rate per step;
- the epoch boundaries;
- token shares and decision-token weight (with `--tokenizer-dir`);
- the logged loss, gradient norm and clipped steps of every step (with `--log`).

**Decision scores (GPU, one forward pass per case).** `python -m sentinel.eval.decision_scores` has four commands:
- `score` reads the log-probabilities of the three decision tokens after the reply prefix. Several adapters share one loaded base model, and `--quantization bnb-4bit` scores on the training precision.
- `analyze` (CPU) reports per checkpoint the AUC, mean scores, pair outcomes, results per language and vulnerability family, and the flips between consecutive checkpoints.
- `calibrate` (CPU) chooses a finding bias on the calibration set only.
- `screen` (CPU) applies a shortlist rule fixed in advance before full generation runs (§ 6).

Scoring refuses final-test and holdout exports.

**Calibrated decoding.**
- `HFBackend` adds the bias to the finding token only at the decision position, and only when greedy decoding would produce the prefix and a decision token there.
- `run_model --calibration` applies a calibration file only to the exact model it was made for: id, revision, precision and adapter weights hash.
- The artifact records the calibration's provenance. A bias without provenance makes the run non-reportable.
- `select_checkpoint` accepts calibrated runs only with provenance, and reports the selected bias for the final test.

**Training options and observation.**
- `objective.decision_token_weight` and `objective.patch_task_weight` change the objective. Without them, or at 1.0, the loss is exactly V6's `model(**inputs).loss` (test).
- Every evaluation now also logs the validation loss per task and the decision accuracy, recall, FPR and finding rate of the validation detection examples, from the same forward pass.

**Configs.** Each changes exactly one variable against V6 (test): `configs/train/qlora_ornith15_9b_v6_to1_{baseline_dense,decision_w10,detection_only,lr5e-5}.yaml`.

**Deliberately not done:**
- **No balanced or stratified sampler.** The window mix is measured as normal (§ 3), and the summed decision weight of the two classes is already near balance.
- **No experiment with intermediate patch weights.** `patch_task_weight` exists for a follow-up only if E3 shows interference.
- **No sweep of the decision weight.** One value, 10, is tested.
- **No calibration on the development set or any test set.**
- **No Garden v7:** no dataset defect was found that only new data could fix.

## 6. Calibration and screening protocol v1 (fixed on 2026-11-26, before any calibrated result)

1. **Calibration set:** the training validation split only. That is 146 detection records, all `retired_development`, with 58 complete pairs and 0 repositories shared with the development set. `decision_scores` refuses records outside the validation split or with a development, final-test, holdout, benchmark or quarantine role.
2. **Bias:** per checkpoint and precision, the bias with the highest recall at FPR ≤ 0.15 on the calibration set. Ties go to higher accuracy, then the smaller |bias|. The margin below 0.20 is deliberate: with 58 calibration negatives, the FPR moves in steps of 1.7 points. A cross-fitted estimate (two folds by pair) is reported with it.
3. **Screening (development set, teacher-forced):** checkpoints with recall ≥ 0.27 and FPR ≤ 0.23, greedy or calibrated, go to full generation. At most 8 per training run, by decision accuracy. Screening never selects.
4. **Selection:** rule v2, unchanged (invalid ≤ 0.02, recall ≥ 0.30, FPR ≤ 0.20, reportable, then decision accuracy). It is applied to `run_model` artifacts, with greedy and calibrated candidates side by side. If nothing is eligible, nothing is selected.
5. **Final test:** once, with the selected checkpoint and, if calibrated, the same calibration file. `final_test` and the reserved holdout are never scored with `decision_scores` and never used to choose a bias, a checkpoint or a training setting.

**Owner decision (2026-11-26):** calibrated candidates are admitted (`docs/TRAINING_AUTHORIZATION.md`). This changes the decoding policy of the evaluation; the thresholds stay the same.

## 7. Experiment matrix

Every training run: a fresh LoRA on `ornith-ai/Ornith-1.5-9B@489cb97…` (never `init_adapter`), V6 data, an evaluation and a checkpoint every 10 steps, all kept.

| ID | Change against V6 | Tests | Steps / checkpoints | Main readouts | Decision after it |
| --- | --- | --- | --- | --- | --- |
| **E0** | none (existing V6 checkpoints, foundation, V3 step 200) | H1, H2, H3 | 0 / 14 existing | AUC and mean finding score per checkpoint; pair outcomes; flips 150 → 175; bf16 vs NF4 for 125, 150, 175, 200; calibrated candidates under rule v2 | AUC stable at 175+ → threshold drift confirmed; calibration is the cheapest fix and training targets stability (E2 first). AUC falls → training changes needed (E2, E3, E4). Precision shift → evaluate on NF4 from then on (a protocol change for the owner). |
| **E1** | eval/save every 10 (`baseline_dense`) | H6; the control for E2–E4 at the same cadence | 346 / 35 | where the swing falls relative to step 173; validation decision metrics per step; closeness to V6 at shared steps | The control. E2–E4 are only compared with E1, never with V6's 14 checkpoints: more checkpoints mean more chances to pass rule v2. |
| **E2** | `decision_token_weight: 10` | H7 | 346 / 35 | validation decision metrics and AUC over steps; CWE accuracy, localization and strict recall must not drop | Better than E1 on rule v2 and steadier → the leading candidate. |
| **E3** | `tasks: [detection]` | H4 | 254 / 26 | as E2 | Confounded by design: fewer steps, and no patch validation loss. A clear gain points to interference; the follow-up would then test `patch_task_weight` < 1, not dropping patch supervision. |
| **E4** | `learning_rate: 5.0e-5` | H5 | 346 / 35 | swing size between neighbouring checkpoints; underfitting | Steadier at equal or better rule v2 results → keep the lower rate. |

**Order** (E0–E4 approved by the owner on 2026-11-26):
1. Run E0 and write its report.
2. Train and evaluate E1, E2, E3 and E4, one after another.
3. Select over all runs under rule v2.

The final test is used once, after all experiments, on the one checkpoint selected over all of them. Every experiment informs the analysis, whichever decision rule applies.

**Why weight 10 for E2.** A finding reply of 52 tokens then gives the decision token 10/61 ≈ 16 % of the example's loss instead of 1.9 %, and a no_finding reply of 36 tokens 10/45 ≈ 22 %. The remaining tokens still dominate, so CWE, lines and evidence keep their supervision.

## 8. Compute estimates

The basis is measured: V6 took 2.4 h for 346 steps with 14 evaluations and 14 checkpoints, with a peak of 21.56 GiB on one L40S. Everything else below is an **estimate** and should be replaced by the timings in the first artifacts.

| Work | Estimate |
| --- | --- |
| E0 decision scores: 16 models × (239 development + 146 calibration cases), one forward pass each, bf16 | ≈ 0.5–1 h, including model loads |
| E0 NF4 scores of 4 checkpoints | ≈ 0.25 h |
| E0 `run_model` with calibration, at most 8 shortlisted | ≈ 1–2 h; per-case generation time is in the V6 artifacts (`efficiency`) |
| E1, E2, E4 training: as V6 plus 21 more evaluations (204 examples) and saves | ≈ 2.8–3.2 h each |
| E3 training: 72.6 % of V6's tokens, 254 steps | ≈ 1.9–2.3 h |
| Screening per training run: 35 checkpoints × 385 cases | ≈ 1–1.5 h |
| Generation for at most 8 shortlisted per run | ≈ 1–2 h |
| **E0–E4 in total** | **≈ 20–25 GPU hours** |

- At USD 1.861 per hour for `g6e.xlarge` On-Demand in `us-east-1`, that is about USD 40–50 plus storage. The rate comes from public price aggregators on 2026-11-26; check it in the AWS console before starting.
- **Memory:** E2's objective computes the cross-entropy only at supervised positions, while the default loss upcasts the logits of every position. Its peak should not exceed V6's. This is an estimate.
- **Disk:** about 0.5 GB per checkpoint, so about 18 GB per 35-checkpoint run. 200 GB holds the foundation, V6 and all four runs.

## 9. Files

| File | Purpose |
| --- | --- |
| `sentinel/train/diagnostics.py` | Order, schedule, token and log diagnostics (CPU) |
| `sentinel/train/{data,config,trainer}.py` | Decision index, objective weights, validation per task and per decision |
| `sentinel/eval/decision_scores.py` | `score`, `analyze`, `calibrate`, `screen` |
| `sentinel/eval/{backends,run_model,runner,select_checkpoint}.py`, `data/schemas/eval_run.schema.json` | Calibrated decoding with provenance |
| `configs/train/qlora_ornith15_9b_v6_to1_*.yaml` | E1–E4 |
| `tests/test_training_optimization.py` | Tests for all of the above |
| `docs/GPU_RUNBOOK.md` § 16 | Commands |
