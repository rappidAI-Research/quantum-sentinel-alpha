# Training on Garden v4: approaches A and B

Prepared on 2026-11-25. No v4 training has run, and nothing on this page is a measured improvement. Data: `docs/GARDEN_V4.md`. Evaluation and selection: `docs/EVALUATION.md`. GPU commands: `docs/GPU_RUNBOOK.md` § 13.

## Evidence this proposal rests on

| Fact | Source |
| --- | --- |
| The first run (`qsa-qlora-ornith15-9b-v0`) trained Garden v3: 1,871 examples, 2 epochs, 234 steps, LR 1e-4, rank 16. | `configs/train/qlora_ornith15_9b.yaml`, `docs/TRAINING.md` |
| Selection rule v1 picked `step-000200` (1.7 epochs) on the v3 development set. | your report of 2026-11-25 |
| On that set: recall 34.9 %, strict recall 12.7 %, FPR 14.8 %, abstention recall 100 %. | same |
| The abstention recall coincides with a dataset shortcut that v4 removes. | `docs/GARDEN_V4.md` |
| Approved v4 training data: 1,382 detection + 561 patch = **1,943** examples (v3: 1,871, +3.8 %); validation 200. | `python -m sentinel.train --dry-run` |
| Tokens with the pinned tokenizer at 4,096: 0 dropped; median 781, maximum 3,176; 1.86 M tokens per epoch; 167 k supervised. | `--check-tokens` (Ornith tokenizer files only) |

The loss curves, the per-checkpoint development results and the run's `run.json` are on the GPU machine, not in this repository. The hyperparameters below therefore do not use them. Please add them to `runs/` on the next session.

## The two approaches

| | A: continue | B: fresh |
| --- | --- | --- |
| Config | `configs/train/qlora_ornith15_9b_v4_continue.yaml` | `configs/train/qlora_ornith15_9b_v4_fresh.yaml` |
| Start | adapter weights of `step-000200` of the first run | a new LoRA on the pinned foundation |
| Optimizer / schedule | **new** AdamW and cosine schedule; the checkpoint's optimizer, scheduler and RNG state are not loaded | new |
| Learning rate | 5e-5, warmup 5 % | 1e-4, warmup 3 % (as the first run) |
| Epochs | 1 (122 steps) | 2 (244 steps) |
| Rank / alpha / targets / sequence length | 16 / 32 / 200 modules / 4,096 | the same |
| Checkpoints | every 25 steps, all kept | every 25 steps, all kept |

Both runs use Garden v4 (`configs/garden/seed_v4.yaml`). The validation loss is computed on the retired v3 development set: 142 detection and 58 patch examples. Nothing of the fresh v4 development set enters training. Every other setting (foundation, prompt contract, end-of-turn token, seed) equals the first run. So A, B and the first run differ only in starting point, data and schedule.

**Why these values:**
- The first run's best checkpoint was at 1.7 of 2 epochs. That is no evidence that more epochs help, so B keeps 2 epochs.
- A sees the v3 examples a third time, which risks overfitting, so A runs 1 epoch at half the learning rate.
- Rank and alpha stay unchanged. With 1,943 examples, a larger rank is not justified by any measurement.
- Checkpoints are saved every 25 instead of every 50 steps, and all are kept, so the development comparison has enough candidates. Each checkpoint is about 0.5 GB.

**What A's binding checks** (`sentinel/train/trainer.py` `check_init_adapter`, also in `--dry-run`):
1. The checkpoint's `checksums.json` is intact, with no file changed, added or missing.
2. Its provenance names the pinned foundation, the config sha256 of `configs/train/qlora_ornith15_9b.yaml` (`c91cb7c7…`) and the Garden v3 data sha256 (`c47e3616…`, the value the first run's `run.json` records).
3. Its state is step 200.
4. Its LoRA shape equals the config's.

After loading, every tensor is compared with the file. A's own checkpoints record the initial adapter's sha256, so a resume cannot mix starting points.

## Expectations

These are hypotheses, not results:
- **B mainly tests the data change.** It has only 3.8 % more approved examples than v3. A difference from the first run should therefore come from the removed shortcuts and the combined excerpts, not from volume. A lower abstention recall would not be a regression in itself: v3's 100 % rested on a sentence the model could match. Compare abstention precision and the finding recall on the v4 development set.
- **A keeps what the first run learned.** It may also keep the shortcut behaviour. Its development results show whether one epoch on the changed contexts overrides that.
- **Neither approach adds much data.** The largest lever is not a training setting. It is the rights decision on the review queue (`docs/GARDEN_V4.md` § Rights review).
  - 128 repositories with 342 records are prepared and decision-ready.
  - Training after that decision would be on 2,440 examples (1,724 detection + 716 patch; in-memory simulation with the real gate), 26 % more than today. That would be a new curate and a new data hash, so the configs above would not change; only the data does.
  - If you decide before training, train after the decision.

## Selection

Rule v1 applies unchanged on the v4 development set, to the checkpoints of both runs together. `select_checkpoint` labels them `<run>/step-NNNNNN` and refuses runs on any other case set. The first run's `step-000200` and the foundation are evaluated on the same set as references.
