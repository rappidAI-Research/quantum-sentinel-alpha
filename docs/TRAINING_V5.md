# Fresh QLoRA training on Garden v5

Prepared on 2026-11-25; nothing has been trained.

**Owner decision (2026-11-25).**
- Only a fresh adapter from the pinned `ornith-ai/Ornith-1.5-9B@489cb97…` is trained.
- There is no continued training from the first run's adapter: the continuation config of `docs/TRAINING_V4.md` is removed. The trainer's `init_adapter` check stays, but no committed config uses it.
- The first run (`qsa-qlora-ornith15-9b-v0`, `step-000200`) stays as the historical baseline.

**Config:** `configs/train/qlora_ornith15_9b_v5.yaml`.
- Same foundation, LoRA shape (rank 16, alpha 32, 200 modules), sequence length (4,096), prompt contract and seed as the first run.
- Learning rate 1e-4 with 3 % warmup, 2 epochs.
- A checkpoint every 25 steps; all are kept for the development comparison.

## Data the parameters were checked against

`python -m sentinel.train --config configs/train/qlora_ornith15_9b_v5.yaml --dry-run --check-tokens`, with the pinned tokenizer files only:

| | Examples | Detection / patch | Tokens (median / max / total) | Supervised tokens |
| --- | ---: | --- | --- | ---: |
| train | 2,249 | 1,600 / 649 | 802 / 3,064 / 2.14 M | 192,708 |
| validation (retired v3 development set; loss only) | 198 | 141 / 57 | 808 / 2,692 / 0.19 M | 18,243 |

- The table is after the owner's approvals of 2026-11-25 (`docs/TRAINING_AUTHORIZATION.md`).
- 0 examples are dropped at 4,096 tokens, so the sequence length stays.
- 2 epochs at an effective batch of 16 are 282 optimizer steps, against 234 in the first run, whose best checkpoint was at 1.7 epochs. With 20 % more data than that run, nothing argues for more epochs or a larger rank. The parameters therefore stay as they are.
- **Ready to train:** follow `docs/GPU_RUNBOOK.md` § 14.

## Selection and test

1. Rule v1 (`docs/EVALUATION.md`) on the v5 development set: 163 cases.
2. QuickEval v1 and `coding-v1` as a regression check.
3. The final test, once: `final_test` (9 cases) and `previous_holdout` (453 cases), reported separately. `previous_holdout` is only an independent test if the Garden v3 holdout was never evaluated.

The first run's `step-000200` is evaluated on the same sets as the reference. GPU commands: `docs/GPU_RUNBOOK.md` § 14.
