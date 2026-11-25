# Fresh QLoRA training on Garden v6

Prepared on 2026-11-25; nothing has been trained, no AWS resource was started.

**Owner decision (2026-11-25, unchanged).** Only a fresh adapter from the pinned `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` is trained. No continued training from the first run's adapter; `qsa-qlora-ornith15-9b-v0` (`step-000200`) stays the historical baseline.

**Config:** `configs/train/qlora_ornith15_9b_v6.yaml`. QLoRA NF4 with double quantization, BF16 compute, LoRA rank 16 / alpha 32 / dropout 0.05 on the 200 verified target modules, sequence length 4,096, learning rate 1e-4 with 3 % warmup, 2 epochs, effective batch 16, a checkpoint every 25 steps, all kept. Detection and patch tasks.

## Data the parameters were checked against

`python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6.yaml --dry-run --check-tokens --tokenizer-dir <pinned tokenizer files>`:

| | Examples | Detection / patch | Tokens (median / max / total) | Supervised tokens |
| --- | ---: | --- | --- | ---: |
| train (authorized after the owner's approvals of 2026-11-25) | 2,758 | 2,024 / 734 | 775 / 3,064 / 2.56 M | 233,463 |
| validation (loss only) | 204 | 146 / 58 | 805 / 2,692 / 0.20 M | 18,692 |
| train with the two negative kinds validated by the human review (projection) | 2,891 | 2,157 / 734 | 760 / 3,064 / 2.63 M | 238,649 |
| train with copyleft repositories as well (projection) | 3,595 | 2,652 / 943 | 770 / 3,286 / 3.28 M | 302,052 |

The projections encode the hypothetical training sets of `data/garden/seed_v6/cohort/approval_scenarios.json` with the same tokenizer and rules; they are not approved.

**Why the parameters stay as they are.**
- 0 examples are dropped at 4,096 tokens in every scenario (longest 3,286), so the sequence length stays.
- 2 epochs at an effective batch of 16 are 346 optimizer steps now (362 with the validated negative kinds, 450 with copyleft repositories; first run: 234 steps, its best checkpoint at 1.7 epochs). A checkpoint every 25 steps gives 14 to 18 checkpoints, within `keep_last_checkpoints: 20`.
- The data grows by 23 % over Garden v5 (at most 60 % in the projections) and the composition shifts little: tool tests are 7 % of the train examples (199 of 2,758), the median length falls from 802 to 775 tokens. No measurement argues for a larger rank, a different learning rate or more epochs; changing them would also confound the comparison with the first run.
- **Ready to train** after the owner's approvals of 2026-11-25 (`docs/TRAINING_AUTHORIZATION.md`): follow `docs/GPU_RUNBOOK.md` § 15. The human review can still add the two negative kinds later; that would be a new curate and a new data hash, the config stays.

## Selection and test

1. **Rule v2** (`docs/EVALUATION.md`, fixed before any v6 result) on the v6 development set: eligible only with invalid rate ≤ 0.02, recall ≥ 0.30 and false-positive rate ≤ 0.20; then the highest decision accuracy. If no checkpoint is eligible, nothing is selected.
2. QuickEval v1 and `coding-v1` as a regression check only.
3. The final test, once: `final_test` and `previous_holdout`, reported separately. `previous_holdout` is only an independent test if the Garden v3 holdout was never evaluated (not confirmed; the repository has no record of a holdout run).
4. External benchmarks are scored separately and never mixed with these function-level results.

The first run's `step-000200` and the foundation are evaluated on the same sets as references. GPU commands, integrity checks and backup: `docs/GPU_RUNBOOK.md` § 15 (AWS EC2 `g6e.xlarge`, NVIDIA L40S 48 GB).
