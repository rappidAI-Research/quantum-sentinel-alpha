# Fresh QLoRA training on Garden v6

Prepared on 2026-11-25; nothing has been trained, no AWS resource was started.

**Owner decision (2026-11-25, unchanged).** Only a fresh adapter from the pinned `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` is trained. No continued training from the first run's adapter; `qsa-qlora-ornith15-9b-v0` (`step-000200`) stays the historical baseline.

**Config:** `configs/train/qlora_ornith15_9b_v6.yaml`. QLoRA NF4 with double quantization, BF16 compute, LoRA rank 16 / alpha 32 / dropout 0.05 on the 200 verified target modules, sequence length 4,096, learning rate 1e-4 with 3 % warmup, 2 epochs, effective batch 16, a checkpoint every 25 steps, all kept. Detection and patch tasks.

## Data the parameters were checked against

`python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6.yaml --dry-run --check-tokens --tokenizer-dir <pinned tokenizer files>`:

| | Examples | Detection / patch | Tokens (median / max / total) | Supervised tokens |
| --- | ---: | --- | --- | ---: |
| train (authorized today) | 2,153 | 1,538 / 615 | 804 / 3,064 / 2.06 M | 184,544 |
| validation (loss only) | 193 | 138 / 55 | 808 / 2,692 / 0.19 M | 17,880 |
| train after every decision-ready approval (projection) | 2,795 | 2,088 / 707 | 757 / 3,064 / 2.54 M | 230,207 |
| train with copyleft repositories as well (projection) | 3,510 | 2,591 / 919 | 769 / 3,286 / 3.20 M | 294,613 |

The projections encode the hypothetical training sets of `data/garden/seed_v6/cohort/approval_scenarios.json` with the same tokenizer and rules; nothing is approved.

**Why the parameters stay as they are.**
- 0 examples are dropped at 4,096 tokens in every scenario (longest 3,286), so the sequence length stays.
- 2 epochs at an effective batch of 16 are 270 optimizer steps today, 350 after the decision-ready approvals and 440 with copyleft repositories (first run: 234 steps; its best checkpoint was at 1.7 epochs). A checkpoint every 25 steps gives 11 to 18 checkpoints, within `keep_last_checkpoints: 20`.
- The data grows by at most 56 % and the composition shifts little: tool tests would be 9–11 % of the train examples, median length falls from 804 to 757 tokens. No measurement argues for a larger rank, a different learning rate or more epochs; changing them would also confound the comparison with the first run.
- **Train only after the owner decisions** in `data/garden/seed_v6/cohort/OWNER_DECISION.md`. Without them the authorized data is 4 % smaller than Garden v5's (the OpenSSF benchmark reservation and the stricter later-fix check), and training now would mostly repeat the v5 data.

## Selection and test

1. **Rule v2** (`docs/EVALUATION.md`, fixed before any v6 result) on the v6 development set: eligible only with invalid rate ≤ 0.02, recall ≥ 0.30 and false-positive rate ≤ 0.20; then the highest decision accuracy. If no checkpoint is eligible, nothing is selected.
2. QuickEval v1 and `coding-v1` as a regression check only.
3. The final test, once: `final_test` and `previous_holdout`, reported separately. `previous_holdout` is only an independent test if the Garden v3 holdout was never evaluated (not confirmed; the repository has no record of a holdout run).
4. External benchmarks are scored separately and never mixed with these function-level results.

The first run's `step-000200` and the foundation are evaluated on the same sets as references. GPU commands, integrity checks and backup: `docs/GPU_RUNBOOK.md` § 15 (AWS EC2 `g6e.xlarge`, NVIDIA L40S 48 GB).
