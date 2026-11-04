# Training Foundation: Qwen3.5-9B

Status: **selected** (2026-11-04). Source of truth: `configs/model/foundation.yaml`, validated by `python -m sentinel.foundation`.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-9B` |
| Revision (immutable HF commit) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| License | Apache-2.0 |
| bf16 weights | 19.3 GB |
| Checkpoint published | 2026-02-27 (HF `createdAt`); revision last modified 2026-03-02 |

Revision, license, architecture and dates were read from the Hugging Face API on 2026-11-04. No weights were downloaded.

## Decision

Masterplan v1.2 (November 2026) selects Qwen3.5-9B as the training foundation. This replaces the planned Qwen3.5-4B vs 9B QuickEval bake-off as a *selection gate*.

**Basis: expected capability, not measured security advantage.** The reasons are:

- more capacity for multi-step code reasoning, CWE discrimination and evidence-backed explanations;
- expected headroom for patch generation and abstention after security fine-tuning;
- fits unquantized (bf16) on one 48 GB L40S for evaluation, and a QLoRA run with a 4-bit base fits the same GPU class;
- Apache-2.0 keeps the open-weight release path compatible.

What this decision does **not** claim:

- No QuickEval or coding-regression result exists for either checkpoint.
- 9B has not been shown to outperform 4B on security tasks.

## Consequences

- Training work (data formatting, QLoRA configuration, calibration) targets `Qwen/Qwen3.5-9B@c202236…`.
- The evaluation infrastructure is unchanged. The foundation baseline (the un-adapted 9B on QuickEval v1 and `coding-v1`) is the reference every adapter must beat, and it runs through the existing path:

  ```bash
  python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b --output runs/qwen35-9b-baseline.json
  ```

  `sentinel.foundation` checks that this bake-off candidate carries exactly the selected model ID and revision.
- The 4B comparison stays available as an **optional** efficiency study (`docs/BAKEOFF.md`). It is not a prerequisite for training.
- Changing the model or revision is a deliberate change: update `configs/model/foundation.yaml`, the bake-off candidate pin, this record and `docs/STATUS.md` together.

## Open evidence

1. Foundation baseline run of the selected checkpoint (QuickEval v1 + `coding-v1`) before any adapter. Requires approved GPU time.
2. A first QLoRA calibration run to confirm memory, throughput and sequence length on the target GPU.
