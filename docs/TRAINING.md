# QLoRA Training

Status: **implemented, not run.** No GPU training has happened and no model weights were downloaded. Training is blocked until Sentinel Garden has training-approved records (`docs/RIGHTS_REVIEW.md`); the trainer refuses to start without them.

Code: `sentinel/train/`. Configuration: `configs/train/qlora_qwen35_9b.yaml`.

## What gets trained

| Item | Value |
| --- | --- |
| Base | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (read from `configs/model/foundation.yaml`; the training config cannot name another model) |
| Loaded class | `Qwen3_5ForCausalLM` via `AutoModelForCausalLM` (text-only; the vision tower and MTP head are not instantiated), 8.95 B parameters |
| Quantization | 4-bit NF4 with double quantization, bf16 compute (bitsandbytes, CUDA). `mode: none` = bf16 LoRA |
| LoRA | rank 16, alpha 32, dropout 0.05, on 200 modules: `q/k/v/o_proj` of the 8 full-attention layers, `in_proj_qkv/in_proj_z/out_proj` of the 24 Gated DeltaNet layers, `gate/up/down_proj` of all 32 MLPs. 40,108,032 trainable parameters (0.45 %) |
| Excluded modules | `linear_attn.in_proj_a/in_proj_b` (4096 → 32 gating projections, too narrow for rank-16 updates), `lm_head`, embeddings, norms, `conv1d`, `A_log`, `dt_bias` |
| Data | `train.chat.jsonl` = Garden `train` split, governance-approved only; validation loss on the Garden `validation` split |
| Format | exactly the evaluation prompt (`quickeval-review-v1`, chat template with `enable_thinking=False`, generation prompt on); loss only on the assistant JSON reply plus `<|im_end|>` |
| Defaults | lr 1e-4 (cosine, 3 % warmup), 2 epochs, micro-batch 1 × 16 accumulation, max_seq_len 4096, gradient checkpointing, grad-norm clip 1.0, seed 1234 |

The module layout was taken from the real configuration, not assumed: `configs/train/qwen35_9b_linear_modules.txt` lists every linear module of the model built from `config.json` at the pinned revision (meta device, no weights). At runtime the trainer counts the modules its regex matches and stops if the count is not 200 (`TargetModuleMismatch`).

## Governance (fail closed)

`sentinel.train.data.load_training_data` refuses to return data unless:

- `train.jsonl` is non-empty and every record is `training_approved` in the committed manifest **and** still passes the governance gate with today's registry and approvals (a revoked approval stops training);
- `train.chat.jsonl` is byte-for-byte the rendering of `train.jsonl`;
- every code excerpt matches its manifest content hash;
- records are in the right split, no repository is in both train and validation, and no record comes from a holdout or quarantine repository;
- the prompt version matches the evaluation prompt.

Examples longer than `max_seq_len` are dropped and listed in `run.json`, never truncated.

## Commands

```bash
pip install -e '.[train]'                       # CUDA host
python -m sentinel.train --dry-run              # config + governance + data checks, no model
python -m sentinel.train                        # train; refuses a dirty Git tree unless --allow-dirty
python -m sentinel.train --resume               # continue from the latest checkpoint
```

Exit codes: 0 completed, 1 failed, 2 governance/data refusal, 3 CUDA out of memory, 4 interrupted.

Outputs in `runs/train/<run_name>/` (Git-ignored):

| Path | Content |
| --- | --- |
| `run.json` | status (`completed`, `failed`, `failed_oom`, `interrupted`), Git commit and dirty flag, config hash, data hash and counts, requested and loaded foundation revision, library/CUDA/GPU versions, seed, schedule, per-step loss / lr / grad norm / tokens per second / peak VRAM, validation losses, elapsed time. Written in every outcome. |
| `checkpoints/step-NNNNNN/` | LoRA weights, optimizer, scheduler and RNG states; written atomically, last 3 kept |
| `adapter/` | final adapter only (`adapter_model.safetensors`, `adapter_config.json` with base revision, `training_run.json`); the trainer refuses to save any tensor that is not a LoRA tensor |

On `CUDA out of memory`, `run.json` records `failed_oom` with a hint: lower `micro_batch_size` (raise `gradient_accumulation_steps`) or `max_seq_len`, keep gradient checkpointing on, and resume with `--resume`.

## Baseline and comparison

Every adapter is compared with the unchanged foundation under the same evaluation contract:

```bash
# 1. foundation baseline (once)
python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b \
    --output runs/eval/qwen35-9b-baseline.json
# 2. adapter on the same base, same cases, prompt and generation settings
python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b \
    --adapter runs/train/qsa-qlora-qwen35-9b-v0/adapter --output runs/eval/qwen35-9b-qlora-v0.json
# 3. side by side (refuses if the evaluation contract differs)
python -m sentinel.eval.compare runs/eval/qwen35-9b-baseline.json runs/eval/qwen35-9b-qlora-v0.json
```

`--adapter` refuses an adapter trained for another base model or revision before anything is downloaded, and the run artifact records the adapter's weight and config hashes. Add `--coding-suite coding-v1` to both runs to measure coding regression; the Garden holdout reservation (`holdout_reserved.eval_cases.jsonl`) can be evaluated with `--model ... --cases ...` for a temporal/generalization check.

## Tests

- `tests/test_train_contracts.py` (runs in CI, no torch): config binding to the foundation, LoRA targets against the recorded module layout, config validation, every governance refusal, prompt masking, drop-not-truncate, deterministic batching, adapter identity checks.
- `tests/test_train_smoke.py` (needs `torch`, `transformers`, `peft`; skipped in CI): trains a tiny randomly initialised `Qwen3_5ForCausalLM` with Gated DeltaNet and full-attention layers on CPU through the real loop, and checks checkpoints, resume, adapter-only output, foundation-revision refusal, target-count refusal and OOM recording.

## Before the first real run

1. Training-approved records exist (`python -m sentinel.train --dry-run` passes).
2. Approved GPU time; foundation baseline run completed.
3. A short calibration run (`training.max_steps: 20`) on the target GPU to confirm memory and throughput at `max_seq_len 4096`; adjust micro-batch/accumulation from `run.json`.
