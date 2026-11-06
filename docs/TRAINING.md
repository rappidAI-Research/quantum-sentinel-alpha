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
| Data | `train.chat.jsonl` = Garden `train` split, governance-approved only; validation loss on the Garden `validation` split. `data.tasks: [detection]` by default; adding `patch` also trains patch generation on verified upstream fixes (`train.patch.chat.jsonl`, docs/PATCH_SUPERVISION.md) |
| Format | exactly the evaluation prompt (`quickeval-review-v2`, chat template with `enable_thinking=False`, generation prompt on); loss only on the assistant JSON reply plus `<|im_end|>`. Replies carry no `confidence` (docs/CALIBRATION.md) |
| Defaults | lr 1e-4 (cosine, 3 % warmup), 2 epochs, micro-batch 1 × 16 accumulation, max_seq_len 4096, gradient checkpointing, grad-norm clip 1.0, seed 1234 |

The module layout was taken from the real configuration, not assumed: `configs/train/qwen35_9b_linear_modules.txt` lists every linear module of the model built from `config.json` at the pinned revision (meta device, no weights). At runtime the trainer counts the modules its regex matches and stops if the count is not 200 (`TargetModuleMismatch`).

## Governance (fail closed)

`sentinel.train.data.load_training_data` refuses to return data unless:

- `train.jsonl` / `validation.jsonl` hold exactly the records the committed manifest marks `training_approved` for that split (none added, none dropped), and every record still passes the governance gate with today's registry and approvals (a revoked approval stops training);
- **target integrity**: for every record, the canonical sha256 of its full input (code, context, file, language), its full target (decision, CWE, span, evidence text, evidence source, upstream patch) and its rendered detection example equal the `supervision` hashes in the committed manifest. Editing an explanation, a patch or the prompt context is detected even when the code hash and the label fields are unchanged, and so is a changed rendering (for example a reintroduced confidence value);
- `train.chat.jsonl` (and `train.patch.chat.jsonl`) are byte-for-byte the rendering of the verified `train.jsonl`, every patch example matches its `patch_example_sha256`, and the export files match `SHA256SUMS`;
- records are in the right split, no repository is in both train and validation, and no record comes from a holdout or quarantine repository;
- the prompt version matches the evaluation prompt.

Nothing is repaired: any mismatch is a `GovernanceError` (exit code 2) and `curate` must regenerate the export. Tests: `tests/test_train_integrity.py` edits the exports consistently (records, chat files and checksums regenerated, only the committed manifest untouched) and checks each attack is refused.

Examples longer than `max_seq_len` are dropped and listed in `run.json`, never truncated.

## Commands

```bash
pip install -e '.[train]'                       # CUDA host
python -m sentinel.train --dry-run              # config + governance + data checks, no model
python -m sentinel.train                        # train; refuses a dirty Git tree unless --allow-dirty
python -m sentinel.train --resume               # continue from the latest checkpoint
```

Resume only continues the same run: every checkpoint stores `provenance.json` (base model and revision, config sha256, training-data sha256 and the record-id list hash), and `--resume` refuses a checkpoint whose provenance differs from the current inputs - for example after a revoked approval removed records. Interrupted saves (`step-NNNNNN.tmp`) are never resumed from.

Checkpoint trust boundary: each checkpoint carries `checksums.json` (sha256 of every file) and resume refuses a checkpoint with a missing, extra or changed file. Adapter weights are read from safetensors, and the optimizer/scheduler/RNG state with `torch.load(weights_only=True)`, so a pickle planted in `trainer_state.pt` cannot execute code even if the checksums were rewritten too (run version 2; earlier checkpoints used `weights_only=False` and are not resumable). The loader also compares every exported record's label (decision, CWE, lines, evidence source) with the committed manifest, so an edited export cannot turn an approved `finding` into a `no_finding` example.

Exit codes: 0 completed, 1 failed, 2 governance/data refusal, 3 CUDA out of memory, 4 interrupted.

Outputs in `runs/train/<run_name>/` (Git-ignored):

| Path | Content |
| --- | --- |
| `run.json` | status (`completed`, `failed`, `failed_oom`, `interrupted`), Git commit and dirty flag, config hash, data hash, task mix and counts per task, requested and loaded foundation revision, library/CUDA/GPU versions, seed, schedule, per-step loss (mean over the step's micro-batches) / lr / grad norm / tokens per second / peak VRAM, validation losses, elapsed time. Written in every outcome. |
| `checkpoints/step-NNNNNN/` | LoRA weights, optimizer, scheduler and RNG states; written atomically, last 3 kept |
| `adapter/` | final adapter only (`adapter_model.safetensors`, `adapter_config.json` with base revision, `training_run.json` with data hash, tasks and prompt versions); the trainer refuses to save any tensor that is not a LoRA tensor |

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

`--adapter` refuses an adapter trained for another base model or revision before anything is downloaded, and an adapter trained with another review prompt than the run uses; the run artifact records the adapter's weight and config hashes. The bake-off config uses `quickeval-review-v2`, so the baseline and every adapter share one prompt. It also includes `coding-v1`, whose generated code runs only in the container sandbox (docs/CODING_SANDBOX.md); without a sandbox the run stops before the model is loaded, and `--skip-coding-regression` gives a security-only run (both runs must then skip it, or `compare` refuses them). Order of checks before any model is loaded: adapter base model, revision and prompt, then the sandbox preflight. The Garden holdout reservation (`holdout_reserved.eval_cases.jsonl`) can be evaluated with `--model ... --cases ...` for a temporal/generalization check.

## Tests

- `tests/test_train_contracts.py` (runs in CI, no torch): config binding to the foundation, LoRA targets against the recorded module layout, config validation, every governance refusal, prompt masking, drop-not-truncate, deterministic batching, adapter identity checks.
- `tests/test_train_integrity.py` (CI): target-integrity attacks on the exports.
- `tests/test_train_smoke.py` (needs `torch`, `transformers`, `peft`; skipped in CI): trains a tiny randomly initialised `Qwen3_5ForCausalLM` with Gated DeltaNet and full-attention layers on CPU through the real loop, and checks checkpoints, resume, adapter-only output, foundation-revision refusal, target-count refusal, OOM recording, tampered checkpoints and planted pickles, the logged step loss and the patch task.
- `tests/test_qwen_compat.py` (needs `transformers` and `SENTINEL_QWEN_DIR` with the pinned revision's tokenizer, chat template, config and weight index, about 23 MB; no weights): with the real tokenizer the training prompt is token-for-token the evaluation prompt, prompt and reply do not merge across the boundary, only the reply plus `<|im_end|>` is supervised; every text weight of `Qwen3_5ForCausalLM` is present in the checkpoint index after transformers' `model.language_model.` → `model.` prefix mapping (vision and MTP weights are ignored).

## Reliability review (November 2026)

Checked without weights or GPU, against the real Qwen3.5-9B tokenizer and weight index at the pinned revision:

| Area | Result |
| --- | --- |
| Tokenization and masking | Correct: prompt identical to evaluation, no boundary merge, loss on reply + `<|im_end|>` only (the tokenizer's EOS); padding is `<|endoftext|>` on the right and masked |
| Weight loading | Correct: all 427 text tensors map onto `Qwen3_5ForCausalLM`; a missing tensor already stops the run |
| LoRA targets | Correct: 200 modules from the real layout, count enforced at runtime |
| Gradient accumulation | Correct since run version 1 (partial windows averaged over their own size); the logged step loss was the last micro-batch times the window size and is now the step mean |
| Checkpoint/resume | Fixed: unverified `torch.load(weights_only=False)` replaced by checksums + `weights_only=True` |
| Evaluation compatibility | Fixed: the adapter records its prompt version and the evaluation refuses a different prompt |
| Known limits | Loss is averaged per micro-batch, not per token, across a step (standard; long replies weigh like short ones). Right padding is safe for the causal DeltaNet layers; `micro_batch_size: 1` avoids padding entirely. |

## Before the first real run

1. Training-approved records exist (`python -m sentinel.train --dry-run` passes).
2. Approved GPU time; foundation baseline run completed.
3. A short calibration run (`training.max_steps: 20`) on the target GPU to confirm memory and throughput at `max_seq_len 4096`; adjust micro-batch/accumulation from `run.json`.
