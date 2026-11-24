# Training Foundation: Ornith-1.5-9B

Status: **selected** (2026-11-24). Source of truth: `configs/model/foundation.yaml`, validated by `python -m sentinel.foundation`.

| Field | Value |
| --- | --- |
| Model | `ornith-ai/Ornith-1.5-9B` |
| Revision (immutable HF commit) | `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` (= `main` on 2026-11-24; last modified 2026-08-23, repository created 2026-08-18) |
| Architecture | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`); trained and evaluated text-only as `Qwen3_5ForCausalLM` |
| Text model | 32 layers (24 Gated DeltaNet, 8 full attention), hidden 4096, MLP 12288, vocabulary 248,320; 8,953,803,264 text parameters incl. `lm_head` |
| Checkpoint | 4 safetensors shards, 19,306,208,736 bytes (≈ 18.0 GiB), bf16; also holds a vision tower and an MTP head that are not loaded |
| License | MIT according to the model-card metadata. The card's `LICENSE` link returns 404 at this revision and on `main`. The card says Ornith builds on Qwen3.5 (Apache-2.0). |
| Transformers | card requires ≥ 5.8.1, the checkpoint was saved with 5.12.1; this repository requires `>=5.12,<6` (offline checks used 5.24.0) |

Everything above was read from the Hugging Face API and the small files at the pinned revision on 2026-11-24 (config, tokenizer files, chat template, weight index and the safetensors headers via HTTP range requests). **No weights were downloaded.** SHA-256 values of the files the pipeline depends on are pinned in `configs/model/foundation.yaml` (`files_sha256`).

## Decision

The project owner replaced Qwen3.5-9B with Ornith-1.5-9B as the training foundation on 2026-11-24. **Basis: expected capability, not a measured security advantage.**

- Ornith-1.5-9B is a further-trained Qwen3.5-9B-architecture model. Its config, all 775 tensor names and the text-model tensor shapes match Qwen3.5-9B exactly. `tokenizer.json`, `vocab.json` and `merges.txt` are byte-identical. The QLoRA configuration, its 200 LoRA targets and the Garden v3 token budget therefore carry over unchanged (verified below, not assumed).
- Its authors trained it further for coding and agentic tasks. The model card reports higher coding scores than Qwen3.5-9B; these numbers are vendor-reported and not reproduced here.
- The official chat template has a non-thinking generation prompt (`enable_thinking=false`), which the strict JSON output contract needs.
- Fits unquantized (bf16) on one 48 GB L40S for evaluation; QLoRA with a 4-bit base fits the same GPU.

What this decision does **not** claim:

- No QuickEval or coding-regression result exists for Ornith-1.5-9B.
- The vendor benchmarks have not been reproduced and do not measure vulnerability detection.
- Ornith has not been shown to outperform Qwen3.5-9B on security tasks.

## Verified compatibility (offline, real files, no weights)

| Check | Result | How |
| --- | --- | --- |
| Loaded class and weights | `AutoModelForCausalLM` builds `Qwen3_5ForCausalLM`; every text weight exists in the checkpoint index after the `model.language_model.` → `model.` mapping; only vision and MTP tensors are left out | `tests/test_foundation_compat.py` |
| LoRA targets | 249 linear modules; the unchanged regex matches 200 (`q/k/v/o_proj` ×8, `in_proj_qkv/in_proj_z/out_proj` ×24, `gate/up/down_proj` ×32); 40,108,032 parameters at rank 16; every shape equals the safetensors header | `scripts/inspect_foundation.py --check` |
| Tokenizer | identical to Qwen3.5-9B; `<|im_end|>` (248046) is the EOS and end of turn, `<|endoftext|>` (248044) the pad; `<think>` (248068) and `</think>` (248069) are ordinary, non-special tokens | `tests/test_foundation_compat.py` |
| Chat template | differs from Qwen3.5-9B only in how *earlier* assistant turns are rendered (Ornith always writes the `<think>…</think>` block). The generation prompt is identical: with `enable_thinking=False` it ends in `<|im_start|>assistant\n<think>\n\n</think>\n\n` | file diff, compat test |
| Training rendering | training prompt is token-for-token the evaluation prompt; only the JSON reply plus `<|im_end|>` is supervised; no reasoning token is ever a training target; a finished example equals the template's own rendering of the conversation | compat test |
| Token budget | Garden v3 encoded with the Ornith tokenizer: train 1,871 kept / 0 dropped (median 764, max 2,918, 1,732,959 tokens per epoch), validation 187 / 0 (median 793, max 2,481) | `python -m sentinel.train --dry-run --check-tokens` |
| Generation stop | **fixed:** the checkpoint has no `generation_config.json`, so the config-derived EOS is `<|endoftext|>` and generation would run past the answer into a new turn. The evaluation backend now stops at `<|im_end|>` and decodes nothing after the first end of turn (the same was latent for Qwen3.5-9B) | `tests/test_generation_stop.py`, compat test |

**Not yet verified (needs a GPU):** how reliably Ornith follows the JSON contract in non-thinking mode before training, NF4 loading and peak memory, and throughput with or without the optional Gated DeltaNet kernels. The parser keeps rejecting malformed output as `invalid` and strips any `<think>` block the model might still emit.

## Consequences

- Training targets `ornith-ai/Ornith-1.5-9B@489cb97…` (`configs/train/qlora_ornith15_9b.yaml`). The training config cannot name another model.
- The baseline is the un-adapted Ornith on QuickEval v1 (and `coding-v1` when the sandbox is available), run through `configs/model/baseline_ornith15.yaml`. `sentinel.foundation` checks that its `ornith15-9b` candidate carries exactly the pinned model ID and revision. The required comparison is Ornith before vs after training.
- Qwen3.5-9B stays in the same config as an optional reference. The Qwen3.5-4B vs 9B bake-off (`configs/model/bakeoff_qwen35.yaml`, `docs/BAKEOFF.md`) is archived.
- Changing the model or revision is a deliberate change: update `configs/model/foundation.yaml`, the baseline candidate pin, `configs/train/ornith15_9b_linear_modules.txt` (`scripts/inspect_foundation.py --write`), this record and `docs/STATUS.md` together.
- Temporal quarantine: the frozen cutoff (2026-07-01) was chosen to lie after the Qwen3.5-9B checkpoint (2026-02-27). Ornith-1.5-9B was published on 2026-08-18, so its own training may include fixes from the quarantine window. The split policy stays frozen and training data is unaffected; in evaluations, report the 183 Garden v3 quarantine records fixed before 2026-08-18 separately from the 172 fixed afterwards. QuickEval v1 (clean-room synthetic cases) is unaffected.
- Before any weight release: clarify the missing `LICENSE` file and the obligations from the Qwen3.5 (Apache-2.0) base, together with the open questions in `docs/TRAINING_AUTHORIZATION.md`.

## History

| Period | Foundation | Basis |
| --- | --- | --- |
| 2026-11-04 → 2026-11-24 | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache-2.0) | expected capability (Masterplan v1.2); replaced the planned 4B vs 9B bake-off as a selection gate. No run was made with it. |
| since 2026-11-24 | `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` (MIT) | expected capability (owner instruction) |

## Open evidence

1. Foundation baseline run of Ornith-1.5-9B (QuickEval v1, and `coding-v1` if the sandbox is available) before any adapter.
2. The first QLoRA run: memory, throughput and the before/after comparison on the same evaluation contract (`docs/GPU_RUNBOOK.md`).
