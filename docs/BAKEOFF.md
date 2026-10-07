# Qwen3.5-4B vs Qwen3.5-9B QuickEval Bake-off

Status: **prepared, not run.** Running it needs a CUDA GPU and explicit approval for any paid instance.

## What is held identical

`configs/model/bakeoff_qwen35.yaml` fixes everything except model identity:

- cases: `eval/sentinelbench/quickeval_v1.jsonl` (86 clean-room cases, hashed into every artifact)
- prompt: `quickeval-review-v1` (`sentinel/eval/prompt.py`, hashed into every artifact)
- output contract, strict parser and scorer (versioned in every artifact)
- greedy decoding, `max_new_tokens: 384`, `enable_thinking: false`, seed 0
- runtime: CUDA, bfloat16, no quantization. Both models fit unquantized on one 48 GB L40S.
- coding regression: `coding-v1` (16 clean-room Python tasks, `eval/coding/coding_v1.jsonl`, hashed into every artifact), generated with the same settings

A candidate may override `device`/`dtype`/`quantization` only with a written `override_reason`.
`compare` refuses (exit code 2) to compare runs whose cases, prompt, parser, scorer, generation settings or coding suite differ, and lists runtime and kernel differences.

A run is marked non-reportable if the loaded model revision or the loaded parameter dtype differs from the request (both are recorded in the artifact), if `transformers` is older than 5.12, or if it used a subset of cases or uncommitted code.

## Runbook (any Linux/CUDA host; not AWS-specific)

1. Revisions are pinned in the config (4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, fetched 2026-10-07). Re-pinning is a deliberate change: set `revision: null`, run `python scripts/pin_hf_revisions.py`, review and commit. Unpinned runs are refused unless you pass `--allow-unpinned`, and are then marked non-reportable.
2. Install: `pip install -e '.[model]'` (transformers ≥5.12,<6). Use identical environments for both runs. The optional `flash-linear-attention`/`causal-conv1d` kernels change throughput, so install them for both runs or for neither. The artifact records which are present.
3. Smoke test (cheap, never reportable):
   `python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-4b --limit 5 --output runs/smoke-4b.json`
4. Full runs from a clean checkout. A dirty working tree makes the run non-reportable.

   ```bash
   python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-4b --output runs/qwen35-4b.json
   python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b --output runs/qwen35-9b.json
   python -m sentinel.eval.compare runs/qwen35-4b.json runs/qwen35-9b.json
   ```

5. Keep both artifacts. They hold every raw (reasoning-stripped) output, parsed prediction and per-case latency, so results can be audited or re-scored later.

## Reading the result

No combined score or winner is computed. The decision weighs precision/false-positive rate, recall, localization, CWE accuracy, abstention behaviour, invalid-output rate, coding regression, VRAM, latency and throughput.

QuickEval v1 is public, small (37 vulnerable / 34 hard-negative / 15 abstention cases) and authored by the project. Treat differences of one or two cases as noise. Use it to find large gaps, not for release claims. `coding-v1` is equally small (16 tasks): it detects a clear coding regression, not fine-grained skill.

`coding-v1` executes model-generated Python in a child process with CPU/memory/file-size/time limits, an empty environment and a temporary directory. That is containment, not a sandbox: run the bake-off on a disposable evaluation host without credentials you would mind losing. Its answers share `max_new_tokens: 384`; an implementation cut off by that limit fails, identically for both models.

## Cost note

Two models × 86 cases with ≤384 new tokens each should take tens of minutes of single-GPU time, plus model download. This is an estimate, not a measurement: without the optional kernels, Qwen3.5's linear-attention layers fall back to slower PyTorch code. Do not launch a paid instance without explicit approval.
