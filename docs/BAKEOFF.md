# Qwen3.5-4B vs Qwen3.5-9B QuickEval Bake-off

Status: **prepared, not run.** Running it needs a CUDA GPU and explicit approval for any paid instance.

## What is held identical

`configs/model/bakeoff_qwen35.yaml` fixes everything except model identity:

- cases: `eval/sentinelbench/quickeval_v1.jsonl` (86 clean-room cases, hashed into every artifact)
- prompt: `quickeval-review-v1` (`sentinel/eval/prompt.py`, hashed into every artifact)
- output contract, strict parser and scorer (versioned in every artifact)
- greedy decoding, `max_new_tokens: 384`, `enable_thinking: false`, seed 0
- runtime: CUDA, bfloat16, no quantization. Both models fit unquantized on one 48 GB L40S.

A candidate may override `device`/`dtype`/`quantization` only with a written `override_reason`.
`compare` refuses (exit code 2) to compare runs whose cases, prompt, parser, scorer or generation settings differ, and lists runtime and kernel differences.

## Runbook (any Linux/CUDA host; not AWS-specific)

1. Pin both models. Set `revision` for each candidate to the full 40-character commit SHA of the Hugging Face repository and commit the config. Unpinned runs are refused unless you pass `--allow-unpinned`, and are then marked non-reportable.
2. Install: `pip install -e '.[model]'`. Use identical environments for both runs. The optional `flash-linear-attention`/`causal-conv1d` kernels change throughput, so install them for both runs or for neither. The artifact records which are present.
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

QuickEval v1 is public, small (37 vulnerable / 34 hard-negative / 15 abstention cases) and authored by the project. Treat differences of one or two cases as noise. Use it to find large gaps, not for release claims. Coding regression has an interface (`sentinel/eval/coding_regression.py`) but no suite yet, so artifacts report it as `not_run`.

## Cost note

Two models × 86 cases with ≤384 new tokens each should take tens of minutes of single-GPU time, plus model download. This is an estimate, not a measurement: without the optional kernels, Qwen3.5's linear-attention layers fall back to slower PyTorch code. Do not launch a paid instance without explicit approval.
