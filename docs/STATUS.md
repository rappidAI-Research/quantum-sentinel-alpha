# Quantum Sentinel Alpha Status

Last updated: 2026-10-07

## Current phase

Phase 0 - Foundation. Milestone completed: **QuickEval + real model runner + Qwen3.5 4B/9B bake-off readiness**. The bake-off itself has **not** been run.

## Completed

- Foundation: versioned finding schema, Sentinel Garden source registry (default-deny), threat model, data-governance rules, QuickEval case/prediction contracts, deterministic engineering baseline and scorer.
- QuickEval v1 (`eval/sentinelbench/quickeval_v1.jsonl`): 86 clean-room synthetic cases with hashed provenance, validated line spans, accepted alternate CWEs and 30 vulnerable/secure neighbour pairs.
- Shared evaluation path (`sentinel.eval.run_model`): one prompt (`quickeval-review-v1`), one strict output parser and one scorer for every model. Malformed output is scored as an `invalid` prediction and never repaired. Reasoning blocks are stripped and never stored.
- Hugging Face backend for `Qwen/Qwen3.5-4B` / `Qwen/Qwen3.5-9B` through `AutoModelForCausalLM`, with configurable device, dtype, bitsandbytes 4/8-bit quantization and generation limits. The backend fails on missing checkpoint weights.
- Reportability gate: a run is reportable only with an immutable 40-character model revision that matches the loaded revision, a real backend, the full case file and a clean, known Git commit. Unpinned real runs are refused unless `--allow-unpinned` is passed, and are then marked non-reportable.
- Run artifact (JSON, schema `data/schemas/eval_run.schema.json`): security metrics, efficiency metrics (load time, eval time, mean latency, generated tokens, tok/s, peak VRAM), runtime/library versions, case-set hash, prompt hash, parser/scorer versions, Git SHA and per-case outputs.
- Extended metrics: hard-negative accuracy, strict recall (decision + location + CWE), vuln/fix pair accuracy, abstention precision, over-abstention, findings on abstention cases, invalid-output rate, per-language decision accuracy. Patch metrics stay `null` because patch evaluation is not implemented.
- Bake-off config `configs/model/bakeoff_qwen35.yaml` with shared cases, prompt and generation settings. Per-candidate runtime overrides require a written reason. Runbook: `docs/BAKEOFF.md`.
- `sentinel.eval.compare`: side-by-side security/abstention/efficiency table, runtime/kernel differences and per-case disagreements. It computes no winner and refuses to compare runs that used different evaluation contracts.
- Coding-regression extension point (`sentinel/eval/coding_regression.py`). No suite is registered yet; artifacts report `not_run`.
- Heavy dependencies are optional extras: `.[model]` (torch, transformers ≥5.2,<6, accelerate) and `.[quant]` (bitsandbytes).

## QuickEval state

| Group | Python | JS/TS | Go | Total |
| --- | --- | --- | --- | --- |
| Vulnerable (`finding`) | 13 | 13 | 11 | 37 |
| Hard negative (`no_finding`) | 12 | 12 | 10 | 34 |
| Abstention (`abstain`) | 5 | 5 | 5 | 15 |
| **Total** | 30 | 30 (18 JS / 12 TS) | 26 | **86** |

CWEs covered: 89, 943, 78, 22, 79, 1336, 918, 502, 1321, 95, 639, 328, 338, 347, 295, 601.

- Public / synthetic: QuickEval v1 (86) and the 12-case harness seed. Both are clean-room, public and registered `eval_only` (`quickeval-synthetic-v1`), so they are **not** a holdout.
- Private holdout: not started. No private cases or answers are in the repository.
- Public benchmarks (VulnBench, SeCodeBench, SeCodePLT, PrimeVul): registered evaluation-only. No cases imported.
- Contamination note: QuickEval v1 was written after the Qwen3.5 checkpoints were released, so it cannot be in their training data. Future models may see it because it is public.

## Benchmark state

No model has been evaluated. The only QuickEval v1 results are from the regex engineering baseline (backend `baseline`). They exist to test the harness and **are not model performance**.

## Model state

Foundation is **not frozen**. Qwen3.5-4B and Qwen3.5-9B are bake-off candidates. Neither has been downloaded, run or modified. No adapter or training run exists. Exact revisions are **not pinned yet**: `revision: null` in the bake-off config. Hugging Face was not reachable from the development environment, so revisions must be pinned before the first reportable run.

## Dataset state

Source registry: 7 sources, all `production_approved: false`. Training records: 0. Training-approved sources: 0.

## AWS state

No AWS resource was started for this milestone. Planning assumptions are unchanged: `us-east-1`, G/VT Spot quota of 4 vCPUs, target `g6e.xlarge` (1× L40S 48 GB), and roughly USD 1,160 of promotional credits. Quota, credits, Spot capacity and price must be verified right before any paid run. A paid run requires explicit approval.

## Open decisions

- Approve the paid GPU time for the bake-off (`g6e.xlarge` or any other CUDA host).
- Freeze 4B vs 9B after the identical bake-off and a coding-regression check.
- Choose a coding-regression suite (small, license-clean, executed in an isolated sandbox).
- Exact training-source approvals after rights/terms review.
- Final LoRA rank/LR/sequence length after one short L40S calibration.
- Release thresholds after the foundation baseline.
- Repository license (affects redistribution status of the project-authored eval cases).

## Next action

1. Pin the Hugging Face commit SHAs for both candidates in `configs/model/bakeoff_qwen35.yaml`.
2. After approval, run the `--limit 5` smoke test and then both full runs on one CUDA host, as described in `docs/BAKEOFF.md`.
3. Compare the runs with `sentinel.eval.compare` and make the foundation decision by human judgement.
4. In parallel, start the private SentinelBench holdout from rights-reviewed upstream fixes.
