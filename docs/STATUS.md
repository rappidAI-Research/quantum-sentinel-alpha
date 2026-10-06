# Quantum Sentinel Alpha Status

Last updated: 2026-10-06

## Current phase

Phase 0 - Foundation.

## Completed

- Masterplan v1.1 Lean reviewed end-to-end and adopted as the architectural handoff, with the explicit naming override below.
- Current `rappidAI-Research` GitHub organization inspected; `rappidAI-Research/quantum-sentinel-alpha` exists as the public project repository.
- Current Ghost remote architecture checked; Ghost remains a separate deterministic runtime and is not a blocker for Sentinel training.
- Foundation and evaluation bootstrap implemented on `foundation/project-bootstrap`; GitHub App access is active and remote publication is in progress.
- Versioned finding schema implemented.
- Sentinel Garden source registry and validation policy implemented with default-deny training approval.
- Initial threat model and data-governance rules implemented.
- QuickEval/SentinelBench case and prediction contracts implemented.
- Synthetic 12-case public harness seed added across Python, JavaScript and Go; it is explicitly not the private SentinelBench holdout.
- Deterministic engineering baseline and scorer implemented; end-to-end harness verified.

## Current decisions

- Permanent public/model name: **Quantum Sentinel Alpha**.
- Technical/repository name: `quantum-sentinel-alpha`.
- Hugging Face target: `rappidAI/quantum-sentinel-alpha`.
- No numbered Quantum Sentinel model line is active; older `Quantum Sentinel 1` / `quantum-sentinel-1` naming in the masterplan is superseded.
- Foundation model is **not frozen**. Qwen3.5-9B and Qwen3.5-4B remain bake-off candidates.
- Public security benchmarks are evaluation-only unless explicitly reclassified after contamination/rights review.
- No dataset source is approved for training yet.
- No paid AWS workload has been launched.

## Benchmark state

No reportable SentinelBench model score exists yet. The evaluation harness is working on a 12-case synthetic public contract seed. Before the 4B vs 9B bake-off, build the real private QuickEval toward 50-100 representative cases; public benchmark cases must not be copied into training artifacts.

## Dataset state

Source registry exists. Training records: 0. Training-approved sources: 0. Real advisory-to-commit collection has not started.

## Model state

No weights downloaded or modified. No adapter/training run exists. Exact upstream model revisions remain unpinned until the bake-off is prepared.

## AWS state

Project handoff states `us-east-1`, `All G and VT Spot Instance Requests = 4 vCPUs`, target `g6e.xlarge`, and approximately USD 1,160 promotional-credit planning envelope. Quota, remaining credits, service eligibility, Spot capacity and current price must be verified in AWS immediately before the first paid GPU run.

## Open decisions

- 4B vs 9B foundation freeze after identical security bake-off.
- Exact training-source approvals after rights/terms review.
- Final LoRA rank/LR/sequence length after one short L40S calibration.
- Release thresholds after foundation baseline.

## Next action

Merge the foundation + evaluation bootstrap, then build the private 50-100 case QuickEval and model-runner adapter for the identical 4B vs 9B bake-off.
