# SentinelBench / QuickEval

SentinelBench is the private, versioned evaluation holdout described by the Quantum Sentinel Alpha masterplan. The public repository contains the evaluation **contract and harness**, not the private release holdout or its ground-truth artifacts.

## Case files in this directory

| File | Cases | Purpose |
| --- | --- | --- |
| `public_seed.jsonl` | 12 | Tiny synthetic harness seed for scorer/unit tests. Not model-quality evidence. |
| `quickeval_v1.jsonl` | 86 | Clean-room synthetic QuickEval used for the identical Qwen3.5-4B vs 9B bake-off. |

`quickeval_v1.jsonl` contains 37 vulnerable cases, 34 hard negatives and 15 abstention cases across Python (30), JavaScript/TypeScript (18/12) and Go (26). 30 vulnerable cases are paired via `pair_id` with a fixed or look-alike secure neighbour. `pair_accuracy` only credits a model that gets both sides right.

Each case carries:

- `case_type`: `vulnerable` → `finding`, `hard_negative` → `no_finding`, `abstention` → `abstain`
- `context`: neutral engine-style facts shown to the model. For abstention cases it states what is *not* included, for example a callee, caller or sanitizer.
- `ground_truth`: the primary CWE and line span, plus `alternate_cwes` for genuinely equivalent labels (e.g. CWE-77 for OS command injection).
- `provenance`: origin and a `content_sha256` of the code, checked by validation.

All cases were written clean-room for this project. They contain no copied upstream code. Because they are public, they are **not a holdout** and are registered as `eval_only` (`quickeval-synthetic-v1` in `data/registry.yaml`) so they can never enter training data.

## Non-synthetic cases

Cases derived from real repositories or public benchmarks (`source_kind: private_holdout` / `public_eval`) must carry `provenance.origin: upstream_derived` with an `upstream` block (source ID, repository, 40-character revision, file, license, license and redistribution status). They belong in the Git-ignored `eval/sentinelbench/private/` directory. Validation refuses non-synthetic cases in the public tree. Where redistribution rights are unclear, keep only metadata publicly and the snippet privately, or write a clean-room equivalent.

## Targets

The full private holdout should grow toward roughly 500-1,000 cases near release. Tracked metrics: precision/FPR, hard-negative accuracy, recall/F1, strict recall, pair accuracy, localization Top-1/overlap, CWE accuracy, abstention recall/precision/over-abstention, invalid-output rate, patch verification (not yet implemented, reported as null), plus runtime VRAM/latency/tok/s.
