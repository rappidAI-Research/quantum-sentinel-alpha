# SentinelBench / QuickEval

SentinelBench is the private, versioned evaluation holdout described by the Quantum Sentinel Alpha masterplan. The public repository contains the evaluation **contract and harness**, not the private release holdout or its ground-truth artifacts.

`public_seed.jsonl` is a small synthetic engineering seed used only to prove the harness, metrics and model-adapter path. It is **not** SentinelBench-1, is not a release benchmark, and must never be reported as model quality evidence.

Before the 4B vs 9B foundation bake-off, QuickEval should be expanded to roughly 50-100 representative private cases spanning vulnerable examples, hard negatives and explicit insufficient-evidence/abstention cases. The full private holdout should grow toward roughly 500-1,000 cases near release.

Tracked metrics follow the masterplan: precision/FPR, recall/F1, localization Top-1/overlap, CWE accuracy, patch application and functional/security/regression verification, explicit abstention quality, plus run-time VRAM/latency/tok/s captured by model runners.

Private case material belongs outside the public repository. `eval/sentinelbench/private/` is ignored by Git.
