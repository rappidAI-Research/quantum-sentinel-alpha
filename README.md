# Quantum Sentinel-Alpha

`quantum-sentinel-alpha` is the [rappidAI Research](https://www.rappidai-research.com)
project for a security-specialized code model. The model and a separate
repository-security engine are in development; no model weights or Engine CLI
are released.

The model is intended to specialize in source-code vulnerability detection, localization, CWE classification, evidence-backed explanation, remediation, minimal patch generation, secure code review, false-positive rejection, and explicit abstention. Initial language focus: Python, JavaScript/TypeScript, and Go.

The model and the Sentinel Engine are separate layers. The model is intended to be independently downloadable and locally runnable after release. The engine adds repository indexing, deterministic signals, context selection, verification, and structured JSON/SARIF output. Tool execution and sandbox enforcement remain outside the model; optional Ghost integration may be added later.

## Current phase

Phase 0 → 1, ready for the first training run: the training foundation is `ornith-ai/Ornith-1.5-9B` at revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` (a Qwen3.5-architecture model, see `docs/FOUNDATION.md`; Qwen3.5-9B was the foundation until 2026-11-24). No model has been trained and no paid AWS workload has been launched; the GPU procedure is `docs/GPU_RUNBOOK.md`. The project owner approved the three Garden data sources and 535 permissively licensed repositories for training (`docs/TRAINING_AUTHORIZATION.md`); Sentinel Garden v3 (`docs/GARDEN_V3.md`) holds 1,464 training-approved records (1,871 training examples for detection and patch generation).

See [project status](docs/STATUS.md) for the operational source of truth.

## Local setup

Use Python 3.11 or newer. The default validation runs offline and needs no GPU
or model weights. Model and training extras are documented separately.

```sh
git clone https://github.com/rappidAI-Research/quantum-sentinel-alpha.git
cd quantum-sentinel-alpha
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

The project-source license and future artifact terms are not declared. The
foundation model's Apache-2.0 license does not automatically license this
repository or a future Sentinel release.

## Foundation validation

```bash
python -m sentinel.foundation
pytest -q
```

The foundation check validates the versioned finding contract, the Sentinel Garden source registry and repository approvals, the public evaluation case files, the coding-regression reference solutions, the (optional) bake-off config, the selected training foundation and the committed Sentinel Garden manifest. It intentionally does not perform network calls or GPU work. CI (`.github/workflows/ci.yml`) runs both commands on every pull request.

## Evaluation harness

The public repository ships the SentinelBench/QuickEval contract and a small synthetic harness seed, not the private release holdout.

```bash
python -m sentinel.eval.baseline \
  --cases eval/sentinelbench/public_seed.jsonl \
  --output /tmp/qsa-baseline.jsonl

python -m sentinel.eval.quickeval \
  --cases eval/sentinelbench/public_seed.jsonl \
  --predictions /tmp/qsa-baseline.jsonl \
  --json
```

The synthetic seed exists to validate the scorer and model-adapter path. Its scores are not model-quality evidence.

## QuickEval model runs

`eval/sentinelbench/quickeval_v1.jsonl` holds 86 clean-room cases: vulnerable code, hard negatives and abstention cases. Every model goes through the same prompt, strict output parser and scorer. Each run writes one JSON artifact with security metrics, efficiency metrics and per-case outputs.

```bash
# No GPU: full path with the label-blind regex baseline (not a model result)
python -m sentinel.eval.run_model --backend baseline \
  --cases eval/sentinelbench/quickeval_v1.jsonl --output runs/baseline.json

# Real model (Linux/CUDA, pip install -e '.[model]'); --revision must be a 40-char commit SHA
python -m sentinel.eval.run_model --bakeoff configs/model/baseline_ornith15.yaml --candidate ornith15-9b \
  --output runs/eval/ornith15-9b-baseline.json

# Side-by-side comparison of two runs (no automatic winner)
python -m sentinel.eval.compare runs/eval/ornith15-9b-baseline.json runs/eval/ornith15-9b-qlora-v0.json
```

The foundation baseline is configured in `configs/model/baseline_ornith15.yaml` (Qwen3.5-9B as an optional reference). The earlier Qwen3.5 4B vs 9B bake-off is archived (`configs/model/bakeoff_qwen35.yaml`, `docs/BAKEOFF.md`).

## Sentinel Garden (training data)

Sentinel Garden discovers GitHub-reviewed advisories through OSV, resolves each one to its upstream fix commit and parent, and turns single-function fixes into vulnerable / fixed (hard negative) / abstention records with provenance, license evidence, quality checks, deduplication, contamination checks and a repository-level split. Training export is default-deny.

```bash
python -m sentinel.garden discover   # network: OSV metadata
python -m sentinel.garden run        # network: upstream git; then offline curation
python -m sentinel.garden audit      # network: OSV later-fix lookups; stratified manual sample
python -m sentinel.garden review     # network: deps.dev; rights review queue and approval templates
python -m sentinel.garden check      # offline consistency check of the committed manifest
```

Current training dataset (`sentinel-garden-v3`, `configs/garden/seed_v3.yaml`, frozen split policy `sentinel-garden-split-v1`): 4,200 records, 3,427 accepted (1,566 findings, 1,392 post-fix hard negatives, 469 abstentions; Go, Python, JavaScript, TypeScript; 1,363 verified pairs also serve as patch ground truth), **1,464 training-approved** (`docs/TRAINING_AUTHORIZATION.md`). Every exported training example is bound to the committed manifest by hash (`docs/TRAINING.md`). Garden v1 and v2 stay immutable references (`docs/GARDEN_V2.md`, `docs/GARDEN_V3.md`). Quality audit: `data/garden/seed_v3/AUDIT_REPORT.md`; independent human review packages prepared, not yet reviewed. Rights review: `data/garden/seed_v3/REVIEW_QUEUE.md` and `data/garden/seed_v3/cohort/COHORT_REVIEW.md`, workflow in `docs/RIGHTS_REVIEW.md`. Detection confidence is not supervised (`docs/CALIBRATION.md`); patch generation is a separate task (`docs/PATCH_SUPERVISION.md`).

Upstream code stays in the Git-ignored `artifacts/garden/`; `data/garden/seed_v*/` hold only metadata, statistics and review evidence. See `docs/SENTINEL_GARDEN.md` and `docs/DATA_GOVERNANCE.md`.

## QLoRA training (implemented, not run)

`python -m sentinel.train` fine-tunes the pinned `ornith-ai/Ornith-1.5-9B` with 4-bit QLoRA on governance-approved Sentinel Garden records only, in the official non-thinking chat format (no reasoning text is trained). Its dry run passes on the approved Garden v3 data (1,871 train / 187 validation examples); it refuses (exit code 2) whenever approvals or data hashes do not match. Adapters are evaluated against the unchanged foundation with `run_model --adapter` and `sentinel.eval.compare`. See `docs/TRAINING.md`.

```bash
python -m sentinel.train --dry-run --check-tokens   # governance + data checks + token budget with the pinned tokenizer; no model, no GPU
```
