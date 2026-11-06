# Quantum Sentinel Alpha

`quantum-sentinel-alpha` is a rappidAI Research project for a small, open-weight cybersecurity model and a verified repository-security engine.

The model is intended to specialize in source-code vulnerability detection, localization, CWE classification, evidence-backed explanation, remediation, minimal patch generation, secure code review, false-positive rejection, and explicit abstention. Initial language focus: Python, JavaScript/TypeScript, and Go.

The model and the Sentinel Engine are separate layers. The model remains independently downloadable and locally runnable. The engine adds repository indexing, deterministic signals, context selection, verification, and structured JSON/SARIF output. Tool execution and sandbox enforcement remain outside the model; optional Ghost integration may be added later.

## Current phase

Phase 0 → 1: the training foundation is selected (`Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, see `docs/FOUNDATION.md`) and Sentinel Garden produces a first real dataset seed. No model has been trained, no dataset source or repository is approved for training, and no paid AWS workload has been launched.

See `docs/STATUS.md` for the operational source of truth.

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
python -m sentinel.eval.run_model --model Qwen/Qwen3.5-4B --revision <sha> \
  --cases eval/sentinelbench/quickeval_v1.jsonl --output runs/qwen35-4b.json

# Side-by-side comparison of two runs (no automatic winner)
python -m sentinel.eval.compare runs/qwen35-4b.json runs/qwen35-9b.json
```

The optional 4B vs 9B bake-off is configured in `configs/model/bakeoff_qwen35.yaml` (`docs/BAKEOFF.md`); its `qwen35-9b` candidate is also the foundation baseline run.

## Sentinel Garden (training data)

Sentinel Garden discovers GitHub-reviewed advisories through OSV, resolves each one to its upstream fix commit and parent, and turns single-function fixes into vulnerable / fixed (hard negative) / abstention records with provenance, license evidence, quality checks, deduplication, contamination checks and a repository-level split. Training export is default-deny.

```bash
python -m sentinel.garden discover   # network: OSV metadata
python -m sentinel.garden run        # network: upstream git; then offline curation
python -m sentinel.garden audit      # network: OSV later-fix lookups; stratified manual sample
python -m sentinel.garden review     # network: deps.dev; rights review queue and approval templates
python -m sentinel.garden check      # offline consistency check of the committed manifest
```

Current seed (`sentinel-garden-seed-v1`, frozen split policy `sentinel-garden-split-v1`): 2,773 records, 2,309 accepted (1,047 findings, 954 post-fix hard negatives, 308 abstentions; Go, Python, JavaScript, TypeScript; 941 verified pairs also serve as patch ground truth), **0 training-approved**. Every exported training example is bound to the committed manifest by hash (`docs/TRAINING.md`). Quality audit: `data/garden/seed_v1/AUDIT_REPORT.md`; independent human review packages prepared, not yet reviewed. Rights review queue (761 repositories, nothing approved): `data/garden/seed_v1/REVIEW_QUEUE.md`, workflow in `docs/RIGHTS_REVIEW.md`. Detection confidence is not supervised (`docs/CALIBRATION.md`); patch generation is a separate task (`docs/PATCH_SUPERVISION.md`).

Upstream code stays in the Git-ignored `artifacts/garden/`; `data/garden/seed_v1/` holds only metadata, statistics and the review queue. See `docs/SENTINEL_GARDEN.md` and `docs/DATA_GOVERNANCE.md`.

## QLoRA training (implemented, not run)

`python -m sentinel.train` fine-tunes the pinned `Qwen/Qwen3.5-9B` with 4-bit QLoRA on governance-approved Sentinel Garden records only. It refuses to start while no record is training-approved (exit code 2). Adapters are evaluated against the unchanged foundation with `run_model --adapter` and `sentinel.eval.compare`. See `docs/TRAINING.md`.

```bash
python -m sentinel.train --dry-run   # governance + data checks, no model, no GPU
```
