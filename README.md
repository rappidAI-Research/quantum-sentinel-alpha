# Quantum Sentinel Alpha

`quantum-sentinel-alpha` is a rappidAI Research project for a small, open-weight cybersecurity model and a verified repository-security engine.

The model is intended to specialize in source-code vulnerability detection, localization, CWE classification, evidence-backed explanation, remediation, minimal patch generation, secure code review, false-positive rejection, and explicit abstention. Initial language focus: Python, JavaScript/TypeScript, and Go.

The model and the Sentinel Engine are separate layers. The model remains independently downloadable and locally runnable. The engine adds repository indexing, deterministic signals, context selection, verification, and structured JSON/SARIF output. Tool execution and sandbox enforcement remain outside the model; optional Ghost integration may be added later.

## Current phase

Phase 0: repository and contract foundation. No model has been frozen or trained, no dataset source is approved for training, and no paid AWS workload has been launched.

See `docs/STATUS.md` for the operational source of truth.

## Foundation validation

```bash
python -m sentinel.foundation
pytest -q
```

The foundation check validates the versioned finding contract, the Sentinel Garden source registry, the public evaluation case files, the coding-regression reference solutions and the bake-off config. It intentionally does not perform network calls or GPU work. CI (`.github/workflows/ci.yml`) runs both commands on every pull request.

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

The identical 4B vs 9B bake-off is configured in `configs/model/bakeoff_qwen35.yaml`. See `docs/BAKEOFF.md`.
