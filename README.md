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

The foundation check validates the versioned finding contract and the Sentinel Garden source registry. It intentionally does not perform network calls or GPU work.
