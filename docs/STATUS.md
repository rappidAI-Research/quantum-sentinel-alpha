# Quantum Sentinel Alpha Status

Last updated: 2026-11-04

## Current phase

Phase 0 → 1. Completed milestones: QuickEval + real model runner + bake-off readiness (PR #2), pre-bake-off hardening, and now **foundation finalization (Qwen3.5-9B)**, **Sentinel Garden** (data pipeline) and the **first Sentinel Garden dataset seed** (staging only, not training-approved). No model has been run or trained.

## Completed

### Foundation and evaluation (earlier milestones)

- Versioned finding schema, Sentinel Garden source registry (default-deny), threat model, data-governance rules, QuickEval case/prediction contracts, deterministic engineering baseline and scorer.
- QuickEval v1 (`eval/sentinelbench/quickeval_v1.jsonl`): 86 clean-room synthetic cases with hashed provenance, validated line spans, accepted alternate CWEs and 30 vulnerable/secure neighbour pairs.
- Shared evaluation path (`sentinel.eval.run_model`): one prompt (`quickeval-review-v1`), one strict output parser and one scorer for every model; malformed output is scored `invalid`, never repaired; reasoning is stripped and never stored.
- Hugging Face backend (`AutoModelForCausalLM`, device/dtype/bitsandbytes/generation settings), reportability gate (immutable revision, loaded revision and dtype match, full case file, clean Git commit), JSON run artifacts with security and efficiency metrics, `sentinel.eval.compare` (no automatic winner).
- Coding-regression suite `coding-v1` (16 clean-room Python tasks, contained child-process execution), `scripts/pin_hf_revisions.py`, GitHub Actions CI (`pytest` + `python -m sentinel.foundation`).

### Milestone 1 - Foundation finalization

- **Selected training foundation: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`** (Apache-2.0, `Qwen3_5ForConditionalGeneration`, 19.3 GB bf16; checkpoint published 2026-02-27). Source of truth `configs/model/foundation.yaml`, decision record `docs/FOUNDATION.md`.
- The decision basis is recorded as `expected_capability`: 9B was chosen for its expected capabilities, **not** for an experimentally demonstrated security advantage. No QuickEval or coding-regression result exists for either Qwen3.5 checkpoint.
- `sentinel.foundation_model` validates the config and checks that the bake-off candidate `qwen35-9b` (the foundation baseline path) carries exactly the selected model ID and revision. `python -m sentinel.foundation` reports it.
- The 4B vs 9B bake-off is kept as an **optional** efficiency comparison (`docs/BAKEOFF.md`), not a prerequisite. `configs/model/foundation_candidates.yaml` was replaced by `configs/model/foundation.yaml`.
- No weights were downloaded; no training was started.

### Milestone 2 - Sentinel Garden

Working end-to-end pipeline in `sentinel/garden/` (docs: `docs/SENTINEL_GARDEN.md`), standard library + existing dependencies only:

1. **Discovery** (`osv.py`): GitHub-reviewed GHSA advisories from the public OSV export (PyPI, npm, Go) → candidates with 1-2 localizable CWEs and exactly one full-SHA upstream fix commit; deterministic salted-hash sampling; per-repository/ecosystem caps; frozen candidate list.
2. **Revision resolution** (`gitsource.py`): partial, shallow fetch of fix commit + parent by full SHA; vulnerable revision = first parent.
3. **Diff extraction and context** (`build.py`, `code.py`, `diff.py`): function-level vulnerable/fixed excerpts for fixes whose logic change is confined to one function (Python `ast`, Go gofmt layout, JS/TS masked brace matching); comment/import/docstring-only hunks ignored; helper code added by the fix included in the fixed excerpt; file imports as context; insertion-point localization for added checks; upstream patch restricted to the function; caller-without-callee abstention records.
4. **Provenance, license and terms tracking**: advisory, OSV record hash, repository, both revisions, fix date, file/function/line ranges, content and normalized hashes, root license files with sha256 (detected license always `unreviewed`). New registry source `ghsa-osv-fix-commits` (`train_candidate`, not approved). Per-repository approvals in `data/garden/approvals.yaml` (schema-validated, bound to the reviewed license file hash, empty = default-deny).
5. **Quality checks and deduplication** (`quality.py`): 20+ record checks, strict patch application (patch must turn the vulnerable excerpt into the fixed one), sibling rejection, exact/near duplicates, label-conflict detection, evaluation-contamination checks against every QuickEval/seed/private case and `coding-v1` task (exact, Jaccard, containment, repository overlap).
6. **Normalization and export** (`split.py`, `governance.py`, `pipeline.py`, `render.py`): repository-level split + temporal cutoff, default-deny training gate, deterministic JSONL, chat rendering with the QuickEval prompt (every target round-trips through the strict eval parser), holdout export in `eval_case` format, statistics, report and review queue. `verify` rebuilds every record and compares content hashes; `check` validates the committed manifest offline and runs in `sentinel.foundation`/CI.

### Milestone 3 - First dataset seed: `sentinel-garden-seed-v1`

Produced by the pipeline from real advisories and upstream fix commits (OSV export read 2026-11-04, snapshot bound 2026-11-01). Full numbers: `data/garden/seed_v1/REPORT.md` / `stats.json`.

| Stage | Count |
| --- | ---: |
| GHSA advisories inspected (1,400 per ecosystem) | 4,200 |
| Candidates (single fix commit, localizable CWE, caps applied) | 1,413 |
| Fix commits turned into records (one-function rule) | 367 |
| Records built | 869 |
| Records accepted by quality checks | **815** |
| Records rejected (label conflicts 6, trivial span 14, diff inconsistency 5, syntax 2, sibling 27) | 54 |
| **Training-approved records** | **0** |

Accepted records by language and label:

| Language | finding | no_finding (post-fix hard negative) | abstain | Total |
| --- | ---: | ---: | ---: | ---: |
| Python | 92 | 92 | 27 | 211 |
| JavaScript | 79 | 79 | 24 | 182 |
| TypeScript | 49 | 49 | 16 | 114 |
| Go | 124 | 125 | 59 | 308 |
| **Total** | **344** | **345** | **126** | **815** |

- 342 complete vulnerable/fixed pairs from 301 repositories; 97 distinct primary CWEs. Most frequent: CWE-79 (31), CWE-400 (25), CWE-22 (22), CWE-601 (15), CWE-1321 (14), CWE-200 (13), CWE-78 (13), CWE-863 (12), CWE-20 (11). 56 findings carry a coarse CWE (flagged).
- Localization (diff-derived): 198 changed-line spans, 100 insertion points (missing checks), 46 mixed.
- Splits (repository-level, temporal cutoff 2026-07-01, provisional): train 563 records / 217 repos, validation 75 / 27, holdout_reserved 96 / 33, temporal_quarantine 81 / 28. No repository is in two splits; no train/validation record is after the cutoff.
- Evaluation contamination: 0 hits against QuickEval v1, the public seed, `coding-v1` (and `private/`, which is empty here).
- Determinism: `curate` is byte-identical across runs; `verify` rebuilt all 869 records from the frozen candidates with 0 hash mismatches.
- Manual spot check by the implementing engineer (not a review or approval): 22 sampled findings across all four languages; labels and localizations matched the upstream fixes, 2 spans were wider than necessary but correct. One systematic issue (Python docstring edits widening spans) was found and fixed before the final run.

**Approval status:** all 815 accepted records are a **non-approved staging dataset**. 638 train/validation records are `pending_review`; holdout and quarantined records are `excluded` from training. Blockers on every accepted record: `repository_not_reviewed`, `source_not_approved:ghsa-osv-fix-commits`, `source_not_approved:osv`, `source_not_approved:github-advisory-db`. `train.jsonl` and `validation.jsonl` are empty. Detected licenses (unreviewed): 658 records permissive, 89 strong copyleft (mostly AGPL-3.0), 15 weak copyleft, 53 unknown or no root license file.

Upstream code is **not committed** (redistribution unreviewed, repository license undeclared). The Git-ignored `artifacts/garden/seed_v1/` holds the full records; `data/garden/seed_v1/` holds candidates, manifest (metadata only), statistics, report and the review queue.

## QuickEval state

| Group | Python | JS/TS | Go | Total |
| --- | --- | --- | --- | --- |
| Vulnerable (`finding`) | 13 | 13 | 11 | 37 |
| Hard negative (`no_finding`) | 12 | 12 | 10 | 34 |
| Abstention (`abstain`) | 5 | 5 | 5 | 15 |
| **Total** | 30 | 30 (18 JS / 12 TS) | 26 | **86** |

- Public / synthetic: QuickEval v1 (86) and the 12-case harness seed, both registered `eval_only`, **not** a holdout.
- Private holdout: not started. Sentinel Garden's `holdout_reserved` records (96) are exported in eval-case format, but their identities are public, so they are a temporal/generalization reservation, not a private holdout.
- Public benchmarks (VulnBench, SeCodeBench, SeCodePLT, PrimeVul): registered evaluation-only. No cases imported.

## Benchmark state

No model has been evaluated. The only QuickEval v1 results are from the regex engineering baseline and are not model performance.

## Model state

Foundation **selected**: `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Not downloaded, not run, not modified. No adapter, training run or training code exists.

## AWS state

No AWS resource was started. Planning assumptions are unchanged: `us-east-1`, G/VT Spot quota of 4 vCPUs, target `g6e.xlarge` (1× L40S 48 GB), roughly USD 1,160 of promotional credits. Quota, credits, Spot capacity and price must be verified right before any paid run. A paid run requires explicit approval.

## Blockers and open decisions

1. **Training approvals (blocks any training export).** An authorized reviewer must (a) approve the registry sources `ghsa-osv-fix-commits`, `osv` and `github-advisory-db` (terms, attribution for GHSA CC-BY-4.0 metadata), and (b) review repositories in `data/garden/seed_v1/REVIEW_QUEUE.md` and record decisions in `data/garden/approvals.yaml`. Policy decision needed on copyleft (AGPL/GPL) and unlicensed repositories.
2. Approval of paid GPU time for the foundation baseline run (QuickEval v1 + `coding-v1` on the 9B checkpoint).
3. Freeze the temporal cutoff and split salt before large-scale collection.
4. QLoRA training code/config does not exist yet; final LoRA rank/LR/sequence length after one short L40S calibration.
5. Whether to replace `coding-v1` containment (child process + rlimits) with a real sandbox before running it on shared hosts.
6. Release thresholds after the foundation baseline; repository license (affects redistribution of project-authored eval cases and any future committed data).
7. Masterplan v1.2 itself is not in this repository; this milestone follows the task brief that summarizes it.

## Next action (toward the first QLoRA run)

1. Rights review: approve the three registry sources and review the permissive-licensed repositories in `REVIEW_QUEUE.md` (exact license file links and sha256 are listed). Reviewing those 196 repositories would make up to 509 train/validation records eligible. Then `python -m sentinel.garden curate` populates `train.jsonl`, `validation.jsonl` and `train.chat.jsonl` deterministically - no rebuild needed.
2. With approved GPU time, run the foundation baseline: `python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b --output runs/qwen35-9b-baseline.json`.
3. Implement the QLoRA training config/script consuming `train.chat.jsonl` (4-bit base `Qwen/Qwen3.5-9B@c202236…`, prompt `quickeval-review-v1`), run a short L40S calibration, then the first QLoRA run, evaluated on QuickEval v1, `coding-v1` and the Garden validation split against the baseline.
