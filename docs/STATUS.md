# Quantum Sentinel Alpha Status

Last updated: 2026-11-04

## Current phase

Phase 0 → 1. Completed milestones: QuickEval + real model runner + bake-off readiness (PR #2), pre-bake-off hardening, **foundation finalization (Qwen3.5-9B)** and **Sentinel Garden** (PR #4), the **QLoRA training pipeline** (PR #5, not run), and the **Sentinel Garden seed over the full advisory population** with a quality audit and a rights review queue (staging only, not training-approved). No model has been run or trained.

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

### PR #4 review hardening

Codex review findings fixed with regression tests (`tests/test_garden_regressions.py`):

- Repository keys keep dots (`example/webapp.js`); only a trailing `.git` is removed, so repository-level contamination checks match discovery keys.
- Python callers are found with the tokenizer: names inside strings, comments or docstrings are not calls. This removed one real false abstention record (aiohttp: `write_eof` was only mentioned in an error message). The quality check uses the same logic.
- Line classification is structural on the whole file instead of per line: a Go `*out = in` dereference, a `#` or `from x import y` line inside a Python string, a Go string-literal line outside the import block and lines inside multi-line template literals / raw strings are code; block-comment continuations, docstrings, import statements and bracket-only lines are not.

Adjacent defects found and fixed while doing this: TypeScript `import x = require('y')` without a semicolon, identifiers starting with `import`, and semicolon-free JS calls misread as method definitions (which created a phantom `mergeKeys` function inside an abstention excerpt). Second review round: Python normalization now keeps significant indentation (a fix that moves `return secret` into an `if is_admin:` block was rejected as cosmetic and collided in deduplication), and TypeScript return-type annotations such as `): { html: string } {` or `Promise<{...}>` are skipped before the body brace. Upstream caches now fetch with `--filter=tree:0`: the previous `blob:none` filter downloaded a full tree per commit and filled 29 GB of disk for the first 975 repositories; a monorepo fix now costs megabytes. Third review round: string, template, regex and raw-string literals are single exact tokens in the normalization (a fix that only changes whitespace inside a regex such as `^[a-z ]+$` was rejected as cosmetic), and bracket-only lines are no longer "no logic": moving a closing brace changes control flow, counts for the one-function rule and is localized at the change point. Line classes are now split into noise (blank, comment, docstring, import) and non-localizable (noise plus bracket-only).

Two operational defects were found while rebuilding: (1) the presence check `git cat-file -e <sha>` triggered Git's lazy fetch of the commit's whole closure in a partial clone, which pulled entire histories (grafana: 10 M objects, 2.5 GB) and exhausted the disk; presence is now tracked in a marker file and the whole cache for 975 repositories is 0.5 GB; (2) a fix commit that is also the parent of another fix became a shallow boundary and looked parentless on a rebuild (12 records); parents are now read from the raw commit object and such commits are deepened. Fourth review round: line breaks outside literals are kept (collapsed) in the Go/JS/TS normalization, because automatic semicolon insertion makes `return\n  !x` differ from `return !x`; and lines inside multi-line literals (Go raw strings, template literals) are code even when whitespace-only. Property tests now check that no whitespace edit inside a literal and no join of two code lines is ever treated as cosmetic. Fifth review round: between a parameter list and a body brace only a signature continuation (nothing, `=>`, a return-type annotation) is accepted, so a semicolon-free call followed by `return {` is no longer a phantom method; operator runs are single tokens (`a + ++b` differs from `a++ + b`); imports inside functions are code, and a module-level name re-bound to a different module (e.g. Go `template` from text/template to html/template) rejects the fix as `module_import_rebound` instead of attributing it to an unrelated function. Sixth round: re-binding detection and noise classification share one definition of module scope (including module-level `try`/`if`/`with` blocks, excluding function and class bodies), and a removed block comment still separates tokens. Seventh round: Go function detection runs on the masked source, so `func` text inside raw strings or comments is never a function (rebuild unchanged). Eighth round: near-duplicate and label-conflict candidates are found with exact prefix filtering instead of an index that skipped frequent shingles, so large copy families are compared too (property-tested against brute force; no record changed). Ninth round: Python is tokenized without dedenting the text first (only the outer INDENT/DEDENT of a method excerpt is dropped), so whitespace-only lines inside multi-line strings stay significant; 43 normalized hashes changed, no record content, status or split. Tenth round: a caller only counts when it really reaches the callee - a locally shadowed name (assignment, parameter, import, nested definition) does not, and a method or a function assigned to an object is only reached through its own receiver (`self.`/`this.`/the Go receiver of the same type/the same object path), not through a same-named member of another object; a regex directly after an `if (...)`/`while (...)`/`for (...)` head is a literal, and `a++ / 2` is a division. The rebuild removed 15 abstention records whose call went to another object or could not be shown to reach the callee (e.g. `a.DiscoverKeys(...)` for a plain function `DiscoverKeys`); totals below are regenerated. Eleventh round: a `self.`/`this.` call only counts when caller and callee belong to the same class (2 more abstention records removed), and destructured `require` imports are tracked per binding for the re-binding check (no record affected). Twelfth round, and the end of hand-written lexing: Go, JavaScript and TypeScript are now read from tree-sitter parse trees (`sentinel/garden/syntax.py`; grammar versions pinned in `pyproject.toml`, recorded in `stats.json` and enforced by `check`). Literals, comments, normalization tokens (operators are the grammar's own tokens, so `===!` equals `=== !`), function boundaries, calls, imports and local names come from the same tree; a nested function only has callers inside its enclosing function. A differential run over all 548 JS/TS/Go files of the seed found no Go difference and only corrections for JS/TS (named function expressions, multi-line and `require('m').x` imports, a BOM before `import`, functions the old scanner missed); the rebuild renamed 6 pairs (e.g. `async` -> anonymous, `<anonymous>` -> `cors`), accepted one previously truncated nestjs pair and added 2 abstention records. Thirteenth round: JSX text is compared as the string React renders (Babel's whitespace rule), and line breaks inside JSX markup are layout, not statement boundaries (2 normalized hashes changed, no record). The dataset was rebuilt from an empty cache; totals below are the regenerated ones (one post-fix record is rejected as a near label conflict with an abstention record of the same `buger/jsonparser` function; one TypeScript fix added).

### Milestone 3 - Dataset seed `sentinel-garden-seed-v1` (full population, audited, rights queue)

Produced by the pipeline from real advisories and upstream fix commits (OSV export read 2026-11-04, snapshot bound 2026-11-01), **after** freezing the split policy `sentinel-garden-split-v1` (`configs/garden/split_policy.yaml`: salt `sentinel-garden-seed-v1-split`, 10 % validation, 10 % holdout, temporal cutoff 2026-07-01, sha256 `9aa83ed2…`; `check` fails if it changes). Discovery covers **every** GitHub-reviewed GHSA advisory of the snapshot for PyPI, npm and Go; the earlier 2,625 candidates are a strict subset of the current list. Full numbers: `data/garden/seed_v1/REPORT.md` / `stats.json`.

| Stage | Count |
| --- | ---: |
| GHSA advisories inspected (all of the snapshot) | 19,959 (Go 5,067, PyPI 6,813, npm 8,079) |
| Candidates (single fix commit, localizable CWE, max. 4 per repository) | 4,830 (Go 1,335, PyPI 1,524, npm 1,971) |
| Fix commits turned into records (one-function rule) | 1,206 |
| Records built | 2,773 |
| Records accepted | **2,322** |
| Records rejected (a record can fail several checks): audit exclusions - later fix of the same function 126, reported incomplete fix 47, unverifiable later fix 10, broad localization span 89, manual 10, invalid abstention 1, siblings of excluded records 104; pipeline checks - siblings 56, trivial span 29, label conflicts 36, diff inconsistency 14, duplicates 14, syntax 4 | 451 |
| **Training-approved records** | **0** |

Accepted records by language and label:

| Language | finding | no_finding (post-fix hard negative) | abstain | Total |
| --- | ---: | ---: | ---: | ---: |
| Python | 302 | 271 | 89 | 662 |
| JavaScript | 302 | 273 | 63 | 638 |
| TypeScript | 161 | 150 | 47 | 358 |
| Go | 288 | 266 | 110 | 664 |
| **Total** | **1,053** | **960** | **309** | **2,322** |

- 947 complete vulnerable/fixed pairs; 911 repositories; 161 distinct primary CWEs. Most frequent: CWE-79 (98), CWE-22 (95), CWE-1321 (78), CWE-400 (49), CWE-20 (43), CWE-200 (42), CWE-1333 (39), CWE-601 (33), CWE-78 (32). 165 findings carry a coarse CWE (flagged).
- Localization (diff-derived): 661 changed-line spans, 296 insertion points (missing checks), 96 mixed.
- Splits (repository-level, frozen policy): train 1,737 records / 685 repos, validation 191 / 81, holdout_reserved 235 / 89, temporal_quarantine 159 / 65. No repository is in two splits; no train/validation record is after the cutoff.
- Evaluation contamination: 0 hits against QuickEval v1, the public seed, `coding-v1` (and `private/`, which is empty here).
- Determinism: `curate` is byte-identical across runs; `verify` rebuilds every record from the frozen candidates with 0 hash mismatches. Go/JS/TS structure comes from tree-sitter with pinned grammars (recorded in `stats.json`, enforced by `check`).

**Quality audit** (`python -m sentinel.garden audit`, `data/garden/seed_v1/AUDIT_REPORT.md`, reproducible from the manifest and the Git cache; it checks consistency, it does **not** verify exploits):

- Automated, 1,157 fix groups: advisory↔code correspondence strong 661 / moderate 305 / weak 177; fix pattern consistent with the CWE family 696/1,030. Audit and curate form a fixed point: deduplication ignores audit exclusions, so a second audit run reproduces the exclusions, the audit results and the manifest byte for byte.
- Exclusions (all written to `audit/exclusions.yaml` and applied by `curate`; records are only ever excluded, never relabelled):
  - later fixes (OSV query of the same package, same function changed again with ≥ 0.5 token similarity to the post-fix code): 126 exclusions of post-fix records whose hard negative is not safe;
  - advisories that report an incomplete fix: 47;
  - a later fix commit of the same repository that cannot be read (not in the upstream repository): 10 exclusions in 8 groups, because the hard negative is then unverified; an incomplete recheck keeps the exclusions of the previous run;
  - **broad localization spans** (> 12 lines and > 40 % of the function): 89 groups, because every broad span in the manual sample was imprecise (it covered renames or docstrings committed with the fix);
  - abstentions whose caller passes only literals: 1 (near duplicates across splits are now rejected earlier, by deduplication).
- Stratified manual sample (44 items: 32 pairs + 12 abstentions, salted, per language × CWE family), reviewed by the implementing AI engineer (**not** an authorized reviewer; `audit/manual_review.yaml`; 21 items from the first review, 23 added by the larger population and reviewed in a second batch). Rates count only items of the current sample:
  - whole sample: wrong labels 0/32; localization errors 4/32 (12.5 %, Wilson 95 % 5.0–28.1 %); hard negative not safe 1/32 (0.6–15.7 %); CWE questionable 11/32 (kept, flagged); invalid abstentions 0/12 (0–24.3 %);
  - **items that survive the automated exclusions** (the estimate for the exported data): localization errors 1/29 (3.4 %, 0.6–17.2 %), hard negative not safe 1/28 (0.6–17.7 %), invalid abstentions 0/11 (0–25.9 %). Survival is decided per question on the record it judges (post-fix for hard-negative safety, abstention for abstention validity).
  - Manual exclusions (10 records): two groups with imprecise spans from the first review (golang-jwt/jwt, auth0/node-jsonwebtoken), two from the second (pyod, orval), and the gotenberg post-fix record, whose advisory reports a bypass of exactly that fix.
- The sample is small: the upper bounds, not the point estimates, are what to plan with.

**Rights review** (`python -m sentinel.garden review`, workflow `docs/RIGHTS_REVIEW.md`, queue `data/garden/seed_v1/REVIEW_QUEUE.md`): 766 repositories with train/validation records, each with the license file URL at the used revision, its sha256 and the deps.dev registry license. Tier A (permissive, consistent evidence) 513 repos / 1,260 records; tier B (permissive, open question) 111 / 330; tier C (weak copyleft) 15 / 33; tier D (strong copyleft, unknown, none) 127 / 305. deps.dev agrees for 559, disagrees for 140, unavailable 28, unrecognised 39; the license file changed between the used revisions in 23 repositories; a failed license lookup is recorded as missing evidence and never yields tier A. Approval templates (25 batches, tiers A and B) are not loadable until a reviewer fills them. Source dossiers for `ghsa-osv-fix-commits`, `osv` and `github-advisory-db` (terms, hashes, open questions) are in `docs/RIGHTS_REVIEW.md`. **Nothing is approved.** Non-permissive repositories need a written `non_permissive_justification` in addition to approval.

**Reconstruction and external resources:** `python -m sentinel.garden reconstruct` rebuilds the Git-ignored records from upstream for an approved subset and checks every record hash (no substitution when an upstream is gone). `data/garden/seed_v1/EXTERNAL_RESOURCES.json` lists every upstream repository and commit the dataset depends on; exports carry `SHA256SUMS`. No upstream code is committed or uploaded anywhere.

**Approval status:** all 2,322 accepted records are a **non-approved staging dataset**. 1,928 train/validation records are `pending_review`; 845 records (holdout, quarantine, rejected) are `excluded`. Blockers on every accepted record: `repository_not_reviewed`, `source_not_approved:ghsa-osv-fix-commits`, `source_not_approved:osv`, `source_not_approved:github-advisory-db`. `train.jsonl` and `validation.jsonl` are empty. Detected licenses (unreviewed, accepted records): 1,924 permissive, 188 strong copyleft (mostly AGPL-3.0), 35 weak copyleft, 68 unknown, 107 without a root license file.

Upstream code is **not committed** (redistribution unreviewed, repository license undeclared). The Git-ignored `artifacts/garden/seed_v1/` holds the full records; `data/garden/seed_v1/` holds candidates, manifest (metadata only), statistics, reports, audit results and the review queue.

### Milestone 4 - QLoRA training pipeline (PR #5, not run)

`python -m sentinel.train` (config `configs/train/qlora_qwen35_9b.yaml`, docs `docs/TRAINING.md`): 4-bit NF4 QLoRA with bf16 compute on the pinned foundation, LoRA r16/α32 on the 200 linear modules of the real architecture (24 Gated DeltaNet + 8 full-attention layers; 40.1 M trainable parameters). It loads only governance-approved `train.chat.jsonl` (labels and content hashes checked against the manifest, fail closed), trains the assistant reply only, checkpoints atomically with provenance (resume refuses a checkpoint trained on other data or config), writes adapter-only output and a `run.json` in every outcome (including CUDA OOM). CPU smoke tests train a tiny randomly initialised `Qwen3_5ForCausalLM` through the real loop. Nothing was trained; no weights were downloaded.

## QuickEval state

| Group | Python | JS/TS | Go | Total |
| --- | --- | --- | --- | --- |
| Vulnerable (`finding`) | 13 | 13 | 11 | 37 |
| Hard negative (`no_finding`) | 12 | 12 | 10 | 34 |
| Abstention (`abstain`) | 5 | 5 | 5 | 15 |
| **Total** | 30 | 30 (18 JS / 12 TS) | 26 | **86** |

- Public / synthetic: QuickEval v1 (86) and the 12-case harness seed, both registered `eval_only`, **not** a holdout.
- Private holdout: not started. Sentinel Garden's `holdout_reserved` records (235) are exported in eval-case format, but their identities are public, so they are a temporal/generalization reservation, not a private holdout.
- Public benchmarks (VulnBench, SeCodeBench, SeCodePLT, PrimeVul): registered evaluation-only. No cases imported.

## Benchmark state

No model has been evaluated. The only QuickEval v1 results are from the regex engineering baseline and are not model performance.

## Model state

Foundation **selected**: `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Not downloaded, not run, not modified. The QLoRA pipeline exists (Milestone 4) but has not been run: no adapter or training run exists.

## AWS state

No AWS resource was started. Planning assumptions are unchanged: `us-east-1`, G/VT Spot quota of 4 vCPUs, target `g6e.xlarge` (1× L40S 48 GB), roughly USD 1,160 of promotional credits. Quota, credits, Spot capacity and price must be verified right before any paid run. A paid run requires explicit approval.

## Blockers and open decisions

1. **Training approvals (blocks any training export).** An authorized reviewer must (a) approve the registry sources `ghsa-osv-fix-commits`, `osv` and `github-advisory-db` (dossiers in `docs/RIGHTS_REVIEW.md`; GHSA metadata is CC-BY-4.0 and needs attribution), and (b) review repositories in `data/garden/seed_v1/REVIEW_QUEUE.md` (tier A first) and record decisions in `data/garden/approvals.yaml`. Policy decision needed on copyleft (AGPL/GPL), unlicensed repositories and the 140 deps.dev disagreements.
2. Approval of paid GPU time for the foundation baseline run (QuickEval v1 + `coding-v1` on the 9B checkpoint).
3. A second, independent manual audit sample by a human reviewer (the current one is AI-reviewed and small; localization error upper bound 17 % after the automated exclusions).
4. Final LoRA rank/LR/sequence length after one short L40S calibration run of the existing pipeline.
5. Whether to replace `coding-v1` containment (child process + rlimits) with a real sandbox before running it on shared hosts.
6. Release thresholds after the foundation baseline; repository license (affects redistribution of project-authored eval cases and any future committed data).
7. Masterplan v1.2 itself is not in this repository; this milestone follows the task brief that summarizes it.

## Next action (toward the first QLoRA run)

1. Rights review: approve the three registry sources and review the tier A repositories in `REVIEW_QUEUE.md` (license file links and sha256 listed; fill `review/approvals_template_batch_*.yaml` and merge into `data/garden/approvals.yaml`). Tier A alone covers 513 repositories and up to 1,260 train/validation records. Then `python -m sentinel.garden curate` populates `train.jsonl`, `validation.jsonl` and `train.chat.jsonl` deterministically - no rebuild needed.
2. With approved GPU time, run the foundation baseline: `python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b --output runs/qwen35-9b-baseline.json`.
3. `python -m sentinel.train --dry-run` (passes once approvals exist), a short L40S calibration (`max_steps: 20`), then the first QLoRA run, evaluated with `run_model --adapter` on QuickEval v1, `coding-v1` and the Garden validation split against the baseline (`docs/TRAINING.md`).
