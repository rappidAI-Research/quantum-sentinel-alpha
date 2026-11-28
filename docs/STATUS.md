# Quantum Sentinel Alpha Status

Last updated: 2026-11-28

## Current phase

Phase 0 → 1. Completed milestones: QuickEval + real model runner + bake-off readiness (PR #2), pre-bake-off hardening, **foundation finalization (Qwen3.5-9B)** and **Sentinel Garden** (PR #4), the **QLoRA training pipeline** (PR #5, not run), the **Sentinel Garden seed over the full advisory population** with a quality audit and a rights review queue (PR #6), the **training-readiness hardening** of November 2026 (target integrity, no invented confidence, patch supervision, independent-review packages, checkpoint trust boundary; coding sandbox in a separate PR), the **pilot rights review** (PR #9), **Sentinel Garden v2 with the training-cohort rights review** (PR #10), the **training authorization** (PR #12) and **Sentinel Garden v3** (PR #13) and the **Ornith-1.5-9B foundation integration** (this milestone). The data work for the first training run is complete and **authorized** (`docs/TRAINING_AUTHORIZATION.md`, decisions of 2026-11-15 and 2026-11-24): Garden v3 holds 1,464 training-approved records from 533 repositories, i.e. 1,871 training and 187 validation examples (detection + patch), and `python -m sentinel.train --dry-run` passes. What remains is approved GPU time. No model has been run or trained. The training foundation is now **`ornith-ai/Ornith-1.5-9B@489cb97…`** (docs/FOUNDATION.md), integrated and checked offline against its real files; the GPU procedure is docs/GPU_RUNBOOK.md.

## Completed

### Training Optimization v2: bottlenecks and a sequential plan (2026-11-28, PR open, nothing trained)

`docs/TRAINING_OPTIMIZATION_V2.md`, `docs/GPU_RUNBOOK.md` § 17.
- **v1 results (reported):**
  - E2 (decision weight 10) step 200 calibrated is the selected candidate: accuracy 60.3 %, recall 41.6 %, FPR 15.2 %, strict recall 14.2 %.
  - E3 (detection only) passes rule v2 only at step 200.
  - E1 and E4 were not run.
- **Bottlenecks (derived from the reported operating points; to be confirmed with real AUCs):**
  - Discrimination: d′ 0.78–0.84 for E2, about AUC 0.72, no better than V3/V6. Recall rose mainly because the threshold moved.
  - Explanation: the share of true positives with correct CWE and lines is 29–47 % for every model, the foundation included.
- **Supervision audit (measured):**
  - all real findings are explained by advisory summaries;
  - 24–28 % insertion-point targets;
  - a length shortcut in train only (CodeQL tool tests);
  - no contradictory labels;
  - development has more out-of-excerpt context (44 %) and more MoreFixes cases (36 %) than the calibration set.
- **Added:**
  - a pairwise ranking objective on the 734 verified train pairs (V6/E2 path unchanged);
  - exact LoRA checkpoint averaging;
  - paired comparison with McNemar p;
  - true-positive quality and per-source analysis;
  - the supervision audit;
  - configs F1–F3;
  - `scripts/to2_gpu.sh`.
- **Next (needs approval, one stage at a time):**
  - F0, no training (≈ 2 GPU h);
  - F1, pair ranking;
  - F2, LR 5e-5;
  - F3 only if both pass.

  F0–F2 total ≈ 12–15 GPU h. Assumption: the release targets (AUC ≈ 0.93 needed) are not reachable with this foundation and data by training changes alone.

### Training Optimization v1: root cause analysis and controlled experiments (2026-11-26, PR open, nothing trained)

`docs/TRAINING_OPTIMIZATION_V1.md`, `docs/GPU_RUNBOOK.md` § 16.
- **V6 result (reported):** no checkpoint eligible under rule v2. Step 150 flags 97 of 212 cases (recall 0.575, FPR 0.323), step 175 flags 16 (0.124 / 0.020), and validation loss improves by 0.003 in between.
- **Audit:** order, shuffling, accumulation, loss normalization, masking, schedule, prompt/template, labels and checkpoint files were all verified; none explains the swing. Steps 151–175 have a normal class mix.
- **Verified limitations:**
  - The validation loss is blind to decisions: 0.78 % of its tokens are decision tokens and 65 % are patch tokens.
  - The decision is 0.87 % of the supervised training tokens.
  - Checkpoints every 25 steps cannot locate the swing; the epoch boundary after step 173 also falls between 150 and 175.
  - There was no way to move the decision threshold.
  - Training ran on an NF4 base and evaluation on bf16; the effect is unmeasured.
- **Added:**
  - order/token/log diagnostics;
  - decision scores (AUC, pairs, flips) with screening;
  - calibrated decoding with provenance (bias chosen on the training validation split only);
  - optional decision-token and patch-task weights (default = the V6 loss exactly);
  - validation decision metrics during training;
  - four one-variable experiment configs.
- **Owner decision (2026-11-26):** calibrated candidates in selection, the GPU experiments E0–E4 and the merge are approved (`docs/TRAINING_AUTHORIZATION.md`).
- **Next (on the GPU machine, `docs/GPU_RUNBOOK.md` § 16):**
  1. E0: no training, about 1.5–3 GPU hours.
  2. E1–E4: dense control, decision weight 10, detection only, LR 5e-5.
  3. Selection over all runs and the final test, once.

  All of it is about 20–25 GPU hours. Leading hypothesis: the decision threshold moved while the ranking was preserved. E0 tests it.

### Sentinel Garden v6: sources beyond GHSA/OSV (2026-11-25)

- **Sources** (`docs/GARDEN_V6.md` § 1): MoreFixes (CVE fix commits from NVD and GHSA, Zenodo dump of 2026-06-20) as a discovery index, and CodeQL's security query tests as rule-scoped tool tests. JsVul, PyVul, CVEfixes, Bandit/gosec, SecLLM, Go vulndb and PyPA were checked and not integrated (overlap, missing license, or labels without evidence). The OpenSSF CVE Benchmark is reserved for evaluation; its 99 Garden records are never trained on.
- **Evidence:** a MoreFixes fix commit must be named by the CVE's NVD references, a GHSA record or its own commit message; 1,501 CVEs with only heuristic fix commits are rejected. Tool-test markers, comments and telling names are removed; one finding per test file (template family).
- **Negatives:** the later-fix check now covers every known advisory of a repository; it refutes about 10 % of v5's post-fix negatives (160), which are excluded. Tool-test negatives stay pending like defended siblings.
- **Funnel:** 10,299 candidates + 185 tool-test units; 6,224 records, **5,055 accepted** (v5: 3,898); 222 CWEs among accepted findings.
- **Owner decision (2026-11-25):** the four new sources and 93 repositories (including `github/codeql`) are approved (`docs/TRAINING_AUTHORIZATION.md`). Before it, a license-detector bug (PSF read as GPL) was fixed.
- **Authorized: 2,170 records, 2,758 train examples (2,024 detection + 734 patch; +23 % over v5), 233,463 supervised tokens.** 50 v5 records were reserved for the OpenSSF benchmark and 15 refuted by the stricter check.
- **Still open:** the two negative kinds wait for the independent human review (2,891 examples with them); copyleft repositories need a written basis each (3,595 with them); unlicensed repositories cannot be approved. 5,000–10,000 are not reachable at this evidence bar (`docs/GARDEN_V6.md` § 7).
- **Evaluation:** development 239 cases, untouched final test 91 (v5: 9); selection rule v2 with recall ≥ 0.30 and FPR ≤ 0.20 floors.
- **Training:** fresh adapter, unchanged parameters, ready to start on the GPU machine (`docs/TRAINING_V6.md`, `docs/GPU_RUNBOOK.md` § 15). Not started.
- **Independent human review:** 147 packages, not performed.

### Sentinel Garden v5 (2026-11-25)

- **Owner decision:** no training yet, and later only a fresh adapter from Ornith-1.5-9B. The first run's adapter stays as the baseline; the continuation config was removed.
- **Garden v5** (`docs/GARDEN_V5.md`):
  - **Discovery:**
    - 8,727 candidates: the frozen v3 list plus 377 pull-request fixes (resolved to their merge commit) and 599 over the old repository cap;
    - new candidates wait for an extension of the source approval.
  - **Labels:**
    - combined excerpts need evidence for the labelled function (338 without it are dropped);
    - comments the fix added that name the weakness are left out;
    - caller notes;
    - a new negative kind, defended siblings (30 accepted), pending until its human-review sample validates it.
  - **Shortcuts:** the security-comment cue is gone (6.7 % vs 6.9 %). Surface features predict finding vs no_finding 2.3 points above chance (v4: 6.4).
  - **Funnel:**
    - 4,831 records, 3,898 accepted;
    - **1,741 authorized** after the owner's approvals of 2026-11-25 (85 repositories, source extension to the v5 list);
    - **2,249 train examples** (193 k supervised tokens); ready for the fresh training run.
    - The defended-sibling kind (11 more train records) waits for its independent human review.
    - 5,000–10,000 are not reachable at this quality bar; the measured reasons are in `docs/GARDEN_V5.md` § 6.
  - **Contamination:** 0 shared identities or near-duplicates with any v1–v5 evaluation set.
  - **Evaluation:**
    - development 163 cases;
    - untouched final test only 9 cases;
    - the 453-case previous holdout is a valid final test only if the v3 holdout was never evaluated (owner to confirm).
  - **Independent human review:** 80 packages, not performed.

### Sentinel Garden v4 and second training phase readiness (2026-11-25)

- **Development result of the first run:** rule v1 selected `step-000200` (decision accuracy 63.2 %, recall 34.9 %, strict recall 12.7 %, FPR 14.8 %, abstention recall 100 %). Alpha is not ready for release.
- **Measured v3 weaknesses** (`docs/GARDEN_V4.md` § 1):
  - an abstention-only context sentence (100 % / 0 %) that a rule can match;
  - multi-block excerpts that were always hard negatives;
  - hard negatives that are only post-fix twins;
  - thin CWE families.
  - The per-case development artifacts are on the GPU machine; they are requested for the error analysis.
- **Garden v4** (`docs/GARDEN_V4.md`):
  - same frozen candidate list, so no new source authorization;
  - 346 recovered fix groups (multi-function fixes in one file, contiguous commit chains);
  - context notes on every record kind;
  - stricter cross-split leakage checks (forks, copied fixes, contained excerpts);
  - evaluation roles: the v3 development set is retired; a fresh development set (176 cases, 59 repositories); a final test set no version exported before (45 cases).
  - Contamination audit against every evaluation set of v1–v4: 0 shared identities or near-duplicates; 10 records share a package with the temporal quarantine, by design.
  - **Training-approved: 1,524 records (v3: 1,464)**, i.e. 1,943 train / 200 validation examples.
  - Rights review queue: 1,417 accepted records wait for the owner; 128 repositories with 342 records are decision-ready (2,440 train examples if approved).
  - Independent human review package: 80 items, decisions pending.
- **Training (not run)** (`docs/TRAINING_V4.md`):
  - A continues from `step-000200` with a new optimizer and schedule (1 epoch, LR 5e-5);
  - B is a fresh adapter on v4 (2 epochs, LR 1e-4).
  - Both are bound and checked offline. Selection: rule v1 across both runs on the v4 development set.
  - GPU commands: `docs/GPU_RUNBOOK.md` § 13.

### First QLoRA run and post-training evaluation (2026-11-24)

- **First real QLoRA run** completed on AWS `g6e.xlarge`: Ornith-1.5-9B, 1,871 examples, 2 epochs, 234 optimizer steps; adapter and checkpoints backed up (not in this repository).
- **QuickEval v1 results** reported by the GPU session: F1 0.886 → 0.914, recall 0.838 → 1.000, strict recall 0.595 → 0.811, CWE accuracy 0.710 → 0.811, abstention recall 0.400 → 0.800, decision accuracy 0.779 → 0.884, coding-v1 9/16 → 11/16. **False positive rate rose from 0.059 to 0.206** (precision 0.939 → 0.841). The adapter run is recorded as non-reportable because the evaluator counted the FP32 LoRA parameters as foundation parameters.
- **Evaluator fix:** foundation and adapter parameters are now verified separately and by identity, and checkpoint adapters are checked against their `checksums.json`.
- **Next evaluation (`docs/EVALUATION.md`):**
  - development set from the approved validation split (`sentinel.eval.devset`);
  - checkpoint comparison with a pre-registered rule (`sentinel.eval.select_checkpoint`);
  - final independent test on `holdout_reserved`, once.
  - QuickEval v1 has been inspected and is now a regression check only.
  - No development or holdout result exists yet.

### Ornith-1.5-9B foundation integration (2026-11-24)

- **Active training foundation: `ornith-ai/Ornith-1.5-9B` at revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`** (MIT per model card; `Qwen3_5ForConditionalGeneration`, loaded text-only as `Qwen3_5ForCausalLM`, 8.95 B text parameters). Decision basis `expected_capability` (project owner instruction); no measured security advantage is claimed. Qwen3.5-9B stays documented in `history` and as an optional reference.
- Verified offline against the real files (no weights): same config dimensions, tensor names and shapes as Qwen3.5-9B; byte-identical tokenizer; the chat template's non-thinking generation prompt is identical, so training prompt = evaluation prompt and only the JSON reply plus `<|im_end|>` is supervised. LoRA: 200 targets, 40,108,032 parameters, module list regenerated from Ornith's config (`scripts/inspect_foundation.py`). Garden v3 re-encoded with the Ornith tokenizer: 0 of 2,058 examples dropped at 4,096 tokens (max 2,918). Data hashes unchanged.
- Fixed: the evaluation backend now stops generation at `<|im_end|>` and decodes nothing after the first end of turn. The checkpoint has no `generation_config.json`, so the default EOS would have been `<|endoftext|>` (latent for Qwen3.5-9B too). The trainer refuses a chat template that ignores `enable_thinking=False`. New: `python -m sentinel.train --dry-run --check-tokens`.
- Baseline config `configs/model/baseline_ornith15.yaml` (candidate `ornith15-9b`, checked against the foundation pin). The Qwen3.5 4B vs 9B bake-off is archived.
- Caveat: Ornith was published on 2026-08-18, after the frozen temporal cutoff (2026-07-01). 183 of the 355 quarantine records were fixed before that date and are reported separately in evaluations. The split policy is unchanged.
- Not verified without a GPU: JSON-contract reliability of the untuned model, NF4 load and peak memory, throughput.

### Foundation and evaluation (earlier milestones)

- Versioned finding schema, Sentinel Garden source registry (default-deny), threat model, data-governance rules, QuickEval case/prediction contracts, deterministic engineering baseline and scorer.
- QuickEval v1 (`eval/sentinelbench/quickeval_v1.jsonl`): 86 clean-room synthetic cases with hashed provenance, validated line spans, accepted alternate CWEs and 30 vulnerable/secure neighbour pairs.
- Shared evaluation path (`sentinel.eval.run_model`): one prompt (`quickeval-review-v1`), one strict output parser and one scorer for every model; malformed output is scored `invalid`, never repaired; reasoning is stripped and never stored.
- Hugging Face backend (`AutoModelForCausalLM`, device/dtype/bitsandbytes/generation settings), reportability gate (immutable revision, loaded revision and dtype match, full case file, clean Git commit), JSON run artifacts with security and efficiency metrics, `sentinel.eval.compare` (no automatic winner).
- Coding-regression suite `coding-v1` (16 clean-room Python tasks; generated code runs only in a fail-closed container sandbox, see below), `scripts/pin_hf_revisions.py`, GitHub Actions CI (`pytest` + `python -m sentinel.foundation`).

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
| Records accepted | **2,309** |
| Records rejected (a record can fail several checks): audit exclusions - later fix of the same function 126, reported incomplete fix 47, unverifiable later fix 10, broad localization span 89, manual 10, invalid abstention 1, siblings of excluded records 104, **package shared with validation/holdout 13** (November 2026); pipeline checks - siblings 56, trivial span 29, label conflicts 36, diff inconsistency 14, duplicates 14, syntax 4 | 464 |
| **Training-approved records** | **0** |

Accepted records by language and label:

| Language | finding | no_finding (post-fix hard negative) | abstain | Total |
| --- | ---: | ---: | ---: | ---: |
| Python | 302 | 271 | 89 | 662 |
| JavaScript | 300 | 271 | 63 | 634 |
| TypeScript | 161 | 150 | 47 | 358 |
| Go | 284 | 262 | 109 | 655 |
| **Total** | **1,047** | **954** | **308** | **2,309** |

- 941 complete vulnerable/fixed pairs (all 941 are patch ground truth, docs/PATCH_SUPERVISION.md); 906 repositories; 161 distinct primary CWEs. Most frequent: CWE-79 (98), CWE-22 (95), CWE-1321 (78), CWE-400 (49), CWE-20 (43), CWE-200 (42), CWE-1333 (39), CWE-601 (33), CWE-78 (32). 165 findings carry a coarse CWE (flagged).
- Localization (diff-derived): 661 changed-line spans, 296 insertion points (missing checks), 96 mixed.
- Splits (repository-level, frozen policy): train 1,724 records / 680 repos, validation 191 / 81, holdout_reserved 235 / 89, temporal_quarantine 159 / 65. No repository is in two of train, validation and holdout_reserved, and (since November 2026) no published package spans train and validation/holdout; no train/validation record is after the cutoff. `temporal_quarantine` holds post-cutoff fixes of training repositories by design: 20 of its 159 records share a repository and 6 a package with train/validation, so results on it are not repository-disjoint.
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

**Rights review** (`python -m sentinel.garden review`, workflow `docs/RIGHTS_REVIEW.md`, queue `data/garden/seed_v1/REVIEW_QUEUE.md`): 761 repositories with train/validation records, each with the license file URL at the used revision, its sha256 and the deps.dev registry license. Tier A (permissive, consistent evidence) 511 repos / 1,256 records; tier B (permissive, open question) 110 / 328; tier C (weak copyleft) 15 / 33; tier D (strong copyleft, unknown, none) 125 / 298. deps.dev agrees for 557, disagrees for 137, unavailable 28, unrecognised 39; the license file changed between the used revisions in 23 repositories; a failed license lookup is recorded as missing evidence and never yields tier A. Approval templates (25 batches, tiers A and B) are not loadable until a reviewer fills them. Source dossiers for `ghsa-osv-fix-commits`, `osv` and `github-advisory-db` (terms, hashes, open questions) are in `docs/RIGHTS_REVIEW.md`. **Nothing is approved.** Non-permissive repositories need a written `non_permissive_justification` in addition to approval.

**Reconstruction and external resources:** `python -m sentinel.garden reconstruct` rebuilds the Git-ignored records from upstream for an approved subset and checks every record hash (no substitution when an upstream is gone). `data/garden/seed_v1/EXTERNAL_RESOURCES.json` lists every upstream repository and commit the dataset depends on; exports carry `SHA256SUMS`. No upstream code is committed or uploaded anywhere.

**Approval status:** all 2,309 accepted records are a **non-approved staging dataset**. 1,915 train/validation records are `pending_review`; 858 records (holdout, quarantine, rejected) are `excluded`. Blockers on every accepted record: `repository_not_reviewed`, `source_not_approved:ghsa-osv-fix-commits`, `source_not_approved:osv`, `source_not_approved:github-advisory-db`. `train.jsonl` and `validation.jsonl` are empty. Detected licenses (unreviewed, accepted records): 1,918 permissive, 186 strong copyleft (mostly AGPL-3.0), 35 weak copyleft, 65 unknown, 105 without a root license file.

Upstream code is **not committed** (redistribution unreviewed, repository license undeclared). The Git-ignored `artifacts/garden/seed_v1/` holds the full records; `data/garden/seed_v1/` holds candidates, manifest (metadata only), statistics, reports, audit results and the review queue.

### Milestone 4 - QLoRA training pipeline (PR #5, not run)

`python -m sentinel.train` (config then `configs/train/qlora_qwen35_9b.yaml`, now `configs/train/qlora_ornith15_9b.yaml`; docs `docs/TRAINING.md`): 4-bit NF4 QLoRA with bf16 compute on the pinned foundation, LoRA r16/α32 on the 200 linear modules of the real architecture (24 Gated DeltaNet + 8 full-attention layers; 40.1 M trainable parameters). It loads only governance-approved `train.chat.jsonl` (labels and content hashes checked against the manifest, fail closed), trains the assistant reply only, checkpoints atomically with provenance (resume refuses a checkpoint trained on other data or config), writes adapter-only output and a `run.json` in every outcome (including CUDA OOM). CPU smoke tests train a tiny randomly initialised `Qwen3_5ForCausalLM` through the real loop. Nothing was trained; no weights were downloaded.

### Coding-regression sandbox (runner version 3)

Model-generated code from `coding-v1` runs only in a disposable container (`sentinel/eval/sandbox.py`, policy `configs/eval/coding_sandbox.yaml`, docs `docs/CODING_SANDBOX.md`): digest-pinned `python:3.12-slim` image, `--network none`, read-only root file system and task directory, user 65534, no capabilities, no host environment or credentials, CPU/memory/process/file-size/time limits. A probe container must observe this isolation before the model is loaded, otherwise the run stops (exit 3); `--skip-coding-regression` runs security evaluation only and records `not_run`. The previous child process (same user, file system and network) is kept only for trusted reference solutions. Containers share the host kernel; a disposable evaluation host is still recommended.

### Milestone 5 - Training-readiness hardening (November 2026, not run)

Integrated on top of the coding sandbox (PR #7, merged as `13dae45`): one evaluation path checks the adapter's base model, revision and training prompt, then the sandbox preflight, then loads the model; the review prompt is `quickeval-review-v2`, and runs are compared only with identical prompt, parser, scorer and coding contract (which includes the sandbox policy).

| Area | Problem found | Change | State |
| --- | --- | --- | --- |
| Target integrity | The loader compared label fields and the code hash only. Evidence text, the upstream patch, the prompt context and file name could be edited in `train.jsonl` and re-rendered into `train.chat.jsonl` without detection, and records could be dropped from the export. | Manifest `supervision` block: canonical sha256 of the full input, the full target and every rendered example (detection and patch). The loader recomputes them, requires exactly the manifest's approved records per split and checks `SHA256SUMS`; nothing is repaired. | Implemented and tested (`tests/test_train_integrity.py`) |
| Confidence | Every training finding carried `"confidence": 0.9`, a constant presented as a probability. | Prompt `quickeval-review-v2` and the training replies carry no confidence; parser v3 accepts findings with or without it (v1 outputs parse unchanged; `quickeval-review-v1` is byte-identical and pinned); schemas allow `null`; calibration metrics and a measurement protocol (decision-prefix likelihoods, fit on validation, report on held-out data) in `sentinel/eval/calibration.py`, `docs/CALIBRATION.md`. | Implemented and tested; no calibration measured |
| Patch supervision | Stored upstream patches were never supervised. | Separate task `sentinel-patch-v1`: ground truth only from accepted vulnerable records whose accepted fixed sibling is reproduced exactly by the strictly applied patch; deterministic `train.patch.chat.jsonl` (opt-in `data.tasks: [detection, patch]`) and `holdout_reserved.patch_cases.jsonl`; strict diff parser; offline scorer (well formed, applies, syntax, exact/normalized match, span, size) and `python -m sentinel.eval.patch`. Security is never claimed. | Implemented and tested; functional/security tests not possible without per-project environments (`docs/PATCH_SUPERVISION.md`) |
| Data quality | The AI-reviewed sample is not independent; the audit missed one leakage path. | Blind, reproducible evidence packages for 64 items of the exportable population (`review-packages`), committed index with package hashes, template and validator for `audit/human_review.yaml`, human rates reported separately. New leakage check: a package published from two repository keys (moved, renamed, forked repositories such as `visionmedia/send` → `pillarjs/send`, `MrRio/jsPDF` → `parallax/jsPDF`) excluded 13 train records; quarantine overlap reported. Audit + curate remain a byte-identical fixed point. | Implemented and tested; **human review not performed** |
| QLoRA reliability | Resume unpickled `trainer_state.pt` with `weights_only=False` and no integrity check; the logged step loss was the last micro-batch times the window; adapters did not record their prompt. | Checksummed checkpoints, `weights_only=True`, step-mean loss, prompt and task mix in `training_run.json` with an evaluation-time prompt check. Verified with the real Qwen3.5-9B tokenizer and weight index (no weights): masking, prompt identity and weight-name mapping are correct. | Fixed and tested (CPU smoke tests; GPU not run) |

### Pilot rights-review package (November 2026, proposal only)

`docs/PILOT_DATASET_RIGHTS_REVIEW.md` with `data/garden/seed_v1/pilot/cohort.json` and `proposed_approvals.yaml` (templates, not loadable), generated by `scripts/pilot_cohort.py`: a source-level dossier for `github-advisory-db` (CC-BY-4.0, attribution), `osv` (access path; GHSA content) and `ghsa-osv-fix-commits` (per-repository licenses), and a balanced pilot cohort of 50 Tier A repositories inspected at every used revision for NOTICE files, nested license files and package metadata (final inspection: every revision complete, 0 failed reads; evidence is bound to the exact records, revisions and paths and is invalidated when they change; a failed read is recorded as incomplete evidence, never as absence). 48 are recommended for review (MIT 31, Apache-2.0 14, BSD-2/BSD-3/ISC 1 each), 2 need clarification (a license conflict in `setup.py`; an Apache NOTICE in an MIT project). If the three sources and the 48 repositories were approved, `curate` would export 129 records (train 103, validation 26) and 52 patch examples. Open legal questions (weights as redistribution, GitHub AUP §7, OSV service terms) are documented, not resolved. **Nothing is approved; 0 records are training-approved.**

### Garden v2 and the training cohort (November 2026, proposal only)

`docs/GARDEN_V2.md`, `docs/TRAINING_COHORT_RIGHTS_REVIEW.md`. **Sentinel Garden v2** (`configs/garden/seed_v2.yaml`, `data/garden/seed_v2/`) is the training dataset; v1 stays the immutable reference (all 2,773 v1 records rebuild with identical hashes; every v1 record is in v2 with the same content and split). Same snapshot, thresholds and frozen split policy; differences: repository cap 4 -> 20 and three extraction refinements that are off for v1 (complete helper methods added in a class body, version-string bumps, build/test tooling files). The rejection analysis showed that most `change_outside_function` and `multiple_functions_changed` rejections are genuine (module-level code, the same fix in several functions); the refinements recovered 58 v1 candidates.

| | v1 | v2 |
| --- | ---: | ---: |
| Records accepted | 2,309 | **3,226** (+917) |
| finding / no_finding / abstain | 1,047 / 954 / 308 | 1,476 / 1,307 / 443 |
| Verified vulnerable/fixed pairs | 941 | 1,278 |
| train / validation / holdout / quarantine | 1,724 / 191 / 235 / 159 | 2,276 / 252 / 376 / 322 |
| Finding CWEs | 161 | 192 |

22.7 % of the 1,198 new records were rejected by the unchanged quality checks and audit; audit + curate is a byte-identical fixed point; `garden verify` rebuilt all 3,971 v2 records with 0 hash differences. **Training cohort** (`scripts/training_cohort.py`, hardened inspection from PR #9 plus precise package-metadata parsing and confirmed-absence handling): all 550 tier A repositories; 24 excluded because the training loader refuses repositories that also have holdout/quarantine records; of the other 526, inspected at every used revision (0 failed reads), **484 are recommended for review** (MIT 318, Apache-2.0 120, BSD-3-Clause 30, BSD-2-Clause 9, ISC 7) and 42 need clarification (36 NOTICE/additional license files, 6 package-metadata conflicts). If the three sources and the 484 repositories were approved: **1,297 detection examples (train 1,177, validation 120) and 528 patch examples (479/49)**, 123 CWEs - verified end to end with the real `curate` export and training loader on a scratch copy with simulated approvals (`scripts/authorization_preflight.py`); every example fits 4,096 tokens with the real Qwen3.5-9B tokenizer (max 2,919). The 64 independent-review packages are now drawn from that cohort. **Decision 2026-11-15** (`docs/TRAINING_AUTHORIZATION.md`): the three sources and the 484 recommended repositories are approved by the project owner; `curate` produced exactly the projected 1,297 training-approved records (train 1,177, validation 120; patch 479/49), and only governance fields of the manifest changed. The 42 clarification cases and all other repositories stay default-deny.

### Garden v3 and the extended training data (November 2026)

`docs/GARDEN_V3.md`. **Sentinel Garden v3** (`configs/garden/seed_v3.yaml`, `data/garden/seed_v3/`) is the training dataset; v1 and v2 stay immutable (every v2 record and candidate is in v3 unchanged; `garden verify` rebuilt all 4,200 v3 records with 0 differences; audit + curate at a fixed point).

- **Multi-commit advisories:** advisories naming several fix commits in one repository are used when exactly one fix can be established (one code commit plus tests/docs commits, or identical backports/merge commits: same lines at the same position in the same function); 100 of 1,128 yield records, the others are rejected as distinct or unexplained changes (sampled: genuine). Accepted records 3,226 -> 3,427, verified pairs 1,278 -> 1,363.
- **Loader-consistent curation:** records of a repository that also has holdout/quarantine records are never exported (the loader would refuse them); this prevented a training failure in v3 (2 approved repositories, 6 records).
- **Tier B extension of the rights review** (`scripts/training_cohort.py --extension`): tier B repositories whose questions are formal (no registry cross-check, registry lists bundled licenses, license text updated with the same license, other permissive license) are inspected like tier A; approvals can list further reviewed versions of a license file (`additional_license_file_sha256s`, schema + gate). Package-metadata parsing for setup.cfg/pyproject/setup.py and license spellings made precise.
- **`materialize`:** a training machine rebuilds only the approved records (684 advisories in 533 repositories instead of 7,751), byte-identical to the full `curate` export, checked against every manifest hash.
- **Decision 2026-11-24:** sources extended to v3, 51 tier B repositories approved, training tasks detection + patch. Training data: detection 1,331 / 133, patch 540 / 54; 668 findings over 131 CWEs; every example within 4,096 tokens with the real tokenizer (1.73 M tokens per epoch). Not approved (prepared, default-deny): 26 new tier A repositories, 16 with updated license text, 2 Unlicense, all clarification cases.

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

No model has been evaluated. The only QuickEval v1 results are from the regex engineering baseline and are not model performance. No patch-generation result and no calibration result exists.

## Model state

Foundation **selected**: `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` (until 2026-11-24: `Qwen/Qwen3.5-9B@c202236…`). Not downloaded, not run, not modified. The QLoRA pipeline is ready and its dry run passes on the approved data, but no adapter or training run exists.

## AWS state

No AWS resource was started. Target `g6e.xlarge` (1× L40S 48 GB, 4 vCPUs, 32 GiB) On-Demand in `us-east-1`; On-Demand GPU quota of 4 vCPUs approved; about USD 1,160 of credits last reported. Verify credits and price right before the run. A paid run requires explicit approval. Procedure: `docs/GPU_RUNBOOK.md`.

## Blockers and open decisions

1. **Approval of paid GPU time** (foundation baseline and the first QLoRA run). Training data is authorized (`docs/TRAINING_AUTHORIZATION.md`).
2. Recommended before reporting results: an independent human reviewer for the 64 evidence packages of the approved cohort (`data/garden/seed_v3/audit/human_review_sample.jsonl`, drawn from the training-approved repositories). The existing manual review is AI-assisted and not independent; exclusions would change the dataset.
3. Optional later: 26 further tier A repositories, 16 tier B repositories with an updated license text, 2 Unlicense repositories and the clarification cases, all prepared in `data/garden/seed_v3/cohort/` (`COHORT_REVIEW.md`).
4. Final LoRA rank/LR/sequence length after one short L40S calibration run of the existing pipeline.
5. Release thresholds after the foundation baseline; the open legal questions on weights as redistribution and GitHub AUP §7 before any non-open or commercial release.
6. Masterplan v1.2 itself is not in this repository; this milestone follows the task brief that summarizes it.
7. Before any weight release: the Ornith model card links a `LICENSE` file that is absent at the pinned revision (license MIT per metadata; Qwen3.5 base under Apache-2.0).

## Next action (toward the first QLoRA run)

Follow `docs/GPU_RUNBOOK.md` on `g6e.xlarge`:

1. Approve GPU time; check credits and price.
2. `python -m sentinel.garden materialize --config configs/garden/seed_v3.yaml`, then `python -m sentinel.train --dry-run --check-tokens`.
3. Foundation baseline: `python -m sentinel.eval.run_model --bakeoff configs/model/baseline_ornith15.yaml --candidate ornith15-9b --output runs/eval/ornith15-9b-baseline.json`.
4. Optional 20-step calibration, then the first QLoRA run (`python -m sentinel.train`), detection and patch generation.
5. Adapter evaluation with `run_model --adapter` on the same config and `sentinel.eval.compare` against the baseline; save adapter, `run.json` and evaluation artifacts before terminating the instance.
