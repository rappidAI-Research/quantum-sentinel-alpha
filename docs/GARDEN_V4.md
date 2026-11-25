# Sentinel Garden v4

`sentinel-garden-v4` (`configs/garden/seed_v4.yaml`, committed metadata in `data/garden/seed_v4/`) is the dataset for the second training phase. It builds on Garden v3 and leaves v1–v3 unchanged: their committed files and manifests are untouched, and `check` passes for all four versions.

## Summary

- v4 **does not train on more approved data yet**: 1,524 training-approved records against 1,464 in v3 (+4 %).
- What it changes:
  - two shortcuts are removed from the inputs;
  - 346 new fix groups were recovered from fixes v3 had to reject;
  - every training record is checked against every evaluation set any version produced;
  - there are fresh development and final test sets.
- The larger lever is the rights decision on 1,417 accepted records that wait for review. 128 repositories with 342 of these records are prepared for your decision.
- **5,000 to 10,000 approved examples are not reachable from this source without lowering the bar.** The section "Why not 5,000 to 10,000" gives the measured reasons and the remaining levers.

## 1. Weaknesses of Garden v3 (measured)

`python -m sentinel.garden analyze` (`sentinel/garden/analysis.py`) writes aggregate numbers only (`data/garden/seed_v4/analysis/sentinel-garden-v3.json`, `…-v4.json`). Shortcut probes are fitted on the train split and scored on the repository-disjoint validation split.

**Development results.** The per-case artifacts of the first run are on the GPU machine, not here. From your aggregates on 133 cases:
- 22 of 63 findings were detected, 8 of them with correct CWE and location;
- 8 false positives on 54 hard negatives;
- 16 of 16 abstentions.

The 41 missed findings are `no_finding` or `abstain`; which one needs `runs/eval/dev/step-000200.json`. Error analysis by CWE family and language runs as soon as the artifacts exist (`analyze --runs … --cases …`, GPU_RUNBOOK § 13). Nothing below is inferred from per-case results.

| Weakness | Evidence (v3, approved or accepted data) | v4 |
| --- | --- | --- |
| **Abstention shortcut** | The context sentence "`f` is defined elsewhere in …; its body is not included." appears in **100 % of abstention records and 0 % of all others**, in train and validation. A context-only rule reaches 100 % abstention precision and recall. This matches the 100 % development abstention recall; whether the model used it is unproven. | Every record now names the same-file functions its excerpt calls but does not show (`callee_note`). The sentence is gone (0 records). The note is on 100 % of abstentions and on 36 % of findings and of hard negatives, so its presence no longer decides the label. |
| **Multi-block cue** | Excerpts made of several file blocks (a fix that added a helper) were **100 % no_finding** (15 validation records). | 60 % no_finding (62 records): combined multi-function excerpts are findings with several blocks. |
| Lexical "defensive token" cue | Naive Bayes over code tokens: 60 % accuracy (majority 56 %), but it ranks the vulnerable version above its own fix in **78 %** of pairs. Fixes add checks and escapes. | 79 %: inherent to fix-pair data; not removable without other negatives. |
| Length | best length threshold 52 % (no cue) | 56 % (majority rate) |
| Hard negatives | every hard negative is the post-fix twin of a finding; there is no unrelated secure code. The development FPR therefore measures only near-identical twins. | unchanged (see "Not done") |
| Thin CWE families | 668 findings over 131 CWEs; SQL injection 10, code injection 22, crypto/TLS 29, redirect/SSRF 32 | 705 findings over 134 CWEs (SQL 11, code injection 22, crypto 30, redirect/SSRF 32) |
| Coarse labels | 194 of 668 findings have a coarse CWE (CWE-20, -400, …) or an alternate | unchanged method |
| Localization | median span 2 lines; 28 % insertion points; broad spans (> 12 lines and > 40 % of the function) were already excluded | the same rule, measured against the labelled function of a combined excerpt |
| Context | function plus up to 12 import lines; callee bodies are never shown | plus the signatures of called, not shown same-file functions |
| Patch coverage | 594 of 1,464 approved records are patch-eligible (verified pairs) | 619 of 1,524 |
| Diversity | 533 repositories; the 10 largest hold 7.9 % of the records | 529 repositories; 8.6 % |

## 2. What changed

**1. Same frozen candidate list.**
- v4 does not query OSV again. `discover` copies the v3 list and checks its sha256 (`535494cb…`).
- The source approvals in `data/registry.yaml` are bound to that list and therefore cover exactly the same advisories and fix commits. No new source needs authorization.
- Records newly extracted from **approved repositories** become training-approved under the existing per-repository approvals: same repositories, same licences, same fix commits. Revoke an approval entry if you want to review the new kinds first. 68 records are affected (`excerpt_functions` in the manifest).

**2. Recovered fixes.** All of these are config-gated in `sentinel/garden/build.py`; the v1–v3 configs build exactly as before.

| Recovery | Rule | Candidates built | Accepted groups |
| --- | --- | ---: | ---: |
| Multi-function fixes | 2–3 changed functions in **one** file, ≤ 60 changed lines, no nested functions, each function within the size bounds. Excerpt = the functions in file order, one blank line apart; fixed excerpt likewise, plus added helpers. | 424 (from 1,051 + 536 rejected) | 336 |
| Commit chains | the advisory's commits form **one contiguous first-parent chain** (each commit's parent is the previous one, no merge commit). The combined diff is then exactly the union of the listed commits; the fix date is the last commit's. | 11 (from 778 rejected) | 10 |

**Label for a combined excerpt:**
- `finding` with the advisory's CWE.
- The span is one changed site: the function with the most replaced lines, then the most changed lines, then the earliest. Every site is recorded in `localization.sites`. The output contract has one span; the evaluation scores against the labelled one.
- The post-fix excerpt is a hard negative **for the advisory's weakness only** (`label_scope`), exactly as for single functions.
- No abstention record: one withheld function per abstention.

**Rejected as before:**
- logic changes in several files;
- more than 3 functions;
- nested functions (`overlapping_fix_blocks`, 86);
- non-contiguous commit sets (the backport vs. partial-fix question needs full history).

**3. Context notes.** See section 1.
- An abstention record is built only if its withheld callee is among the first 8 named callees, so it is named like any other. 3 v3 abstention records fail that and are not in v4: 1 accepted, approved; 2 were already rejected.
- Contexts change for 49 % of the v3 sample. Code, labels, spans and patches are unchanged (checked on 300 v3 fixes: 694 records identical apart from the context).

**4. Evaluation roles.** These give a fresh development set and an untouched final test set (`docs/EVALUATION.md`).

**5. Stricter leakage audit** (`quality.cross_split_containment`, `audit.split_leakage`):
- a training record is also excluded when it shares a **vulnerable** revision with a held-out record (forks share history);
- or when it shares the **fix fingerprint** (the same removed and added lines: a vendored copy or a fork's fix);
- or when it **contains** ≥ 80 % of a held-out excerpt's shingles. A combined excerpt can contain a held-out function without resembling it as a whole.

**6. Tooling:**
- `build --processes N` (the same result, all cores);
- `materialize-eval` (rebuilds and verifies only the evaluation-role records);
- `analyze`.

## 3. Size and composition (verified)

| | v3 | v4 |
| --- | ---: | ---: |
| Candidates (identical list) | 7,751 | 7,751 |
| Fixes turned into records | 1,825 | 2,257 |
| Records built / accepted | 4,200 / 3,427 | 5,065 / **4,047** |
| finding / no_finding / abstain (accepted) | 1,566 / 1,392 / 469 | 1,903 / 1,674 / 470 |
| Verified vulnerable/fixed pairs | 1,363 | 1,643 |
| Distinct finding CWEs (accepted) | 194 | 208 |
| Repositories with accepted records | 1,053 | 1,233 |
| **Training-approved records** | **1,464** | **1,524** |
| Approved train examples (detection + patch) | 1,331 + 540 | 1,382 + 561 |
| Approved validation examples (validation loss only) | 133 + 54 | 142 + 58 |

Of the 1,464 v3-approved records:
- 1,455 are approved in v4;
- 6 are excluded because their repositories gained post-cutoff records (the training loader refuses such repositories);
- 1 is excluded by the stricter leakage audit, together with 1 sibling;
- 1 abstention record is no longer built (context notes, above).

The approved v4 data covers 705 findings, 634 hard negatives and 185 abstentions:
- Languages: Go 433, JavaScript 436, Python 422, TypeScript 233.
- Families: resource limits/ReDoS 100, path traversal 85, other 78, XSS 73, authn/authz 75, prototype pollution 59, deserialization/input validation 53, information exposure 49, command injection 38, redirect/SSRF 32, crypto/TLS 30, code injection 22, SQL injection 11.
- Tokens with the pinned tokenizer: 0 examples dropped at 4,096; median 781, maximum 3,176.

**Quality, automated and unchanged in method.**
- The audit reached a fixed point: two consecutive runs produced byte-identical exclusions.
- `verify` rebuilt every record from the pinned revisions and compared the content hashes.
- `materialize` reproduces the curated training exports byte for byte.
- Automated exclusions:
  - later fix of the same function: 326;
  - broad span: 195;
  - reported incomplete fix: 106;
  - unverifiable later fix: 78;
  - split leakage: 30;
  - invalid abstention: 2.

**AI-assisted spot check of the new label type** (`audit/manual_review.yaml`, the same assisted reviewer and method as v1–v3; not an independent review). Of the 7 combined multi-function pairs in the audit sample:
- 4 are correct with precise or acceptable localization;
- 2 have broad spans, which the automated rule already excludes;
- 1 has a questionable CWE (unauthenticated CBC labelled CWE-327) and an imprecise span. It is excluded by hand.

Seven items give no rate. The independent human review below samples 16 such pairs.

## 4. Contamination audit (all versions)

`analysis/sentinel-garden-v4.json` → `contamination`. The 2,811 accepted training-role records were compared with 3,793 evaluation records:
- the v1–v3 validation, holdout and quarantine records;
- the v4 retired development, development, final test, previous holdout and quarantine records.

| Check | Training records sharing it |
| --- | ---: |
| Advisory ID or alias (CVE, GO-, PYSEC-) | 0 |
| Fix or vulnerable revision in another repository (forks) | 0 |
| Exact normalized code | 0 |
| Fix fingerprint (same fix, other repository) | 0 |
| Package published from another repository in validation/holdout (moves, renames, forks) | 0 |
| Near-duplicate code (Jaccard ≥ 0.7) | 0 |
| Held-out excerpt contained in a training excerpt (≥ 80 %) | 0 |
| Repository previously in validation or holdout | 0 |
| Package shared with the **temporal quarantine** of another repository | 10 (by design: the quarantine is not repository-disjoint and is reported separately) |

These are counts after curation: the audit excluded 30 training records for leakage. QuickEval v1 and `coding-v1` contamination checks (exact, near, contained, same repository) found nothing. Not checkable without the GitHub API: fork relations that share no commit, file or package.

## 5. Rights review queue (nothing approved)

`REVIEW_QUEUE.md`, `review/` (approval templates) and `cohort/` (`scripts/training_cohort.py --inspect --extension`: NOTICE files, nested licences and package metadata at every used revision). The 1,417 accepted records that wait for review:

| Group | Repositories | Records | Your decision |
| --- | ---: | ---: | --- |
| Tier A recommended, new (MIT/Apache/BSD/ISC; no open question at any used revision) | 107 | 234 | approve or not, as on 2026-11-15 (`cohort/proposed_approvals.yaml`) |
| Tier B extension recommended, new (formal questions only) | 21 | 108 | as on 2026-11-24 (`cohort/proposed_approvals_extension.yaml`) |
| Tier A not recommended (NOTICE/additional licence, package metadata, or holdout/quarantine overlap) | 71 | 250 | per repository in `COHORT_REVIEW.md` |
| Tier B, other questions | 64 | 281 | per repository |
| Tier C weak copyleft | 19 | 51 | policy decision |
| Tier D strong copyleft (GPL/AGPL) | 74 | 275 | policy decision; not recommended without legal advice |
| Tier D no licence / unknown | 78 | 207 | not trainable without permission |
| Approved repositories whose licence file changed | 6 | 11 | confirm the new licence file hash |

What approval would add (`curate_records` in memory with the real gate and roles, nothing written):

| Approved | Repositories | Training-approved records | Train examples (detection + patch) |
| --- | ---: | ---: | ---: |
| today | 529 | 1,524 | 1,382 + 561 = 1,943 |
| + tier A recommended | 636 | 1,758 | 1,616 + 670 = 2,286 |
| + tier A and B recommended | 657 | 1,866 | 1,724 + 716 = **2,440** |

The validation records (142) do not change: new validation-split records belong to the development set. After your decision: `curate`, then `materialize` on the GPU machine; the training configs stay as they are.

**Independent human review package.**
- `review-packages` drew 80 items: per language, 12 pairs over the CWE families, 4 pairs of the new extraction and 4 abstentions.
- The blind evidence packages are in `artifacts/garden/seed_v4/review_packages/`; the index is `audit/human_review_sample.jsonl`; the template is `audit/human_review.TEMPLATE.yaml`.
- No decisions exist yet. The audit report says that no independent review was performed.

## 6. Why not 5,000 to 10,000

Each lever, with its measured size:

| Lever | Size | Why not in v4 |
| --- | --- | --- |
| Rights review of accepted records | up to 1,417 records | owner decision (above); 533 of them are copyleft, unlicensed or of unknown licence |
| Advisories that reference a **pull request** but no commit | 720 (exactly one PR, all other filters passed) | a new candidate list needs your source-approval extension; a PR's merge cannot be verified without the GitHub API (squash merges change the commit) |
| Per-repository cap of 20 candidates | 1,891 advisories | concentrates data in a few large repositories; new candidate list |
| Fixes outside functions (module constants, regexes, class attributes) | 1,310 | no function-level label without guessing the affected code |
| Multi-file or added-file fixes | 476 + 567 | the input contract shows one file |
| Non-contiguous multi-commit fixes | 695 | backport or partial fix cannot be told apart without full history |
| Synthetic examples | – | not generated: their correctness cannot be established independently here. QuickEval v1 (synthetic) showed recall 1.0 where the realistic development set shows 0.35. More synthetic data would widen that gap rather than close it. |

**Not done:**
- secure code that is not a post-fix twin (unrelated functions are not verified secure, so labelling them `no_finding` would add noise);
- cross-file context;
- new languages.

## Reproduce

```bash
python -m sentinel.garden discover --config configs/garden/seed_v4.yaml                  # copies the frozen v3 list
python -m sentinel.garden build    --config configs/garden/seed_v4.yaml --processes 4    # upstream git; ~1 h on 4 cores from a warm cache
python -m sentinel.garden curate   --config configs/garden/seed_v4.yaml
python -m sentinel.garden audit    --config configs/garden/seed_v4.yaml                  # then curate again, until the exclusions repeat
python -m sentinel.garden verify   --config configs/garden/seed_v4.yaml
python -m sentinel.garden review   --config configs/garden/seed_v4.yaml
python scripts/training_cohort.py  --config configs/garden/seed_v4.yaml --inspect --extension
python -m sentinel.garden review-packages --config configs/garden/seed_v4.yaml
python -m sentinel.garden analyze  --config configs/garden/seed_v4.yaml --output data/garden/seed_v4/analysis/sentinel-garden-v4.json
```

v4 reuses the v3 audit's AI-assisted manual verdicts (the same record IDs) and adds its own: `audit/manual_review.yaml`.
