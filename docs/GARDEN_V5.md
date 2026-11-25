# Sentinel Garden v5

`sentinel-garden-v5` (`configs/garden/seed_v5.yaml`, committed metadata in `data/garden/seed_v5/`) is the dataset for the fresh QLoRA adapter (`docs/TRAINING_V5.md`). Garden v1–v4 are unchanged, and `check` passes for all five versions.

## Summary: the funnel

> **Update 2026-11-25 (owner decision, `docs/TRAINING_AUTHORIZATION.md`):** the 85 recommended repositories are approved and the source approval covers the v5 candidate list. Training data now: **1,741 authorized records, 2,249 train examples (+16 % over v4, +20 % over the first run), 192,708 supervised tokens**. Still pending: 1,049 records of repositories that were not recommended, 22 defended siblings (kind not validated), 9 with a changed license file. The tables below describe the state before the decision.

| Stage | v4 | **v5** | What decides it |
| --- | ---: | ---: | --- |
| Discovered candidates | 7,751 | **8,727** | 7,751 frozen v3 candidates (byte-identical) + 976 new: 377 pull-request fixes, 599 over the old repository cap |
| Fixes built into records | 2,257 | **2,096** | +177 from the new candidates (68 of them pull-request fixes); −338 combined excerpts without evidence for the labelled site (below) |
| Records built / accepted | 5,065 / 4,047 | **4,831 / 3,898** | unchanged quality checks and audit (fixed point) |
| Accepted findings / hard negatives / abstentions | 1,903 / 1,674 / 470 | 1,786 / 1,603 / 509 | 30 hard negatives are defended siblings (new kind) |
| **Authorized (training-approved) records** | 1,524 | **1,741** | after the owner's approvals of 2026-11-25 (85 repositories, source extension); before them 1,467 |
| **Training examples** (train: detection + patch) | 1,943 | **2,249** (1,600 + 649) | + 198 validation examples (validation loss only) |
| **Supervised tokens per epoch** | 167 k | **193 k** (192,708) | of 2.14 M tokens; 0 examples dropped at 4,096 |
| Independent human review | not performed | **not performed** (80 packages ready) | a person outside the implementation |

**The authorized training set did not grow.** Two things explain it:
- v5 is stricter about labels. 338 combined multi-function fixes lack evidence for the vulnerable location and are no longer used.
- Everything v5 adds needs a decision only the owner may take, so it stays pending:
  - 181 accepted records come from the new candidates (`source_approval_not_extended`);
  - 1,147 records come from repositories not yet reviewed;
  - 22 are defended siblings whose kind is not validated yet.

What your decisions would add, computed with the real gate and roles in memory (nothing written):

| Decision | Repositories | Authorized records | Train examples (detection + patch) |
| --- | ---: | ---: | ---: |
| today | 530 | 1,467 | 1,331 + 538 = 1,869 |
| + the 85 recommended repositories (`cohort/OWNER_DECISION.md`) | 590 | 1,661 | 1,525 + 621 = 2,146 |
| + extend the source approval to the v5 candidate list | 605 | 1,741 | 1,600 + 649 = 2,249 |
| + validate the defended-sibling kind after its human review | 605 | 1,752 | 1,611 + 649 = **2,260** |

**5,000–10,000 examples are not reachable from this source without lowering the bar.**
- Of 8,727 candidates, 76 % fail extraction rules that exist to keep labels unambiguous; section 6 lists each with its size.
- Of 3,898 accepted records, 1,354 are pending and 1,207 are evaluation data or quarantine.
- Even with every decision-ready approval, the training set is 2,260 examples.

**Final approval evidence (`cohort/OWNER_DECISION.md`).** It covers the 85 recommended repositories that have no approval entry yet:
- 62 tier A and 23 tier B extension repositories, 257 records;
- each with the licence file bound by sha256 and the evidence from every used revision.

Of v4's recommendations:
- 70 are still in this list;
- 79 have no accepted train/validation record in v5 any more, because their only records were combined excerpts without site evidence;
- 15 are new.

Approving is your step: copy the templates from `proposed_approvals*.yaml` and fill in the decision fields.

## 1. Discovery beyond the frozen list

**Pull-request fixes** (`sentinel/garden/pulls.py`, `osv.merged_commit`):
- 720 GitHub-reviewed advisories reference exactly one pull request and no fix commit.
- The commit search (`gh api search/commits`; no other GitHub API is used) looks for the default-branch commit whose first message line is "Merge pull request #N from …" or ends in "(#N)". Its first-parent diff is the merged change.
- Of 648 searched pull requests:
  - 457 were resolved: 342 squash or rebase, 115 merge commits;
  - 149 had no such commit;
  - 5 were ambiguous;
  - 37 repositories could not be searched (renamed or deleted).
- 377 resolved pull requests became new candidates; the others duplicated an existing fix commit.
- 68 of them pass the unchanged extraction rules. The main rejections: change outside a function 74, more than four source files 48, no source change in the language 43.
- The resolutions are frozen in `pull_request_resolutions.jsonl`, so a rebuild needs upstream git only.

**Repository cap.** 599 advisories of repositories that had reached the v1–v4 cap of 20 were admitted, up to 40 per repository. 1,282 still exceed it. Their records stay diverse: the 10 largest repositories hold 5.8 % of the accepted records.

**Separation from the approved list:**
- New candidates are appended. The 7,751 v3 rows stay byte-identical, and so does their sha256 `535494cb…`, to which the registry's source approvals are bound.
- Records of the new candidates carry `source_approval_not_extended` until you extend the approval. To do so: point `governance.source_approval_covers` at the v5 list, record the decision in `data/registry.yaml`, then `curate`.

**Other sources, measured and not added:**
- Go vulndb and the PyPA advisory database add only 7 and 56 advisories that have no GHSA alias but have a fix commit.
- None of them carries a CWE. A finding label would need a CWE from yet another source, such as NVD.

## 2. Labels: evidence, negatives, cues

**Labelled site of a combined excerpt** (`build.primary_site_evidence`). v4 labelled the function with the most replaced lines. v5 requires evidence:

| Evidence | Accepted fixes |
| --- | ---: |
| the advisory names exactly one of the changed functions (one function, not just one name) | 25 |
| the fix commit message names exactly one | 17 |
| every site makes the same edit, so every site is the weakness | 34 |
| none (rejected: `multi_function_site_unevidenced`) | 458 candidates |

**Defended siblings, a new negative kind** (`build.defended_siblings`, `fix_guards`):
- **Definition.** A function in the fixed file that the fix did not change, and that already called, at the vulnerable revision, the defence the fix added elsewhere (`secure_filename`, `escape`, `Authorize`, …).
- **Defence names.** They must name a defence. Error types, getters and log calls never count; a first, broader rule produced such false guards and was narrowed.
- **Label.** `no_finding`, scoped to the advisory's weakness (`label_scope`). The audit runs the later-fix check on these functions too.
- **Yield.** 36 built and 30 accepted: about one per 60 fixes, so this is not a volume lever.
- **Gate.** The labels rest on a heuristic. The kind stays pending (`negative_kind_not_validated`) until the independent review of its sample (9 items) supports it and you list it in `governance.validated_negative_kinds`.
- **AI spot check.** Of 3 random siblings, 2 are plausible. In the third (an HTML error template), `escape` is applied to one value while `tabTitle` is interpolated unescaped, which shows why the gate is needed.

**Fix comments:**
- Comments the fix added that name the weakness ("prevent path traversal", a GHSA ID) labelled the post-fix excerpt by themselves.
- v5 leaves these lines out of the fixed excerpt and therefore out of the patch (194 accepted fixes).
- The cue is measured below.

**Caller notes.** Contexts also name the same-file callers of the excerpt ("Called in this file by: …", up to 5). Together with v4's callee notes, the model sees the excerpt's call relationships in its file.

## 3. Shortcut probes

`python -m sentinel.garden analyze`. Each probe is fitted on the train split and scored on the repository-disjoint validation split.

| Probe | v3 | v4 | v5 |
| --- | --- | --- | --- |
| Abstention-only context sentence → abstain | 100 % / 100 % | 0 records | 0 records |
| Security comment in the excerpt: no_finding vs finding | 18.6 % vs 7.8 % | 18.5 % vs 7.1 % | **6.7 % vs 6.9 %** |
| Surface features only (length, blocks, indentation, notes, blank lines, context length), finding vs no_finding | 63.9 % (majority 55.7 %) | 62.0 % (55.6 %) | **57.2 % (54.9 %)** |
| The same, all three decisions | 69.0 % (47.8 %) | 56.9 % (48.7 %) | **49.7 % (47.7 %)** |
| Excerpt length threshold | 51.7 % | 55.6 % | 53.4 % |
| Multi-block excerpt → no_finding | 100 % (15) | 60 % (62) | 86 % (22) |
| Code-token naive Bayes: vulnerable ranked above its own fix | 78 % | 79 % | 76 % |

- Remaining cue 1: excerpts with several blocks are mostly post-fix excerpts with an added helper, because fewer combined vulnerable excerpts survive the evidence rule.
- Remaining cue 2: the lexical difference within a pair is what a fix is. Only negatives that are not twins reduce it, and the evidence-backed ones are few (§ 2).

## 4. Evaluation sets

Roles as in v4 (`docs/EVALUATION.md`): the v3 development set is retired, and no repository moves.

| Role | Records | Repositories | Languages (Go/JS/Py/TS) | Decisions (finding/no_finding/abstain) |
| --- | ---: | ---: | --- | --- |
| `development` (selection) | 163 | 47 | 40 / 32 / 55 / 36 | 78 / 62 / 23 |
| `final_test` (never exported before v4, never evaluated) | 9 | 4 | 0 / 6 / 3 / 0 | 4 / 4 / 1 |
| `previous_holdout` (v3 holdout repositories, rebuilt with v5) | 453 | 109 | 128 / 87 / 147 / 91 | 209 / 186 / 58 |
| `temporal_quarantine` (reported separately) | 441 | – | – | 197 / 180 / 64 |

**Final test.**
- The untouched `final_test` shrank from 45 to 9 cases. Most of v4's came from combined excerpts that v5 no longer accepts. 9 cases cannot support a final conclusion.
- The strong final test is `previous_holdout`: 453 cases in all four languages and 12 CWE families, with abstentions and patch cases. **It is untouched only if the Garden v3 holdout was never evaluated** (GPU_RUNBOOK § 12, last step). I have no information that it was. Please confirm; until then both sets are reported separately.

**Development set.** It covers every family except SQL injection: 0 SQL injection findings among its 78.

**Patch cases.** Both sets carry patch cases (`dev.patch_cases.jsonl`, `holdout_reserved.patch_cases.jsonl`).

## 5. Quality and contamination

**Audit and reproducibility:**
- The audit reached a fixed point: rounds 2 and 3 were byte-identical.
- 690 automated exclusions:
  - later fix of the same function;
  - broad spans;
  - reported incomplete fixes;
  - unverifiable later fixes;
  - leakage: 22 training records (690 exclusions in total).
- The v4 AI-assisted verdicts were carried over. The 5 groups v5 no longer builds were dropped (noted in the file).
- `verify` rebuilt all 4,831 records from the pinned revisions with 0 differences.

**Contamination, all versions** (`analysis/sentinel-garden-v5.json` → `contamination`). The 2,691 accepted training-role records were compared with 3,764 evaluation records of v1–v5:
- 0 shared advisory or CVE;
- 0 shared fix or vulnerable revision;
- 0 shared normalized code;
- 0 shared fix fingerprint;
- 0 shared packages with validation or holdout;
- 0 near-duplicates or contained excerpts;
- 0 repositories that were ever in validation or holdout.
- 9 share a package with the temporal quarantine of another repository; this is by design and is reported separately.

**Independent human review package:**
- `review-packages` drew 80 items (index `audit/human_review_sample.jsonl`; blind packages in the Git-ignored work directory):
  - per language, 12 pairs over the CWE families;
  - up to 4 pairs of the new extraction kinds;
  - 4 abstentions;
  - and, new, up to 6 defended siblings (`negative_safe`; 9 in total).
- No decisions have been recorded. That review is the precondition for validating the defended-sibling kind.

## 6. What remains out of reach, measured

| Rejection (v5 build) | Candidates | Why it stays rejected |
| --- | ---: | --- |
| Change outside a function (module constants, regexes, class attributes) | 1,464 | no function-level excerpt shows the weakness |
| Several functions without evidence for the vulnerable one | 458 + 887 over the limits | unevidenced localization (your rule) |
| Non-contiguous multi-commit fixes | 847 | backport and partial fix cannot be told apart without full history |
| Added/deleted files, more than 4 files | 631 + 546 | the input shows one file. Cross-file excerpts would change the evaluation contract, so they are not built. |
| No source change in the language | 653 | the fix is elsewhere (C extension, configuration, dependency) |

**Synthetic or hand-made examples: investigated, not added.**
- Correctness can be established independently only where an executable oracle exists. Examples: SQL injection against an in-memory SQLite database; path traversal in a temporary directory; command injection with a marker process. The coding sandbox (`docs/CODING_SANDBOX.md`) could run such checks.
- That covers a few families in Python and JavaScript. It would add a separately labelled supplement of perhaps hundreds of templated functions.
- The earlier evidence argues against making it training data now:
  - on QuickEval v1 (synthetic), the first adapter reached recall 1.0;
  - on realistic code (v3 development set), it reached 0.35.
- Recommendation: if at all, train such a supplement only after the fresh v5 baseline, as a separate, measured ablation.

## Reproduce

```bash
python -m sentinel.garden discover --config configs/garden/seed_v5.yaml                  # only to regenerate: candidates.jsonl is committed (needs the OSV snapshot cache)
python -m sentinel.garden build    --config configs/garden/seed_v5.yaml --processes 4
python -m sentinel.garden curate   --config configs/garden/seed_v5.yaml
python -m sentinel.garden audit    --config configs/garden/seed_v5.yaml                  # then curate again, until the exclusions repeat
python -m sentinel.garden verify   --config configs/garden/seed_v5.yaml
python -m sentinel.garden review   --config configs/garden/seed_v5.yaml
python scripts/training_cohort.py  --config configs/garden/seed_v5.yaml --inspect --extension   # cohort/OWNER_DECISION.md
python -m sentinel.garden review-packages --config configs/garden/seed_v5.yaml
python -m sentinel.garden analyze  --config configs/garden/seed_v5.yaml --output data/garden/seed_v5/analysis/sentinel-garden-v5.json
```
