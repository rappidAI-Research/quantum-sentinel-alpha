# Sentinel Garden v6

`sentinel-garden-v6` (`configs/garden/seed_v6.yaml`, committed metadata in `data/garden/seed_v6/`) is the first Garden version with label sources beyond GHSA/OSV. Garden v1–v5 are unchanged and reproducible; `check` passes for all six versions.

## Summary

| Stage | v5 | **v6** | What decides it |
| --- | ---: | ---: | --- |
| Candidates | 8,727 | **10,299** | the v5 list, byte-identical, plus 1,572 MoreFixes CVEs (1,255 named by NVD references, 153 by a GHSA record, 164 by the fix commit's message); 4,182 further CVEs only confirm v5 candidates |
| Tool-test units | – | **185** | CodeQL inline-expectation security tests (Python, JS/TS, Go) |
| Records built / accepted | 4,831 / 3,898 | **6,224 / 5,055** | same extraction, quality checks and audit (fixed point after three rounds) |
| Accepted findings / negatives / abstentions | 1,786 / 1,603 / 509 | **2,387 / 2,080 / 588** | includes 199 tool-test findings and 116 tool-test negatives |
| Distinct CWEs among accepted findings | 204 | **222** | |
| **Authorized (training-approved) records** | 1,741 | **1,676** | 50 v5 records belong to the OpenSSF CVE Benchmark (now reserved), 15 were refuted by the stricter later-fix check |
| **Training examples** (train: detection + patch) | 2,249 | **2,153** (1,538 + 615) | + 193 validation examples |
| **Supervised tokens per epoch** | 192,708 | **184,544** | of 2.06 M tokens; 0 dropped at 4,096 |
| Development / untouched final test | 163 / 9 cases | **239 / 91** | new MoreFixes repositories in the validation and holdout buckets |
| Independent human review | not performed | **not performed** (147 packages ready) | a person outside the implementation |

**The authorized training set did not grow; the dataset did.** Everything the new sources add needs a decision only the owner may take, and v6 is stricter about what was authorized before.

| Owner decision (cumulative; computed with the real gate in memory) | Train examples (detection + patch) | Supervised tokens |
| --- | ---: | ---: |
| today | 1,538 + 615 = **2,153** | 184,544 |
| + approve the MoreFixes, NVD and MoreFixes fix-commit sources | 1,565 + 626 = 2,191 | |
| + the 73 recommended repositories without an approval yet (`cohort/OWNER_DECISION.md`) | 1,756 + 707 = 2,463 | |
| + the CodeQL tool-test source and `github/codeql` (MIT) | 1,955 + 707 = 2,662 | |
| + validate the two heuristic negative kinds after their human review | 2,088 + 707 = **2,795** | 230,207 |
| + the 215 copyleft repositories (single GPL/LGPL/AGPL/MPL/EPL root license), each with a written justification; never unlicensed ones | 2,591 + 919 = **3,510** | 294,613 |

`data/garden/seed_v6/cohort/approval_scenarios.json` has every scenario per kind, source and language.

## 1. Sources: what was checked, what was integrated

Feasibility first (language coverage, unique candidates, overlap with Garden v5, provenance, license, expected yield), then only the sources with real incremental value were implemented.

| Source | Checked | Decision | Why |
| --- | --- | --- | --- |
| **MoreFixes** (Zenodo 20776007, dump of 2026-06-20, CC-BY-4.0) | 3.47 GB dump streamed, metadata tables only; 78,654 CVEs with candidate fixes; 9,238 fix commits change Python/JS/TS/Go | **integrated as discovery index** (`sentinel/garden/morefixes.py`) | the only large source of fix commits outside GHSA/OSV-reviewed packages (applications, unpublished packages, CNA/huntr CVEs) |
| **CodeQL security query tests** (github/codeql@4136c26, MIT) | 185 inline-expectation test units in Python, JavaScript/TypeScript, Go | **integrated as rule-scoped tool tests** (`sentinel/garden/querytests.py`) | the only checked source with explicitly expected positives *and* expected safe code per rule, verified by CodeQL's own test runner |
| JsVul (Zenodo 18195839, CC-BY-4.0) | 21,901 functions, 2,039 vulnerable, 1,935 commits; no CWE in the function export | not integrated | its `sha` is the vulnerable revision (407 equal Garden v5 vulnerable revisions); it aggregates NVD, OSV, CVEfixes, CrossVul, OSSF and SecBench.js, which GHSA/OSV and MoreFixes already cover; labels are "changed in the fix commit" heuristics |
| PyVul (GitHub, no license file) | 1,916 commits with CWE | not integrated | no license; LLM-assisted label cleansing; 1,025 commits are already v6 candidates and 1,469 are in MoreFixes; the 346 others rest on huntr/Snyk pages only |
| CVEfixes v1.0.8 (Zenodo, 12.7 GB) | metadata only | not downloaded | MoreFixes uses the same NVD-reference method on CVEs up to 2026-06 (CVEfixes: 2024-07) |
| Bandit / gosec test samples | 93 Bandit example files, gosec rule samples | not integrated | expectations of API-pattern scanners (a flagged call, a count of issues), weaker than CodeQL's data-flow expectations and without explicit safe variants; CodeQL covers the same rule families |
| SecLLM-Dataset (HF, CC-BY-4.0) | 1,593 records in the four languages | not integrated | every record is one finding of a single SAST tool, without independent confirmation and without fix |
| Go vulndb, PyPA advisory DB | measured for v5 | not integrated | 7 and 56 unique fix-commit advisories, none with a CWE |
| NVD | via MoreFixes' `cve` table | used through MoreFixes | the CVE record (description, CWE, references) is the evidence of a MoreFixes candidate |
| OpenSSF CVE Benchmark (MIT) | 223 JS/TS CVEs with annotated weakness locations | **evaluation only**: its vulnerabilities are excluded from training | `data/garden/benchmarks/ossf_cve_benchmark.jsonl`, role `external_benchmark` |
| OWASP Benchmark Python (GPL-3.0) | 1,230 synthetic test cases with expected results | evaluation only, not yet wired | GPL-3.0: never training data |
| Cisco VLoc Bench | paper (arXiv 2609.15939): 500 vulnerabilities, repository-level file localization | not integrated | repository-level (file F1); its data was not found in a public location from this environment. Re-check before the final test and add it to `external_benchmarks` |

### Funnel per source

| | GHSA/OSV (v5 list) | MoreFixes: NVD reference | MoreFixes: GHSA reference | MoreFixes: commit message | CodeQL tool tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidates / test units | 8,727 | 1,255 | 153 | 164 | 185 |
| … that yielded records | 2,096 | 366 | 57 | 46 | 93 |
| Records | 4,831 | 823 | 128 | 105 | 337 (714 functions before the template-family cap) |
| Accepted | 3,863 | 686 | 101 | 90 | 315 |
| Accepted findings (= unique vulnerabilities; rules for tool tests) | 1,785 | 314 | 48 | 41 | 199 (71 rules) |
| Training-approved | 1,676 | 0 | 0 | 0 | 0 |
| Pending owner decision (accepted) | 1,043 | 536 | 82 | 71 | 315 |

MoreFixes in detail: the dump lists 78,654 CVEs with candidate fixes; 70,645 have no fix in a target-language runtime file, 4,182 confirm a v5 candidate (written to `morefixes_corroborations.jsonl`, never a second record), 1,501 have only a heuristic fix commit, 754 fail selection (rejected CWE 213, repository cap 158, no specific CWE 149, too many CWEs 90, CVE or commit metadata missing 77, more than 3 fix commits 56, several repositories 11). Of the 1,572 candidates, 30 % yield records (v5 candidates: 24 %); the others fail the same unambiguity rules (change outside a function 249, several changed functions 221, a file added or deleted 118, combined excerpt without evidence 91, too many files 89, …), never a relaxed rule.

## 2. Label evidence

Every record states where its label comes from (`provenance.fix_evidence`, `provenance.source_kind`); `analysis.json` reports all statistics per source.

| Grade | Records | Evidence |
| --- | --- | --- |
| `osv_reference` (field absent: v1–v5 records) | GHSA/OSV | a GitHub-reviewed advisory names the fix commit |
| `nvd_reference` | MoreFixes | the CVE's own NVD references name the fix commit (checked in `discover` from the CVE record) |
| `ghsa_reference` | MoreFixes | MoreFixes records that a GitHub Advisory Database record names the commit (`GHSD_DIRECT_COMMIT`); relayed, not re-checked |
| tool test (`source_kind: tool_test`) | CodeQL | the rule's test expectation, checked by CodeQL's test runner |
| heuristic | — | **not accepted**: 1,701 CVEs whose fix commit only MoreFixes' mining (Prospector score) links to them |

A MoreFixes candidate then goes through exactly the extraction, quality checks and audit of a GHSA candidate: the excerpt is rebuilt from the pinned upstream commit, the vulnerable function must be the one the fix changes (one changed function, or the v5 evidence rule for combined excerpts), the patch must reproduce the fixed excerpt exactly, the audit checks that the advisory text corresponds to the code. The evidence text of a finding is the CVE description's first sentence that describes the weakness (CNA boilerplate and product blurbs are skipped).

**CodeQL tool tests.** A function is a `query_test_finding` only if all its `$ Alert` lines are results the rule really reports and the test author expects (no `SPURIOUS` or `MISSING` marker in it), and for a data-flow rule every alert's source lies in the same function. A `query_test_negative` is a function of the same test files with no expected alert and no `MISSING` marker that the test marks safe: a `$ SPURIOUS: Alert` (the rule reports it, the author says it is a false positive), a `$ Source` whose value reaches no reported sink, or an explicit OK/GOOD/safe/sanitized note. Removed from the input: every comment and docstring (all markers), functions and identifiers that name their label (`bad`, `good`, `unsafe`, `taint`, …), and the test file name (a neutral name per language; the real path stays in the provenance). Tool tests are training data only; `github/codeql` is a train repository under the frozen split policy, and they are exempt from the temporal cutoff (their date is the test suite's revision, not a fix date).

## 3. Negative labels: the scope mismatch

The review contract (`quickeval-review-v2`) defines `no_finding` as "the excerpt is not vulnerable". A post-fix twin only shows that *one* weakness was fixed. v6 keeps the contract (one detection task, comparable with every earlier run) and resolves the mismatch on the data side:

1. **Every negative carries its scope** (`target.label_scope`), and the evidence text states only what is known ("the upstream fix for the CWE-79 weakness is applied").
2. **A negative refuted by any known later fix is excluded.** v5 checked later fixes only among the OSV advisories of the record's own packages. v6 (`audit.cross_source_later_fixes`) checks every OSV advisory of every package of the repository *and* every CVE MoreFixes links to a fix commit of the repository, whatever its evidence grade.
   - **Measured:** the wider check refutes **160 post-fix negatives of the v5 list that v5 accepted** (110 by a CVE from MoreFixes, 35 by an advisory of another package of the repository, 15 by Go/PyPA advisories) and leaves 24 unverifiable (the later commit cannot be read): about 10 % of v5's 1,573 accepted post-fix twins were later changed again by a security fix. They are excluded in v6, together with 45 v6 post-fix negatives and 3 defended siblings.
3. **Negative kinds whose label rests on a heuristic stay pending** until their independent review sample validates them: defended siblings (v5) and CodeQL tool-test negatives (v6, rule-scoped).
4. **What remains**: a post-fix negative that passes these checks is still only verified for its own weakness. The residual rate is unknown; the human-review question `hard_negative_safe` measures it on the sample.

## 4. Quality, deduplication and contamination

- **Cross-source duplicates and aliases:** a MoreFixes CVE that is an alias of a v5 advisory (3,817) or whose fix commit a v5 candidate uses (365) is a corroboration, not a candidate. Within the dataset the existing exact, near-duplicate (Jaccard 0.85) and label-conflict (0.92) checks run over all sources together; tool-test variants of one rule in one file are capped to one finding (the file is the template family; 584 → 207 findings before deduplication).
- **Forks, shared revisions, copied fixes, contained code:** the v4 leakage checks (shared vulnerable revision, patch fingerprint, containment 0.8) are unchanged and run across sources; the thresholds are not weakened.
- **Licenses:** a record keeps the root license files of its revision; an approval is bound to the file's sha256. 18 accepted records sit in repositories whose license file changed after the approval (`repository_approval_license_mismatch`).
- **Contamination:** 0 shared identities, fix fingerprints, contained code or near-duplicates between the 3,585 training-role records and the 3,928 evaluation records of all Garden versions (`analysis/sentinel-garden-v6.json`, `contamination`); 9 records share a package with temporal-quarantine records, as in v5.
- **Shortcut probes** (`analysis/sentinel-garden-v6.json`):
  - surface features predict finding vs no_finding on the validation split 0.8 points above the majority rate (v5: 2.3); code-token and context-token probes stay near chance on single records;
  - the security-comment cue stays gone (6.3 % findings, 6.7 % no_finding);
  - **source identity:** surface features recognise a tool test 94.9 % of the time; without the template-family cap tool tests were 83.5 % findings (real-world 44–47 %), with it 63.2 %. This is the main residual risk of the tool-test source, and why it is a separate owner decision;
  - multi-block excerpts are still 87 % negatives on the validation split (23 records).

## 5. Evaluation

| Role | v5 | **v6** | Notes |
| --- | ---: | ---: | --- |
| development (selection) | 163 | **239** (63 repositories; finding 113, no_finding 99, abstain 27; all four languages) | +87 from MoreFixes repositories never evaluated; the GHSA part shrinks from 163 to 152 under the stricter later-fix check |
| `final_test` (untouched) | 9 | **91** (30 repositories; finding 42, no_finding 40, abstain 9) | 82 from MoreFixes; mostly Python and JavaScript |
| `previous_holdout` | 453 | **452** (107 repositories) | the same reservation; independent only if the v3 holdout was never evaluated (not confirmed; the repository holds no record of a holdout run) |
| `external_benchmark` | – | **99** (47 repositories) | OpenSSF CVE Benchmark vulnerabilities found in Garden; never trained on |
| `temporal_quarantine` | 441 | 440 | |

- **Selection rule v2** (fixed before any v6 result): recall ≥ 0.30 and false-positive rate ≤ 0.20 before decision accuracy decides (`docs/EVALUATION.md`).
- **External benchmarks stay separate.** OpenSSF CVE Benchmark (function-level cases can be derived from its annotated locations), OWASP Benchmark Python (GPL-3.0, synthetic, evaluation only) and VLoc Bench (repository-level file F1) are never mixed with Garden's function-level scores. Only the OpenSSF exclusion is wired; the two others are documented, not integrated.
- QuickEval v1 stays a regression check only.

## 6. Governance: what is pending, what your decisions would add

Nothing was approved. New sources and their repositories stay pending (`source_not_approved:<id>`, `repository_not_reviewed`), and so do the two heuristic negative kinds (`negative_kind_not_validated`).

**Decision package** (`data/garden/seed_v6/cohort/`):
- `OWNER_DECISION.md`: the 73 recommended repositories without an approval, each with license, license file and sha256, records and open questions; `proposed_approvals*.yaml`: templates that fail validation until you fill them in;
- `approval_scenarios.json`: the yield of each cumulative decision (table above);
- source decisions: `data/registry.yaml` entries `morefixes` (CC-BY-4.0, attribution required), `nvd-cve`, `morefixes-cve-fix-commits` and `codeql-query-tests` (MIT; the CodeQL CLI with its separate terms is never run or redistributed).

**Open uncertainties:**
- `ghsa_reference` and `commit_message_reference` are weaker grades than `nvd_reference`; they can be approved or left out separately (every record carries its grade).
- 381 of 687 MoreFixes train-split records are in copyleft or unlicensed repositories: the source brings many applications. Repositories without a license can never be approved; copyleft ones only with a written basis.
- Tool tests: the source-identity cue (above) and rule-scoped negatives.
- The independent human review (147 packages: 88 pairs including 24 from MoreFixes, 16 abstentions, 13 defended siblings, 16 tool-test findings, 14 tool-test negatives) has not been performed.

## 7. Why not 5,000–10,000 examples, measured

| Blocker | Size | Why it is not removed |
| --- | --- | --- |
| Extraction rules for unambiguous labels | 76 % of GHSA and 70 % of MoreFixes candidates yield no record | each rule rejects a case where the vulnerable location or the fix cannot be stated without guessing |
| Evaluation roles, holdout and quarantine (frozen split) | 1,470 accepted records | never training data by design |
| Licenses | 381 MoreFixes and 1,026 GHSA train/validation records in unreviewed repositories, most copyleft or unlicensed | public availability is not permission; copyleft needs a written basis, no license means no permission |
| Owner decisions | 2,047 accepted records pending | sources, repositories, negative kinds |
| Sources that would add volume without evidence | heuristic-only MoreFixes fixes (1,501 CVEs), SAST-only datasets, unlabelled function corpora | their labels are not supported by the evidence the brief requires |

With every decision-ready approval the training set is **2,795 examples (+24 % over v5)**; with copyleft repositories as well, **3,510 (+56 %)**. 5,000–10,000 examples at this evidence bar would need fix data of many more vulnerabilities in the four languages than GHSA/OSV and NVD hold today, or a lower bar.

## Reproduce

```bash
# MoreFixes dump (3.47 GB, md5 bdff6120f1e65626c77d42656162316e) from https://zenodo.org/records/20776007
mkdir -p artifacts/sources/morefixes && curl -L -o artifacts/sources/morefixes/dump-2026-06-20.sql.gz "https://zenodo.org/records/20776007/files/dump-2026-06-20.sql.gz?download=1"
python -m sentinel.garden morefixes-index --config configs/garden/seed_v6.yaml   # md5 check, metadata index (offline)
python -m sentinel.garden discover --config configs/garden/seed_v6.yaml          # frozen v5 list + MoreFixes candidates (offline)
python -m sentinel.garden run --config configs/garden/seed_v6.yaml --processes 4 # build (upstream git, CodeQL checkout) + curate
python -m sentinel.garden audit --config configs/garden/seed_v6.yaml             # repeat with curate until the exclusions are a fixed point
python -m sentinel.garden review --config configs/garden/seed_v6.yaml
python -m sentinel.garden review-packages --config configs/garden/seed_v6.yaml
python scripts/training_cohort.py --config configs/garden/seed_v6.yaml --inspect --extension
python scripts/approval_scenarios.py --config configs/garden/seed_v6.yaml
python -m sentinel.garden analyze --config configs/garden/seed_v6.yaml --output data/garden/seed_v6/analysis/analysis.json
python -m sentinel.garden verify --config configs/garden/seed_v6.yaml
python -m sentinel.garden check --config configs/garden/seed_v6.yaml
```

Attribution: MoreFixes (J. Akhondali et al., "MoreFixes: A Large-Scale Dataset of CVE Fix Commits Mined through Enhanced Repository Discovery", PROMISE 2024; dataset CC-BY-4.0, Zenodo record 20776007). CodeQL query tests © GitHub, Inc., MIT. No MoreFixes or CodeQL file, and no upstream code, is committed to this repository.
