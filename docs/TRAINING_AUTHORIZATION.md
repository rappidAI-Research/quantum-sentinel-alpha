# Training authorization - Sentinel Garden (decision record)

| | |
| --- | --- |
| Decided | 2026-11-15 |
| Decided by | Jonas Cikemgil, project owner (reviewer recorded in `data/registry.yaml` and `data/garden/approvals.yaml`) |
| Basis | `docs/TRAINING_COHORT_RIGHTS_REVIEW.md` and the evidence in `data/garden/seed_v2/cohort/` (manifest sha256 `e8250674…` before the decision) |
| Prepared by | the AI engineering assistant, which collected the evidence and wrote the entries on the owner's instruction. The assistant is not a reviewer and is not named as one. |

This records an owner decision. It is not a legal opinion; the open legal questions below remain open.

## What was decided

1. **Sources** (`data/registry.yaml`): `github-advisory-db`, `osv` and `ghsa-osv-fix-commits` are
   `production_approved` for training.
   - Revision: the v2 OSV snapshot (advisories published before 2026-11-01) and the v2 candidate list
     (sha256 `9773f76d…`).
   - `license_status: reviewed`, `terms_status: approved_for_project`.
   - `redistribution_status: not_redistributed`: neither code excerpts nor advisory text are republished
     by this project.
2. **Repositories** (`data/garden/approvals.yaml`): the **484** repositories that the training cohort
   recommended for review are `approved_for_training`.
   - Licenses: MIT 318, Apache-2.0 120, BSD-3-Clause 30, BSD-2-Clause 9, ISC 7.
   - Each approval is bound to the sha256 of the license file the review recorded. If the upstream license
     file differs, the approval no longer applies.
3. **Not approved:**
   - the 42 clarification cases (NOTICE/PATENTS/third-party files, including NOTICE texts detected as
     AGPL/GPL in imgproxy, streamlit and openfga; 6 package-metadata conflicts);
   - every tier B, C and D repository;
   - the 24 tier A repositories that also have holdout/quarantine records.

   They stay default-deny and can be decided later.

## Result (verified)

`python -m sentinel.garden curate --config configs/garden/seed_v2.yaml`:
- **1,297 training-approved records**: train 1,177 (538 findings, 494 verified post-fix hard negatives,
  145 abstentions) and validation 120;
- 479 / 49 patch examples;
- 484 repositories.

The manifest changed only in its governance fields: content, supervision hashes, quality status and splits are unchanged. Holdout and temporal-quarantine records stay excluded by the gate. Garden v1 is not re-curated and keeps 0 approved records as the immutable reference.

`python -m sentinel.train --config configs/train/qlora_qwen35_9b.yaml --dry-run` passes: "train 1177 /
validation 120 approved examples from 484 repositories", every supervision hash and checksum checked, no model loaded. With `tasks: [detection, patch]` the loader also accepts 479 / 49 patch examples.

## Obligations that follow from the decision

- **Attribution:**
  - name the GitHub Advisory Database (CC-BY-4.0, link to https://github.com/advisories) in the model
    card and dataset documentation, and say that advisory summaries are used verbatim as evidence text;
  - list the upstream repositories and commits (`data/garden/seed_v2/EXTERNAL_RESOURCES.json`).
- **Code is not redistributed:**
  - the manifest stays metadata-only;
  - exports stay in the Git-ignored work directory and are never published.
- **Revocation:** removing an entry from `data/garden/approvals.yaml` (or setting a source back to
  `production_approved: false`) takes effect at the next `curate` and at training start.

## Still open (unchanged)

1. Whether published weights are a redistribution of MIT/Apache code or of CC-BY text.
2. GitHub Acceptable Use Policies §7 (research use "only if any publications ... are open access"). Seek
   advice before any non-open or commercial release.
3. OSV service terms beyond the per-source data licenses.
4. Independent human review of the 64 evidence packages (`data/garden/seed_v2/audit/human_review_sample.jsonl`).
   This is recommended before results are reported; it is not a precondition of this decision.

## Decision of 2026-11-24 (Garden v3)

Decided by Jonas Cikemgil (project owner), on the evidence in `docs/GARDEN_V3.md` and
`data/garden/seed_v3/cohort/COHORT_REVIEW.md`; entries prepared by the AI engineering assistant, which is
not a reviewer.

1. **Sources** (`data/registry.yaml`): the approval of the three sources is extended to the Garden v3
   candidate list (same OSV snapshot; candidates sha256 `535494cb…`). Revision and notes are updated, every
   other field is unchanged.
2. **Repositories** (`data/garden/approvals.yaml`): **51 tier B repositories** of the v3 extension,
   inspected exactly like tier A (no NOTICE, additional license file or package-metadata conflict at
   any used revision):
   - **28 `no_registry_crosscheck`:** deps.dev has no usable license for the package; the decision
     rests on the root license file;
   - **23 `registry_lists_bundled_licenses`:** deps.dev lists the root license plus licenses of
     bundled code.
3. **Not approved** (owner's choice, default-deny):
   - the newly recommended tier A repositories (29 at the decision; 26 after the rebuild that followed the code review);
   - the 16 `license_text_updated` repositories, including the one repository that is in both
     `license_text_updated` and `registry_lists_bundled_licenses`;
   - the 2 `other_permissive_Unlicense` repositories;
   - all clarification cases.

   Each can be decided later with the prepared templates (`data/garden/seed_v3/cohort/`).
4. **Training tasks:** detection and patch generation (`data.tasks: [detection, patch]`).

Result (verified):
- `curate` gives **1,464 training-approved records** from 533 repositories: train 1,331 and
  validation 133, plus 540 / 54 patch examples, i.e. **1,871 / 187 training examples**
  (after the rebuild that followed the code review of the multi-commit rule).
- Two previously approved repositories now also have temporal-quarantine records. Their 6 records are
  excluded by the loader rule; the approvals themselves are unchanged.
- `python -m sentinel.train --dry-run` passes.
- Every approved repository's records were re-inspected at the new v3 revisions; all remain clean.

## Decision of 2026-11-25 (Garden v5)

Decided by Jonas Cikemgil (project owner), who gave all approvals the Garden v5 report asked for (`docs/GARDEN_V5.md`, `data/garden/seed_v5/cohort/OWNER_DECISION.md`). The AI engineering assistant prepared the entries; it is not a reviewer.

1. **Sources** (`data/registry.yaml`): the approval of the three sources is extended from the Garden v3 candidate list to the **Garden v5 candidate list** (candidates sha256 `af72c35d…`). That list is the v3 list (byte-identical) plus 377 pull-request fixes resolved to their merge commit and 599 advisories over the former repository cap. `configs/garden/seed_v5.yaml` → `governance.source_approval_covers` names this list.
2. **Repositories** (`data/garden/approvals.yaml`): **85 repositories**, inspected at every used revision like the earlier cohorts:
   - 62 tier A `recommended` (MIT, Apache-2.0, BSD, ISC);
   - 23 tier B `extension` (formal questions only, including 3 Unlicense).

   Each approval is bound to the sha256 of the recorded license file.
3. **Not covered by this decision:**
   - **The defended-sibling negative kind** (`governance.validated_negative_kinds` stays empty). It needs an independent human review of its sample: a permission cannot establish that a label is correct. That review is the owner's own quality requirement for these examples.
   - **The independent human review** of the 80 packages, which can only be done by a person.
   - Repositories that were not recommended:
     - clarification cases;
     - tier B with substantive questions;
     - copyleft, no license or unknown license;
     - license files that differ from the approved one at some revision (9 records).

Result (verified):
- `curate` gives **1,741 training-approved records** from 605 repositories.
- That is **2,249 training examples** (1,600 detection + 649 patch) and 198 validation examples.
- 192,708 supervised tokens per epoch; 0 examples dropped at 4,096 tokens.
- `python -m sentinel.train --config configs/train/qlora_ornith15_9b_v5.yaml --dry-run --check-tokens` passes.
- `materialize` rebuilds exactly these records.
- Holdout, temporal-quarantine and development records stay excluded by the gate and the evaluation roles.

## Decision of 2026-11-25, later (Garden v6)

Decided by Jonas Cikemgil (project owner): "Ich erteile meine Freigabe jetzt schon für alles" for the Garden v6 decision package (`docs/GARDEN_V6.md`, `data/garden/seed_v6/cohort/`). The AI engineering assistant prepared the entries; it is not a reviewer.

1. **Sources** (`data/registry.yaml`), now `production_approved`:
   - `morefixes` (Zenodo record 20776007, CC-BY-4.0; metadata only, attribution in `docs/GARDEN_V6.md`);
   - `nvd-cve` (the CVE records contained in that dump);
   - `morefixes-cve-fix-commits` (fix commits named by NVD references, a GHSA record or the commit's own message; every record keeps its grade);
   - `codeql-query-tests` (github/codeql@4136c26, MIT; the CodeQL CLI is never run or redistributed).

   The GHSA/OSV approvals stay bound to the Garden v5 candidate list; the eval-only and research-reference entries are unchanged.
2. **Repositories** (`data/garden/approvals.yaml`): **93 repositories**, inspected at every used revision:
   - 14 tier A `recommended`, 79 tier B `extension` (formal questions only), among them `github/codeql` (MIT);
   - each approval is bound to the sha256 of the recorded license file.
   - Two corrections made before the inspection (`docs/GARDEN_V6.md` § 6): the license detector no longer reads the PSF license as GPL-2.0 (its history table names the GPL), and in the v6 cohort an additional license file whose own license is permissive (an Apache NOTICE, CPython's PSF license for derived CodeQL test code) is a confirmed fact with redistribution-only obligations, not an open question.
3. **Not covered, and not coverable by a permission:**
   - **The two heuristic negative kinds** (defended siblings, CodeQL tool-test negatives; 151 accepted records, `governance.validated_negative_kinds` stays empty). Whether their labels are right is a quality question that the independent human review answers, as the owner's brief requires.
   - **The independent human review** of the 147 packages: only a person outside the implementation can do it.
   - **Copyleft repositories** (214; GPL/LGPL/AGPL/MPL/EPL): the gate requires a written basis per repository (`non_permissive_justification`), which a blanket approval does not state. With it they would add about 700 train examples (`approval_scenarios.json`, scenario 5).
   - **Repositories without a license or with an unknown or conflicting one:** no permission exists that could be granted.
   - **97 clarification cases** (350 records) whose question is which license applies at all (another license in the package metadata or registry, a root license outside the tier A list such as CPython's PSF-2.0, license files that differ between revisions), and 21 records whose license file differs from the approved one.

Result (verified):
- `curate` gives **2,170 training-approved records** from 666 repositories (Garden v5: 1,741 from 605).
- That is **2,758 training examples** (2,024 detection + 734 patch; +23 % over Garden v5) and 204 validation examples.
- 233,463 supervised tokens per epoch (of 2.56 M); 0 examples dropped at 4,096 tokens.
- `python -m sentinel.train --config configs/train/qlora_ornith15_9b_v6.yaml --dry-run --check-tokens` passes.
- Holdout, temporal-quarantine, development, external-benchmark and tool-test-negative records stay excluded or pending.
