# Training authorization - Sentinel Garden v2 (decision record)

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
