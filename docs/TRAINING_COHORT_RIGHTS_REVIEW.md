# Training cohort rights review (Garden v2) - decisions for the owner

**This is an engineering evidence review, not a legal opinion, and it approves nothing.** The active
approvals file `data/garden/approvals.yaml` is empty and every registry source has
`production_approved: false`, so **0 records are training-approved**. The proposed approvals are
templates that fail schema validation until an authorized reviewer completes them.

Generated files (`python scripts/training_cohort.py --config configs/garden/seed_v2.yaml --inspect`):

- `data/garden/seed_v2/cohort/COHORT_REVIEW.md`: decision groups, the volume, every repository with
  its license link and hash, and every open question;
- `data/garden/seed_v2/cohort/training_cohort.json`: the machine-readable evidence manifest. It holds
  the inspected revisions and paths, file hashes of root, NOTICE and nested license files, package
  metadata licenses, open questions and the proposed decision. The evidence is bound to the exact
  records, revisions and paths and is invalidated when they change;
- `data/garden/seed_v2/cohort/proposed_approvals.yaml`: templates for the recommended repositories.

The pilot (`docs/PILOT_DATASET_RIGHTS_REVIEW.md`, 50 repositories) proved the workflow. This cohort is
the intended first training dataset.

## Scope and method

- **Universe:** every tier A repository of the v2 review queue (`data/garden/seed_v2/REVIEW_QUEUE.md`)
  with accepted train/validation records. Tier A means:
  - the root license is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause or ISC;
  - the license file is byte-identical at every revision the records use;
  - the package registry (deps.dev) names the same license.

  That is 550 repositories. Tiers B (131, an open question already at the root, e.g. the license
  file changed between revisions), C (19, weak copyleft) and D (141: GPL/AGPL, unknown, none or
  conflicting) are outside this cohort.
- **Excluded from the cohort: 24 tier A repositories (107 records)** that also have holdout or
  temporal-quarantine records. The training loader refuses every record of such a repository, to
  keep the temporal holdout free of same-repository training data. Approving one would make the
  training preflight fail.
- **Inspection** of the remaining 526 at every vulnerable and fixed revision they use:
  - root `NOTICE`/`COPYING`/`PATENTS`/`THIRD_PARTY*` files;
  - every license file name (including suffixed ones such as `LICENSE.GPL`);
  - license files in every directory on the path of a used file;
  - license declarations in `package.json`, `pyproject.toml`, `setup.cfg` and `setup.py`.

  A failed Git read is an open question, never absence. A directory counts as absent only when its
  parent tree was read and does not contain it. Final inspection: every revision complete, 0 failed
  reads.
- **Recommended for review** means the tooling found nothing beyond the root license. It is the
  engineering precondition for a reviewer's decision, not the decision.

## Decisions required

### 1. The three sources (`data/registry.yaml`)

`ghsa-osv-fix-commits` (upstream code, per-repository licenses), `osv` (access path) and
`github-advisory-db` (CC-BY-4.0; the GHSA summaries become finding evidence text). The dossiers and
document hashes are in `docs/PILOT_DATASET_RIGHTS_REVIEW.md` and `docs/RIGHTS_REVIEW.md`. For each
source the reviewer sets the fields that `validate_registry` requires:

| Field | Value after approval |
| --- | --- |
| `production_approved` | `true` |
| `revision` | an immutable identifier of what was reviewed (e.g. `osv-export-2026-11-01` plus the v2 `candidates_sha256`) |
| `license_status` | `reviewed` |
| `terms_status` | `approved_for_project` |
| `redistribution_status` | `not_redistributed` (code and summaries are not republished) or `approved`; `metadata_only` is not accepted for an approved source |
| `reviewer` | name of the authorized reviewer |

### 2. Repository groups

| Group | Repositories | Train/validation records | Reviewer decision |
| --- | ---: | ---: | --- |
| `recommended:MIT` | 318 | 777 | confirm MIT for training use |
| `recommended:Apache-2.0` | 120 | 392 | confirm Apache-2.0; no NOTICE file found on the used paths |
| `recommended:BSD-3-Clause` | 30 | 85 | confirm BSD-3-Clause |
| `recommended:BSD-2-Clause` | 9 | 21 | confirm BSD-2-Clause |
| `recommended:ISC` | 7 | 22 | confirm ISC |
| `clarify:notice_or_additional_license_file` | 36 | 110 | read the listed NOTICE/PATENTS/third-party file; decide or leave out |
| `clarify:package_metadata_names_another_license` | 6 | 13 | resolve the metadata conflict; decide or leave out |

The 484 recommended repositories can be decided per license group: one reading of each license text,
then a check that each template's `license_file_sha256` matches the linked file. The 42
clarification cases name the exact file and revision. Notable ones:

- **NOTICE files whose text the detector reads as AGPL:** imgproxy/imgproxy (MIT root, NOTICE
  detected as AGPL-3.0), streamlit/streamlit (NOTICES detected as AGPL-3.0), openfga/openfga (NOTICE
  detected as GPL-2.0).
- **`PATENTS`** in golang/net.
- **Third-party license bundles** in marimo, zot, datadog/import-in-the-middle and pinchtab.
- **Metadata conflicts:**
  - aimhubio/aim: `setup.py` says MIT, root license Apache-2.0;
  - better-auth: ISC metadata vs MIT root;
  - hapijs/nes: BSD-3-Clause metadata vs BSD-2-Clause root;
  - mat-sz/lettersanitizer: `BSD-3-Clause-Clear`;
  - dojo/dojox: `BSD-3-Clause OR AFL-2.1`;
  - easywebpack-cli: unparseable `package.json`.

Most Apache-2.0 NOTICE files carry only an attribution obligation. They are still left to the
reviewer, because some list bundled code under other terms.

Of the 48 repositories the pilot recommended, 46 are recommended here too. twisted/twisted and
sveltejs/svelte moved to tier B, because v2 adds records at revisions with a different license file.

### 3. Volume if the sources and exactly the 484 recommended repositories were approved

Verified end to end by `scripts/authorization_preflight.py`. It runs the real `curate` export and the
real training loader (`load_training_data`, including the supervision-hash and checksum checks) on a
Git-ignored scratch copy with in-memory hypothetical approvals. Token lengths come from the pinned
Qwen3.5-9B tokenizer (small files only, no weights).

| | train | validation | total |
| --- | ---: | ---: | ---: |
| finding | 538 | 56 | 594 |
| no_finding (verified post-fix) | 494 | 49 | 543 |
| abstain (caller without callee) | 145 | 15 | 160 |
| **detection examples** | **1,177** | **120** | **1,297** |
| patch examples (verified pairs) | 479 | 49 | 528 |

- **Languages:** Python 348, JavaScript 419, Go 313, TypeScript 217.
- **CWEs:** 123 distinct.
- **Findings by family:** resource limits/ReDoS 86, path traversal 68, XSS 63, authn/authz 61,
  other 60, prototype pollution 57, deserialization/input validation 49, information exposure 34,
  command injection 32, crypto/TLS 27, redirect/SSRF 27, code injection 20, SQL injection 10.
- **Token lengths:** detection median 761 and maximum 2,919 tokens, patch maximum 2,918. 0 examples
  exceed 4,096.

The record-ID set hash is `352888ae…` (in `training_cohort.json`). This is about ten times the pilot
(129 records). It is enough for a first real adapter and its validation, but still small for a 9B
model. SQL injection and code injection remain thin.

## Attribution and NOTICE obligations

- **GitHub Advisory Database (CC-BY-4.0):** attribute it, e.g. with a link to
  https://github.com/advisories in the model card and dataset documentation. Note that the summaries
  are used verbatim as evidence text.
- **Upstream code:**
  - MIT, ISC and BSD-2-Clause require keeping the copyright and permission notice with any
    redistributed copy of the code;
  - BSD-3-Clause additionally prohibits endorsement using the authors' names;
  - Apache-2.0 requires keeping the license and any NOTICE and stating changes.

  The code is not redistributed by this project (the manifest is metadata only). Whether model weights
  count as redistribution is open (below). The conservative practice is to list every upstream
  repository and commit (`data/garden/seed_v2/EXTERNAL_RESOURCES.json`) in the model card.

## Open legal questions (unchanged from the pilot, not resolved here)

1. Whether published model weights are a redistribution of MIT/Apache code or of CC-BY text.
2. GitHub Acceptable Use Policies §7 (research use of public information "only if any publications
   resulting from that research are open access"): whether it applies to code fetched over git under
   its own license. Seek advice before any non-open release.
3. OSV service terms beyond the per-source data licenses (none found).
4. Model output reproducing short excerpts of training code (no output-side attribution exists).

## After authorization: exact commands

Verified without approvals as far as possible:
- steps 3 to 5 ran on a scratch copy with simulated approvals (`scripts/authorization_preflight.py`);
- step 6 needs a CUDA GPU and downloads the pinned foundation, so it was not run.

```bash
# 1. Sources: the reviewer edits data/registry.yaml as in the table above (three entries).
# 2. Repositories: copy the completed entries from data/garden/seed_v2/cohort/proposed_approvals.yaml
#    into data/garden/approvals.yaml (every <REQUIRED> field filled; reviewer, date, decision).
python -m sentinel.foundation                                             # schema + registry + approvals check
# 3. Reconstruct the records from the pinned revisions (fresh checkout) and apply the decisions
python -m sentinel.garden reconstruct --config configs/garden/seed_v2.yaml
python -m sentinel.garden curate      --config configs/garden/seed_v2.yaml  # writes train/validation exports
python -m sentinel.garden review      --config configs/garden/seed_v2.yaml  # refresh the queue for the new manifest
python scripts/training_cohort.py     --config configs/garden/seed_v2.yaml  # rebind the cohort (offline)
python -m sentinel.garden check       --config configs/garden/seed_v2.yaml
# 4. Training-data preflight (no model, no GPU)
python -m sentinel.train --config configs/train/qlora_qwen35_9b.yaml --dry-run
# 5. Commit the regenerated manifest (only governance fields change) and run the tests
pytest
# 6. First Qwen3.5-9B QLoRA session (CUDA GPU; approved paid GPU time required)
python -m sentinel.train --config configs/train/qlora_qwen35_9b.yaml
```

To train patch generation as well, add `patch` to `data.tasks` in the training config.
