# Training-Rights Review Workflow

Training on Sentinel Garden data needs **two independent approvals**, both made by an authorized human reviewer. This document describes the process and records the evidence collected so far. **Nothing here is an approval, and no part of the tooling can approve anything.** The engineering team prepared the evidence; the decisions are listed under "Decisions required". **For the first training run, start with `docs/TRAINING_COHORT_RIGHTS_REVIEW.md`** (Garden v2: 484 repositories recommended for review in five license groups, 42 clarification cases, exact field changes for the three sources).

## 1. Roles

| Role | Who | Can do |
| --- | --- | --- |
| Authorized rights reviewer | named by rappidAI Research leadership (legal/licensing responsibility) | approve or block registry sources and repositories |
| Engineer / AI assistant | anyone running the pipeline | collect evidence, regenerate queues, implement gates; **never** approve |

An approval records the reviewer's name. An AI assistant must not be named as a reviewer.

## 2. Source level (`data/registry.yaml`)

A source becomes `production_approved: true` only when the reviewer sets `license_status: reviewed`, `terms_status: approved_for_project`, a resolved `redistribution_status`, an immutable `revision` and their name in `reviewer`. `validate_registry` refuses anything less.

### 2.1 `github-advisory-db` (metadata: advisory ID, CWE, summary, references)

| Fact | Evidence |
| --- | --- |
| License: **CC-BY-4.0** | `https://github.com/github/advisory-database/blob/main/LICENSE.md` (retrieved 2026-11-04 from raw.githubusercontent.com, sha256 `9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94`) |
| GitHub terms: "The GitHub Advisory Database is licensed under the Creative Commons Attribution 4.0 license. The attribution term may be fulfilled by linking to the GitHub Advisory Database at https://github.com/advisories or to individual GitHub Advisory Database records used, prefixed by https://github.com/advisories." | GitHub Terms for Additional Products and Features, section "Advisory Database" (`github/docs` source retrieved 2026-11-04, sha256 `f285d9291d579a75f809cb09ce2e92f4451705f67bb9030fa905db682aebf57d`) |
| What we use | GHSA ID, aliases, CWE IDs, summary (≤300 chars, used as `finding` evidence text), publication date, fix-commit reference. Summaries reach training targets; IDs and summaries are committed in `candidates.jsonl`. |
| Attribution we provide | every record carries its GHSA ID; `review_queue.jsonl` links each advisory as `https://github.com/advisories/<GHSA-ID>`. A model card must state the attribution before release. |

### 2.2 `osv` (aggregator through which GHSA records are read)

| Fact | Evidence |
| --- | --- |
| OSV.dev aggregates upstream databases and lists each one's license; for the GitHub Advisory Database it states CC-BY 4.0 | `google/osv.dev` `docs/data.md` (retrieved 2026-11-04, sha256 `edcfab572b1dc676b9de6453ef7029bef244a24ad655d3f0560f59be3ee96a22`) |
| We only read `GHSA-*` records from the public export `https://osv-vulnerabilities.storage.googleapis.com/<ecosystem>/` and the API `https://api.osv.dev/v1` (audit only) | `configs/garden/seed_v1.yaml` |
| OSV.dev software is Apache-2.0 (code, not data) | `google/osv.dev` `LICENSE` |
| Not established by us | separate terms of service for the public bucket/API beyond the per-source data licenses |

### 2.3 `ghsa-osv-fix-commits` (upstream source code)

| Fact | Evidence |
| --- | --- |
| Code comes from each upstream repository at immutable commits (fix commit and its first parent), fetched anonymously over HTTPS | `data/garden/seed_v1/EXTERNAL_RESOURCES.json` |
| Use: function-level excerpts and upstream patches as training input/targets; **not redistributed** (no code is committed; `redistribution_status: not_redistributed`) | `docs/DATA_GOVERNANCE.md` |
| Rights are per repository | section 3 |

### Decisions required (source level)

1. Approve or block `github-advisory-db`, `osv`, `ghsa-osv-fix-commits` for training use, including the attribution wording for the model card.
2. Confirm that using advisory summaries as training targets is within CC-BY-4.0 with attribution (no other terms apply).
3. Decide whether OSV's bucket/API terms of service need a separate review.
4. Policy for publishing trained weights derived from repositories under different licenses (see 3.2).

## 3. Repository level (`data/garden/approvals.yaml`)

`python -m sentinel.garden review` writes `data/garden/seed_v1/REVIEW_QUEUE.md`, `review/review_queue.jsonl` and approval templates `review/approvals_template_batch_NN.yaml`. For every repository with accepted train/validation records it records:

- root license files with sha256 **at every revision the dataset uses** (vulnerable and fixed) and whether they are identical;
- the default-version license per package on deps.dev (PyPI, npm, Go), as an independent signal;
- record counts per split, advisory links, a link to the license file at the pinned revision.

### 3.1 Tiers (recommended order and depth of review)

| Tier | Meaning | Recommended handling |
| --- | --- | --- |
| A | MIT, Apache-2.0, BSD-2/3-Clause or ISC; identical license file at every used revision; deps.dev agrees | standard review: read the license file at the linked revision, check for NOTICE/additional terms, approve or block |
| B | permissive, but deps.dev disagrees or is silent, the license file changed between revisions, several permissive licenses, or another permissive license (Unlicense, 0BSD, Zlib, PSF) | as A, plus resolve the open question listed in the queue |
| C | weak copyleft (MPL-2.0, LGPL, EPL) | policy decision first; approval needs `non_permissive_justification` |
| D | strong copyleft (GPL, AGPL), unknown, missing or conflicting license | blocked by default; approval needs `non_permissive_justification`, otherwise record `decision: blocked` |

The governance gate enforces this: an approval whose `license_spdx` is not permissive leaves records `pending_review` (`non_permissive_license_requires_justification`) unless it carries a written justification.

### 3.2 Procedure per repository

1. Open the license file link (it pins the fixed revision) and the repository root at the vulnerable revision if the queue reports a change.
2. Check for NOTICE files, per-directory licenses, contributor or trademark terms, and a `LICENSE` that differs from the package metadata.
3. Fill in the template entry (`decision`, `license_spdx`, `terms_status`, `redistribution_status`, `reviewer`, `reviewed_at`) and move it into `data/garden/approvals.yaml`. Templates fail schema validation until every `<REQUIRED>` field is filled.
4. Run `python -m sentinel.garden curate` and `python -m sentinel.foundation`. Only records whose repository approval matches the recorded license file hash become `training_approved`.

An approval is bound to the license file content (`license_file_sha256`). If upstream changes its license file, the approval stops applying and the records return to `pending_review`.

### Decisions required (repository level)

1. Review tier A/B repositories in batch order (most records first).
2. Decide a policy for tier C and D repositories (block, or approve with a written justification).
3. Decide whether redistribution of excerpts (for example a public dataset release) is in scope; until then every approval should use `redistribution_status: not_redistributed`.

## 4. What the tooling guarantees

- Default-deny: empty `approvals.yaml` ⇒ zero training records; `python -m sentinel.train` refuses to run.
- No approval can be generated: templates are not loadable and the gate re-checks every record at training time.
- A revoked or changed approval takes effect at the next `curate` and at training start.
