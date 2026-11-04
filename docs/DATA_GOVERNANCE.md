# Sentinel Garden Data Governance

Sentinel Garden is provenance-first. A record can enter a training manifest only after its source passes independent rights/provenance checks and the record itself passes extraction, deduplication, split and verification checks. The implementation is described in `docs/SENTINEL_GARDEN.md`.

## Default-deny rule

`production_approved: false` is the default for every source. Public accessibility is not approval. Approval requires an immutable revision, reviewed license status, project terms approval, resolved redistribution status and a named reviewer (enforced by `sentinel.data.registry.validate_registry`).

Evaluation-only sources remain excluded from training even when they are publicly downloadable. Ground truth, benchmark patches and test oracles must not be copied into training prompts.

## Two approval levels for upstream code

Code resolved from upstream repositories needs both:

1. **Source level** (`data/registry.yaml`): `ghsa-osv-fix-commits` (the code source) and the metadata sources it depends on (`osv`, `github-advisory-db`) must be `production_approved`.
2. **Repository level** (`data/garden/approvals.yaml`): an `approved_for_training` entry per upstream repository, written by an authorized reviewer after reading the repository's license and terms at the pinned revision. The entry records the reviewed license file and its sha256; if the record's license file differs, the approval does not apply. `blocked` entries exclude a repository permanently.

No repository and no source is approved today. Neither the pipeline nor an automated agent may add approvals; a detected license is a hint, not a review.

## Record-level provenance (implemented)

Every Sentinel Garden record preserves: advisory ID and aliases, OSV record hash, source and metadata source IDs, upstream repository, vulnerable and fixed revisions (full SHAs), fix commit date, file path, function name and file line ranges, root license files with sha256 at the fixed revision, detected license (unreviewed), label, localization method, evidence source, content sha256 and normalized-token sha256. See `data/schemas/garden_record.schema.json`.

## Redistribution

Upstream code, excerpts, contexts and patches are **not committed** to this repository: redistribution has not been reviewed and the repository license is not declared. Only metadata (`data/garden/seed_v1/`) is committed, and records are rebuilt deterministically from immutable upstream revisions (`python -m sentinel.garden verify`).

## Split policy

Splits are repository-level, not function-level: a repository contributes to exactly one of train, validation and `holdout_reserved`, and must not contribute to both training and the private SentinelBench holdout. A provisional temporal cutoff (`2026-07-01`, after the selected foundation checkpoint was published) keeps newer fixes out of train and validation; it must be frozen before large-scale collection. Exact, normalized and fuzzy hashes and an embedded-snippet check are used against every evaluation case and coding task to prevent benchmark leakage.

## Removal

`removal_status: ready` means a repository or advisory can be removed by adding a `blocked` approval entry (repository) or by regenerating the frozen candidate list without it, then re-running `build` and `curate`. Because no code is committed, removal never requires rewriting Git history.
