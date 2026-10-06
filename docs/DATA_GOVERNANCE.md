# Sentinel Garden Data Governance

Sentinel Garden is provenance-first. A record can enter a training manifest only after its source passes independent rights/provenance checks and the record itself passes extraction, deduplication, split and verification checks.

## Default-deny rule

`production_approved: false` is the default for every source. Public accessibility is not approval. Approval requires an immutable revision, reviewed license status, project terms approval, resolved redistribution status and a named reviewer.

Evaluation-only sources remain excluded from training even when they are publicly downloadable. Ground truth, benchmark patches and test oracles must not be copied into training prompts.

## Planned record-level provenance

Each eventual training/evaluation record should preserve at least: source repository, vulnerable revision, fixed revision, diff, repo license/terms status, advisory metadata where applicable, vulnerable/fixed context, localization, concise evidence, label, verification artifacts, language/framework, split and content hashes.

## Split policy

Splits are repository-level, not function-level. A repository must not contribute to both training and private SentinelBench holdout. A temporal cutoff will be frozen before large-scale collection. Exact and normalized/fuzzy hashes will be used to reduce benchmark leakage.
