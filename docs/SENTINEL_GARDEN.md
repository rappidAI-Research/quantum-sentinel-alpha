# Sentinel Garden

Sentinel Garden turns real, published security fixes into function-level training records for Quantum Sentinel Alpha. It is provenance-first and default-deny: the pipeline can discover, resolve, check and export data, but only a documented human review can make a record trainable.

Code: `sentinel/garden/`. Configuration: `configs/garden/seed_v1.yaml`. Governance rules: `docs/DATA_GOVERNANCE.md`.

## Pipeline

```bash
python -m sentinel.garden discover   # OSV export -> frozen candidate list            (network: OSV)
python -m sentinel.garden build      # fix commit + parent -> raw records              (network: GitHub git)
python -m sentinel.garden curate     # quality, dedup, contamination, split, gate      (offline)
python -m sentinel.garden run        # build + curate
python -m sentinel.garden verify     # rebuild from the frozen candidates, compare hashes
python -m sentinel.garden check      # offline manifest consistency (also run by `sentinel.foundation` / CI)
```

| Stage | Module | What happens |
| --- | --- | --- |
| 1. Discovery | `osv.py` | Reads `modified_id.csv` and per-advisory JSON from the public OSV export for PyPI, npm and Go. Keeps GitHub-reviewed `GHSA-*` advisories published before the snapshot bound, with 1-2 localizable CWEs and **exactly one** full-SHA upstream fix commit. Advisories are visited in salted-hash order (deterministic, not biased to recent or old ones). Per-repository and per-ecosystem caps keep one project from dominating. The result is frozen in `data/garden/seed_v1/candidates.jsonl`. |
| 2. Resolution | `gitsource.py` | One bare, partial, shallow cache per upstream repository (`git fetch --depth=2 --filter=tree:0 origin <sha>`; trees and blobs are fetched lazily, only along the changed paths). Vulnerable revision = first parent of the fix commit (for merge commits the first-parent diff is the merged change). Everything is addressed by full commit SHA. |
| 3. Extraction | `build.py`, `code.py`, `diff.py` | Only non-test, non-vendored source files in the ecosystem's language count. The logic change must be confined to **one function** (comment/import-only hunks are ignored; new complete top-level helpers added by the fix are allowed and included in the fixed excerpt). Function boundaries: Python `ast` (indentation fallback for Python 2), Go gofmt layout, JS/TS brace matching over a string/comment/regex/template mask. |
| 4. Provenance & license | `build.py`, `licenses.py` | Every record carries advisory ID, aliases, OSV record hash, repository, both revisions, fix commit date, file path, function, file line ranges, content and normalized hashes, and the root license files (path + sha256) at the fixed revision. License *detection* is a hint for the reviewer and is always stored as `status: unreviewed`. |
| 5. Quality | `quality.py` | Record checks (below), pair checks, exact and near-duplicate detection, label-conflict detection, evaluation-contamination checks. |
| 6. Split, gate, export | `split.py`, `governance.py`, `pipeline.py`, `render.py` | Repository-level split with temporal cutoff, default-deny training gate, deterministic JSONL export, statistics and reports. |

## Record types

Each accepted fix yields up to three records sharing a `group_id`:

| Kind | Decision | Code | Label source |
| --- | --- | --- | --- |
| `vulnerable` | `finding` | changed function at the vulnerable revision | CWE from the advisory; span = lines removed by the fix, or the insertion point of an added check (`span_kind`); evidence = advisory summary; `patch` = upstream fix restricted to the function |
| `post_fix` | `no_finding` | same function at the fix revision (+ helpers the fix added) | secure hard negative **scoped to the advisory's weakness** (`label_scope`); other weaknesses are not verified |
| `caller_abstention` | `abstain` | an unchanged function in the same file that calls the vulnerable function, shown without the callee's body | evidence is insufficient by construction: the vulnerable code is withheld (`withheld_callee`) |

`input` has the same fields as an evaluation case (`language`, `file`, `context`, `code`), so `sentinel.eval.prompt.build_messages` renders training and evaluation identically. `render.to_chat_example` produces the chat example; every rendered reply is parsed with the strict QuickEval parser and must reproduce the label (`target_matches_eval_contract`).

## Quality checks

Record level (a failure rejects the record): schema, content/normalized hashes, kind-decision consistency, language matches ecosystem, non-test source path, minimum length, context bounds, syntax well-formedness of the excerpt, high-confidence secret patterns, evidence completeness, CWE validity (no pillar / non-code CWE), localization inside the excerpt and on non-trivial lines, diff consistency (the function-level patch contains exactly the upstream changed lines), revision consistency, label scope present, abstention excerpt calls but does not define the withheld callee, rendered target round-trips through the eval parser.

Pair level: the stored patch must apply strictly to the vulnerable excerpt and produce the fixed excerpt; both sides must share advisory, repository, revisions and file. A failing vulnerable record also rejects its siblings.

Dataset level:

- **Exact duplicates**: normalized-token sha256 (comments and layout ignored); the first record in `record_id` order is kept.
- **Near duplicates**: Jaccard similarity of 5-token shingles (stable blake2b hashes) ≥ 0.85 with the same decision, different fix.
- **Label conflicts**: identical normalized code with different decisions, or ≥ 0.92 similarity with different decisions. A `post_fix` that is near-identical to another advisory's `vulnerable` function (an incomplete fix) loses its hard-negative label.
- **Evaluation contamination**: every case under `eval/sentinelbench/` (including the Git-ignored `private/`) and every `coding-v1` task: exact normalized hash, Jaccard ≥ 0.5, containment of ≥ 80 % of an evaluation snippet's shingles, or the same upstream repository as an upstream-derived evaluation case.

## Splits

- Repository-level buckets by salted sha256 of the repository key: 80 % train, 10 % validation, 10 % `holdout_reserved`. A repository never appears in two buckets.
- Temporal cutoff (provisional, `2026-07-01`): a fix committed on/after it never enters train or validation. In a train/validation repository it becomes `temporal_quarantine`; in a holdout repository it stays a holdout candidate.
- `holdout_reserved` records are also exported as `upstream_derived` evaluation cases (`holdout_reserved.eval_cases.jsonl`, validated against `eval_case.schema.json`). Their identities are public (the candidate list and salt are committed), so they are a **reservation for temporal/generalization checks, not a private holdout**. A private SentinelBench holdout must come from sources that are not listed publicly.

## Governance gate

A record is `training_approved` only if it passed every check, is in train/validation, every registry source it depends on (`ghsa-osv-fix-commits`, `osv`, `github-advisory-db`) is `production_approved`, and its repository has an `approved_for_training` entry in `data/garden/approvals.yaml` whose reviewed license file sha256 matches the record's. Otherwise it is `pending_review` (with named blockers) or `excluded`. `REVIEW_QUEUE.md` lists the repositories to review with the exact license file links and hashes.

## Outputs

Git-ignored work directory `artifacts/garden/seed_v1/`:

| File | Content |
| --- | --- |
| `cache/osv/`, `cache/git/` | OSV downloads, partial git caches |
| `records.raw.jsonl`, `build_log.jsonl` | build output and per-candidate outcome |
| `records.jsonl` | every curated record with code, split, quality and governance |
| `staging.jsonl` | accepted, **not approved** records (the current seed) |
| `train.jsonl`, `validation.jsonl`, `train.chat.jsonl` | training-approved records only (empty until approvals exist) |
| `holdout_reserved.jsonl`, `holdout_reserved.eval_cases.jsonl` | holdout reservation |
| `rejected.jsonl` | rejected records with failed checks |

Committed directory `data/garden/seed_v1/` (metadata only, no upstream code, context, evidence text or patch):

| File | Content |
| --- | --- |
| `candidates.jsonl` | frozen discovery output (advisory metadata incl. GHSA summary, CWE, repository, fix SHA) |
| `discovery_stats.json` | per-ecosystem discovery outcomes |
| `manifest.jsonl` | every record's provenance, label metadata, hashes, split, quality and governance |
| `stats.json`, `REPORT.md` | statistics computed from the manifest |
| `REVIEW_QUEUE.md` | repositories awaiting rights/terms review |

Upstream code is not committed because its redistribution has not been reviewed (`redistribution_status: not_redistributed`) and this repository has no declared license yet. Anyone can rebuild the records byte-for-byte from `candidates.jsonl` and the upstream repositories; `verify` checks every content hash. GHSA metadata in `candidates.jsonl` comes from the GitHub Advisory Database (CC-BY-4.0) via OSV.

## Known limitations

- Labels are derived from maintainers' fixes and GitHub-reviewed advisories; they were not re-verified by running exploits or tests. A `post_fix` record is a hard negative for the advisory's weakness only.
- `finding` evidence is the advisory summary, not a hand-written source-to-sink explanation. Patch records are upstream patches, not model-verified remediations.
- The one-function rule favours small, local fixes; multi-function, cross-file and configuration fixes are not represented.
- JS/TS function detection is heuristic; unusual layouts are rejected rather than guessed, which lowers yield.
- The temporal cutoff and split salt are provisional and must be frozen before large-scale collection.
