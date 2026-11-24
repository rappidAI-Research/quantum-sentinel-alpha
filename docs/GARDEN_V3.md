# Sentinel Garden v3

`sentinel-garden-v3` (`configs/garden/seed_v3.yaml`, committed metadata in `data/garden/seed_v3/`) is the
dataset the first QLoRA run trains on (`configs/train/qlora_qwen35_9b.yaml`). v1 and v2 stay immutable
references: their committed files are unchanged, and every v1 and v2 record is in v3 with the same
content hash and split.

## What changed

**1. Advisories with several fix commits (discovery + build).** v2 dropped every advisory that names
more than one fix commit (2,076 of them, 1,927 in a single repository). v3 keeps the single-repository
ones as candidates. `build` (`resolve_fix_commit` in `sentinel/garden/build.py`) uses one only when the
fix is unambiguous, in one of two cases:

- exactly one commit changes runtime source in the ecosystem's language, and every other commit
  changes only tests, documentation, changelogs or CI files;
- every commit that changes runtime source makes the identical change: same files, same removed and
  added lines. This covers a backport, or a merge commit and the commit it merges. The earliest such
  commit is used, which is the date the fix existed.

Everything else is rejected and never guessed:
- a fix split over different commits (`multi_commit_distinct_changes`);
- a companion commit that touches dependency manifests, vendored code, other languages or
  configuration, any of which might itself be the fix (`multi_commit_unexplained_change`).

Multi-commit candidates only take repository-cap slots that single-commit candidates leave free, so the
v3 candidate list is a strict superset of v2's: all 6,623 v2 candidates are byte-identical, and 1,128
are added. Tests: `tests/test_garden_v3_multicommit.py`.

Result:
- **124 of the 1,128 multi-commit advisories** yield records.
- 693 are rejected as distinct changes. In a sample of 120:
  - a third change different files;
  - a third are file subsets (a fix in several steps);
  - a third change the same files with different lines (partial fixes, or backports to diverged
    branches).

  Telling a backport from a partial fix would need each repository's full history (ancestry). That
  costs too much and is not reliable on the shallow caches, so these stay rejected.
- 116 are rejected because a companion commit changes something else.

**2. Curate applies the training loader's repository rule.** The loader refuses every record of a
repository that also has holdout or temporal-quarantine records. In v3, two approved repositories
gained post-cutoff records (encode/django-rest-framework, protobufjs/protobuf.js), so the curated
export would have made training fail. `curate` now marks such records
`excluded: repository_has_holdout_or_quarantine_records` (6 records). v1 and v2 are byte-identical
under the rule. Test: `test_approved_repository_with_quarantine_records_exports_nothing`.

## Size (verified)

| | v2 | v3 |
| --- | ---: | ---: |
| Candidates | 6,623 | 7,751 |
| Fix commits turned into records | 1,725 | 1,849 |
| Records built / accepted | 3,971 / 3,226 | 4,259 / **3,465** |
| finding / no_finding / abstain | 1,476 / 1,307 / 443 | 1,581 / 1,408 / 476 |
| verified vulnerable/fixed pairs | 1,278 | 1,376 |
| train / validation / holdout / quarantine | 2,276 / 252 / 376 / 322 | 2,427 / 275 / 399 / 364 |
| finding CWEs | 192 | 195 |

New records:
- 243 of the 288 new records were accepted. The rest were rejected by the unchanged quality checks and
  audit.
- 4 v2-accepted records are now rejected: duplicates, label conflicts, or an audit leakage finding
  against the new records.
- No v2 training-approved record was lost.

Integrity:
- `garden verify` rebuilt all 4,259 records from the pinned revisions with 0 differences.
- The second and third audit rounds were byte-identical: the dataset is at a fixed point.

## Training data (after the 2026-11-24 decision, `docs/TRAINING_AUTHORIZATION.md`)

| | train | validation |
| --- | ---: | ---: |
| finding | 606 | 63 |
| no_finding (verified post-fix) | 556 | 54 |
| abstain | 171 | 16 |
| **detection examples** | **1,333** | **133** |
| patch examples | 541 | 54 |
| **training examples (detection + patch)** | **1,874** | **187** |

- **Coverage:** 533 repositories. By language: JavaScript 428, Go 412, Python 404, TypeScript 222.
- **Findings:** 669 across 131 CWEs.
- **Findings by family:** resource limits/ReDoS 94, path traversal 78, authn/authz 73, other 70, XSS 67,
  prototype pollution 58, deserialization/input validation 53, information exposure 45, command
  injection 38, redirect/SSRF 32, crypto/TLS 29, code injection 22, SQL injection 10.
- **Token budget:** with the trainer's own encoding and the pinned Qwen3.5-9B tokenizer, 0 examples
  are dropped at 4,096 tokens. Training examples have a median of 763 and a maximum of 2,918 tokens,
  1.73 M tokens per epoch.

## Training machine: `materialize`

`python -m sentinel.garden materialize --config configs/garden/seed_v3.yaml` rebuilds only the
training-approved records of the committed manifest and writes exactly the files `sentinel.train`
reads, plus `SHA256SUMS`. It builds 685 advisories in 533 repositories instead of all 7,751 candidates.

It refuses on any of these:
- a record that cannot be rebuilt;
- a content or supervision hash that differs from the manifest;
- an approval that no longer passes the gate.

Quality, split and approval status come from the committed manifest and are never recomputed.
`records.raw.jsonl` is not touched.

On v2 and v3 its four files are byte-identical to the full `curate` export, taking 1.5 minutes from a
warm cache. Tests: `tests/test_garden_materialize.py`.

## Reproduce

```bash
python -m sentinel.garden discover --config configs/garden/seed_v3.yaml
python -m sentinel.garden build    --config configs/garden/seed_v3.yaml   # ~20 min, upstream git
python -m sentinel.garden curate   --config configs/garden/seed_v3.yaml
python -m sentinel.garden audit    --config configs/garden/seed_v3.yaml   # then curate again (fixed point)
python -m sentinel.garden verify   --config configs/garden/seed_v3.yaml
python scripts/training_cohort.py  --config configs/garden/seed_v3.yaml --inspect --extension
```
