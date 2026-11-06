# Sentinel Garden v2

`sentinel-garden-v2` (`configs/garden/seed_v2.yaml`, committed metadata in `data/garden/seed_v2/`) is the
dataset the first QLoRA run will train on (`configs/train/qlora_qwen35_9b.yaml`). Garden v1
(`configs/garden/seed_v1.yaml`, `data/garden/seed_v1/`) stays the immutable reference. The current code
rebuilds all 2,773 v1 records with identical content hashes (`garden verify`: 0 problems), and v1's
committed files are unchanged.

**Nothing is training-approved.** All v2 numbers below describe accepted records awaiting rights
review. They are not a dataset anyone has authorized.

## What changed against v1

Same OSV snapshot (advisories published before 2026-11-01, the same sample order and salt), the
same CWE selection, the same quality thresholds and the same frozen split policy
`sentinel-garden-split-v1`. Every v1 candidate is a v2 candidate with an identical entry. Two
documented differences:

1. **`max_candidates_per_repository` 4 → 20.** The v1 cap dropped 3,236 advisories only because their
   repository already had four candidates. The cap limits one project dominating the data. It is not
   a quality rule. At 20, the largest train repository holds 0.8 % of the train records
   (apache/airflow, 18). The split is per repository, so extra advisories of a project stay in that
   project's split.
2. **Three extraction refinements** (`sentinel/garden/build.py`; each is off by default and enabled only
   in the v2 config, so v1 rebuilds unchanged):
   - `member_additions`: a fix that adds **complete** new helper methods or functions inside a class or
     object body (as well as changing one function) is localized like a v1 top-level helper. The helper
     goes into the fixed excerpt and patch. It is never labelled vulnerable. Every inserted line must
     belong to a function that the insertion adds completely. A field, a statement or part of a
     function keeps the v1 rejection.
   - `version_hunks_as_noise`: a hunk outside every function that only changes a version-string
     assignment (`__version__ = "1.2.3"`, `const Version = "v1.2.3"`) is metadata, not logic.
   - `tooling_files_excluded`: build, packaging and test tooling that is not shipped runtime code
     (`*.config.{js,ts,mjs}`, `.eslintrc.js`, `gulpfile.js`, `Gruntfile.js`, `karma.conf.js`, root-style
     `test.js`/`tests.py`, `setup.py`, `noxfile.py`, `fabfile.py`, `magefile.go`) no longer counts as a
     changed source file. v1 already ignored `package.json`, `go.mod` and tests in test directories.

What was **not** changed: the one-function rule (`max_changed_functions: 1`), the file, line and size
limits, the quality checks, the audit, the split policy and the governance gate.

## Rejection analysis (why only a little was recoverable)

300 randomly sampled rejections per category were classified against the real upstream diffs (a
one-off analysis; its findings are summarised here and each fix has regression tests in
`tests/test_garden_v2_extraction.py`):

- `change_outside_function` (1,071 in v1): mostly genuine. Typical cases are fixes in module-level
  script code, configuration objects or literals (e.g. a server `hostname` default), Go `var`/`const`
  blocks or struct fields, or class bodies outside any method. These cannot be given a function-level
  label. Extractor limitations were new helper methods in a class body, version bumps and changed
  tooling files.
- `multiple_functions_changed` (723 in v1): mostly genuine multi-function fixes. Examples: the same
  regex fixed in `hsl` and `hwb` (each function is vulnerable), or a check added in one function and
  a call site adapted in another. There is no reliable way to tell which function carries the
  weakness, so these stay rejected. The only extractor defect was test or tooling files counted as
  source (e.g. `test.js` next to `index.js`).

Effect on the 4,830 v1 candidates: **58 recovered** (38 `change_outside_function`, 14
`multiple_functions_changed`, 6 others). 74 more moved from `change_outside_function` to
`top_level_addition_without_changed_function`: the helper was recognized but no single changed
function owns it in the same file, so they stay rejected. The bulk of the growth comes from the
repository cap.

## Size (verified)

| | v1 | v2 | Δ |
| --- | ---: | ---: | ---: |
| Candidates | 4,830 | 6,623 | +1,793 |
| Fix commits turned into records | 1,206 | 1,725 | +519 |
| Records built | 2,773 | 3,971 | +1,198 |
| **Records accepted** | **2,309** | **3,226** | **+917** |
| vulnerable (`finding`) | 1,047 | 1,476 | +429 |
| post-fix hard negatives (`no_finding`) | 954 | 1,307 | +353 |
| caller abstentions (`abstain`) | 308 | 443 | +135 |
| complete verified vulnerable/fixed pairs (patch-eligible) | 941 | 1,278 | +337 |
| train / validation | 1,724 / 191 | 2,276 / 252 | +552 / +61 |
| holdout_reserved / temporal_quarantine | 235 / 159 | 376 / 322 | +141 / +163 |
| repositories with accepted records | 906 | 1,004 | +98 |
| finding CWEs | 161 | 192 | +31 |

By language (accepted): Go 916, JavaScript 783, Python 1,046, TypeScript 481. Findings by CWE family:
resource limits / ReDoS 242, other 175, authn/authz 165, path traversal 151, XSS/output encoding 148,
deserialization / input validation 117, information exposure 116, prototype pollution 92, command
injection 78, redirect/SSRF 72, crypto/TLS/integrity 62, code injection 39, SQL injection 19. Full
tables: `data/garden/seed_v2/REPORT.md`.

**Rejections of new records:** 926 of the 1,198 new records were accepted, so 22.7 % were rejected.
For the 132 records from recovered extractions the rejection rate is 22.0 %. For the 1,066 records
from the higher cap it is 22.8 %. Main reasons: audit exclusions (later fixes changing the same
function, reported incomplete fixes, broad spans), siblings of excluded records, near/exact label
conflicts, and trivial or diff-inconsistent spans. 12 v1-accepted records are rejected in v2 for
legitimate reasons:
- 7 exact label conflicts with new records (identical code labelled differently by two advisories);
- 2 near duplicates and 1 exact duplicate;
- 2 sharing a package with a validation repository.

## Integrity

- `garden verify --config configs/garden/seed_v2.yaml`: all 3,971 records rebuilt from the pinned
  upstream revisions, 0 content-hash differences.
- Audit + curate is a byte-identical fixed point. The second and third audit rounds give the same
  exclusions (518 automated, 10 AI-assisted manual) and the same manifest (sha256 `e8250674…`).
- Every v1 record is in v2 with the same content hash and the same split. No repository changed split
  (`tests/test_training_cohort.py`).
- Contamination checks against every QuickEval and seed case and `coding-v1` ran in `curate` as for
  v1. Split-leakage checks (shared package, near duplicate of quarantine) ran in the audit.
- New records carry the v1 supervision hashes (input, target, detection and patch example). The
  training loader recomputes them (`authorization_preflight.py` below).
- The AI-assisted manual review of v1 (`audit/manual_review.yaml`) applies unchanged to the same
  groups. It is **not** an independent human review.

## Reproduce

```bash
python -m sentinel.garden discover --config configs/garden/seed_v2.yaml   # OSV export (cached snapshot)
python -m sentinel.garden build    --config configs/garden/seed_v2.yaml   # upstream git, ~15 min
python -m sentinel.garden curate   --config configs/garden/seed_v2.yaml
python -m sentinel.garden audit    --config configs/garden/seed_v2.yaml   # OSV API + git; then curate again
python -m sentinel.garden verify   --config configs/garden/seed_v2.yaml   # rebuild, compare hashes
python -m sentinel.garden check    --config configs/garden/seed_v2.yaml   # offline, also in CI
```

After a fresh checkout, `python -m sentinel.garden reconstruct --config configs/garden/seed_v2.yaml`
rebuilds the Git-ignored records and exports from the pinned revisions and checks every hash.
