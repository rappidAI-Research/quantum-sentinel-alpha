# Sentinel Garden

Sentinel Garden turns real, published security fixes into function-level training records for Quantum Sentinel Alpha. It is provenance-first and default-deny: the pipeline can discover, resolve, check and export data, but only a documented human review can make a record trainable.

Code: `sentinel/garden/`. Configuration: `configs/garden/seed_v1.yaml`. Governance rules: `docs/DATA_GOVERNANCE.md`.

## Pipeline

```bash
python -m sentinel.garden discover   # OSV export -> frozen candidate list            (network: OSV)
python -m sentinel.garden build      # fix commit + parent -> raw records              (network: GitHub git)
python -m sentinel.garden curate     # quality, dedup, contamination, split, gate      (offline)
python -m sentinel.garden run        # build + curate
python -m sentinel.garden audit      # automated quality audit (network: OSV API, upstream git)
python -m sentinel.garden review     # rights-review evidence and queue (network: deps.dev, upstream git)
python -m sentinel.garden verify     # rebuild from the frozen candidates, compare hashes
python -m sentinel.garden reconstruct  # fresh checkout -> full records + SHA256SUMS (see "Reconstruction")
python -m sentinel.garden check      # offline manifest consistency (also run by `sentinel.foundation` / CI)
```

| Stage | Module | What happens |
| --- | --- | --- |
| 1. Discovery | `osv.py` | Reads `modified_id.csv` and per-advisory JSON from the public OSV export for PyPI, npm and Go. Keeps GitHub-reviewed `GHSA-*` advisories published before the snapshot bound, with 1-2 localizable CWEs and **exactly one** full-SHA upstream fix commit. Advisories are visited in salted-hash order (deterministic, not biased to recent or old ones). Per-repository and per-ecosystem caps keep one project from dominating. The result is frozen in `data/garden/seed_v1/candidates.jsonl`. |
| 2. Resolution | `gitsource.py` | One bare, partial, shallow cache per upstream repository (`git fetch --depth=2 --filter=tree:0 origin <sha>`; trees and blobs are fetched lazily, only along the changed paths). Fetched commits are tracked in a marker file instead of probing Git objects, because probing a missing object in a partial clone lazily fetches its entire history; parents are read from the raw commit object, so a fix commit that is the shallow boundary of another fetch keeps its parent. Vulnerable revision = first parent of the fix commit (for merge commits the first-parent diff is the merged change). Everything is addressed by full commit SHA. |
| 3. Extraction | `build.py`, `code.py`, `diff.py` | Only non-test, non-vendored source files in the ecosystem's language count. The logic change must be confined to **one function** (comment/import-only hunks are ignored; new complete top-level helpers added by the fix are allowed and included in the fixed excerpt). Function boundaries: Python `ast` (indentation fallback for Python 2); Go, JavaScript and TypeScript from tree-sitter parse trees (`syntax.py`, grammar versions pinned in `pyproject.toml` and recorded in `stats.json`), which also supply literals, comments, normalization tokens, calls, imports and local names. |
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

`input` has the same fields as an evaluation case (`language`, `file`, `context`, `code`), so `sentinel.eval.prompt.build_messages` renders training and evaluation identically (prompt `quickeval-review-v2`). `render.to_chat_example` produces the chat example; every rendered reply is parsed with the strict QuickEval parser and must reproduce the label without a confidence value (`target_matches_eval_contract`; docs/CALIBRATION.md). Verified vulnerable/fixed pairs also yield patch-generation examples (docs/PATCH_SUPERVISION.md).

Every manifest row carries a `supervision` block (`sentinel/garden/supervision.py`): canonical sha256 of the full input and the full target (including evidence text and patch), of the rendered detection example and, for patch-eligible records, of the patch example, plus the patch-eligibility reason. The training loader recomputes them (docs/TRAINING.md), so the Git-ignored exports cannot be edited without detection; `check` validates the block structurally.

## Quality checks

Record level (a failure rejects the record): schema, content/normalized hashes, kind-decision consistency, language matches ecosystem, non-test source path, minimum length, context bounds, syntax well-formedness of the excerpt, high-confidence secret patterns, evidence completeness, CWE validity (no pillar / non-code CWE), localization inside the excerpt and on non-trivial lines, diff consistency (the function-level patch contains exactly the upstream changed lines), revision consistency, label scope present, abstention excerpt calls but does not define the withheld callee, rendered target round-trips through the eval parser.

Pair level: the stored patch must apply strictly to the vulnerable excerpt and produce the fixed excerpt; both sides must share advisory, repository, revisions and file. A failing vulnerable record also rejects its siblings.

Dataset level:

- **Exact duplicates**: normalized-token sha256 (comments and layout ignored); the first record in `record_id` order is kept.
- **Near duplicates**: Jaccard similarity of 5-token shingles (stable blake2b hashes) ≥ 0.85 with the same decision, different fix.
- **Label conflicts**: identical normalized code with different decisions, or ≥ 0.92 similarity with different decisions. A `post_fix` that is near-identical to another advisory's `vulnerable` function (an incomplete fix) loses its hard-negative label.
- **Evaluation contamination**: every case under `eval/sentinelbench/` (including the Git-ignored `private/`) and every `coding-v1` task: exact normalized hash, Jaccard ≥ 0.5, containment of ≥ 80 % of an evaluation snippet's shingles, or the same upstream repository as an upstream-derived evaluation case.

## Splits

The split policy is **frozen** in `configs/garden/split_policy.yaml` (`sentinel-garden-split-v1`, frozen 2026-11-04) and shared by every Garden dataset version, so a repository keeps its split across versions and can never move from holdout or validation into training. `tests/test_garden_split_policy.py` pins the values; changing them means a new policy ID and an explicit decision.

- Repository-level buckets by `sha256(salt + "\0" + repository_key) % 1000`: 80 % train, 10 % validation, 10 % `holdout_reserved`. A repository never appears in two buckets.
- Temporal cutoff `2026-07-01`: a fix committed on/after it never enters train or validation. In a train/validation repository it becomes `temporal_quarantine`; in a holdout repository it stays a holdout candidate. Rationale: the selected checkpoint was published on 2026-02-27; four months later every newer fix is unseen by our adapters and outside the foundation's pre-training period, while 2018 to mid-2026 history stays available for training.
- `holdout_reserved` records are also exported as `upstream_derived` evaluation cases (`holdout_reserved.eval_cases.jsonl`, validated against `eval_case.schema.json`). Their identities are public (the candidate list and salt are committed), so they are a **reservation for temporal/generalization checks, not a private holdout**. A private SentinelBench holdout must come from sources that are not listed publicly.

## Quality audit

`python -m sentinel.garden audit` checks every fix group with real evidence and writes `data/garden/seed_v1/audit/` (report: `AUDIT_REPORT.md`):

| Check | Evidence | Effect |
| --- | --- | --- |
| Advisory ↔ code correspondence | full OSV advisory text and the upstream commit message: function, file or changed identifier named; commit cites the advisory | strength strong / moderate / weak, reported |
| Fix pattern vs CWE family | remediation indicators in the changed lines (e.g. parameterised queries for CWE-89, path normalisation for CWE-22) | heuristic flag for manual review, never an exclusion |
| Hard-negative validity | every other OSV advisory of the same package: a later upstream fix that changes the same function, an advisory that calls this fix incomplete, or a later fix commit in the same repository that cannot be read (unverified) | `post_fix` record excluded; an incomplete recheck keeps the exclusions of the previous run |
| Abstention validity | the call passes non-constant data; the caller does not duplicate the vulnerable code | abstention excluded |
| Split leakage | the same advisory/CVE or fix commit in two splits, near-identical code (Jaccard ≥ 0.7) across splits, or the same published package (ecosystem + name) from another repository in validation or holdout_reserved (moved, renamed or forked repositories, e.g. `visionmedia/send` → `pillarjs/send`) | train-side record excluded; holdout/validation records are never moved |
| Localization precision | span length and share of the function | a broad span (> 12 lines and > 40 % of the function) excludes the vulnerable record (and with it the group): in the manual sample every broad span was imprecise |

A deterministic stratified sample (per language: pairs round-robin over CWE families, plus abstentions) is reviewed by hand; verdicts are stored with the reviewer's identity in `audit/manual_review.yaml` and reported with Wilson 95 % intervals. Rates count only items of the current sample (sampled items without a verdict are reported as pending; a verdict on a group that is no longer sampled keeps its exclusions but is not counted), once for the whole sample and once for the items that survive the automated exclusions - the estimate for the exported data. Exclusions (`audit/exclusions.yaml`, automated and manual sections) are applied by `curate` as failed checks `audit:<reason>`; records are only ever excluded, never relabelled. No exploit or test is executed.

**Independent human review.** The sample above is AI-reviewed. `python -m sentinel.garden review-packages` draws a separate deterministic sample of the exportable population (accepted train/validation groups; per language 12 pairs round-robin over CWE families and 4 abstentions; salt `sentinel-garden-human-review-v1`) and writes one blind evidence package per item to the Git-ignored `review_packages/` (advisory text and links, fix and vulnerable commit links, the vulnerable excerpt with the labelled span marked, the function-level patch, the fixed excerpt or the abstention excerpt with its withheld callee, expected labels and a checklist on label correctness, CWE attribution and ambiguity, localization, hard-negative safety, pair consistency, abstention validity and recognised duplicates). AI verdicts and automated signals are kept out of the package (`signals.json` next to it). The committed index `audit/human_review_sample.jsonl` binds each item to the package sha256; a reviewer copies `audit/human_review.TEMPLATE.yaml` to `audit/human_review.yaml`, records their identity and whether they are independent of the implementation, and decides each item. `audit` validates the file, ignores decisions on changed packages (stale), applies `exclude_records` as `audit:human_review:*` and reports human error rates separately from automated and AI results. No tool writes decisions; until a person does, the report states that no independent review was performed.

**Temporal quarantine overlap.** By design, post-cutoff fixes of training repositories go to `temporal_quarantine` (never trained on). The audit reports how many quarantine records share a repository or a package with train/validation (20 and 6 of 159 on the current seed); results on the quarantine must report them separately, as the quarantine is not repository-disjoint.

## Rights review

`python -m sentinel.garden review` prepares the repository-level review (tiers A-D, license files at every used revision, deps.dev cross-check, approval templates). Process and source dossiers: `docs/RIGHTS_REVIEW.md`.

## Reconstruction after a fresh checkout

Upstream code is not committed. To rebuild every record:

```bash
pip install -e '.[dev]'
python -m sentinel.garden reconstruct   # needs anonymous HTTPS access to github.com only
cd artifacts/garden/seed_v1 && sha256sum -c SHA256SUMS
```

`reconstruct` checks that every repository in `data/garden/seed_v1/EXTERNAL_RESOURCES.json` is reachable, rebuilds all records from the frozen `candidates.jsonl` at the pinned commits, compares every content hash with `manifest.jsonl`, re-curates, writes the exports with `SHA256SUMS` and a `reconstruction_report.json`. Committed metadata is never modified. If an upstream repository or commit has disappeared, its records are listed as missing (`missing_because_unreachable`) and are **not substituted**; the remaining records are still exported. Any content-hash mismatch, unexpected record or change of acceptance/split fails the reconstruction (exit code 1). OSV and deps.dev are only needed to re-run `discover`, `audit` or `review`, not to reconstruct.

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
| `train.jsonl`, `validation.jsonl`, `train.chat.jsonl`, `train.patch.chat.jsonl` | training-approved records and their detection / patch examples only (empty until approvals exist) |
| `holdout_reserved.jsonl`, `holdout_reserved.eval_cases.jsonl`, `holdout_reserved.patch_cases.jsonl` | holdout reservation (detection and patch cases) |
| `review_packages/<item>/package.md`, `signals.json` | blind evidence packages for the independent human review |
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
- `finding` evidence is the advisory summary, not a hand-written source-to-sink explanation. Patch ground truth is the upstream fix of an audited pair; it is not shown to be secure (no exploit or test is run).
- The one-function rule favours small, local fixes; multi-function, cross-file and configuration fixes are not represented.
- Go/JS/TS structure comes from tree-sitter parse trees; layouts the grammar parses with errors are rejected rather than guessed, which lowers yield.
- Labels have not yet been checked by an independent human reviewer (packages prepared, decisions pending).
