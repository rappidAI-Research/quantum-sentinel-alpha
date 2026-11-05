# Patch supervision and patch evaluation

Status: **implemented and tested offline; not trained, not evaluated on a model.** No record is
training-approved, so `train.patch.chat.jsonl` is empty.

## Why a separate task

The detection reply (`quickeval-review-*`) is a JSON finding; it never contained a patch, so the
upstream patches stored in Garden records taught nothing about remediation. Adding a patch to the
detection reply would change the QuickEval output contract and make every earlier result
incomparable. Patch generation is therefore its own task with its own prompt, parser and scorer;
detection, QuickEval and their scores are unchanged.

## Ground truth (`sentinel/garden/supervision.py`)

A vulnerable record is patch ground truth only if all of the following hold
(`patch_eligibility`; the reason is stored per record in the manifest):

1. the vulnerable record is accepted (every quality check, deduplication, contamination check and
   the audit);
2. it has an upstream patch, restricted to the function;
3. its post-fix sibling exists and is **accepted**: the audit did not find a later fix of the same
   function, an advisory calling the fix incomplete, an unreadable later fix commit or a manual
   exclusion, so rejected, incomplete and unverified fixes are never patch ground truth;
4. sibling and record come from the same repository and split;
5. the patch applies strictly (every context and removed line matches) and turns the vulnerable
   excerpt into the accepted fixed excerpt **exactly**;
6. the patch satisfies the patch output contract (one unified diff for the case file, consistent
   hunk counts).

On the current seed, 941 vulnerable records are eligible (train 709, validation 72,
holdout_reserved 96, temporal_quarantine 64); 106 are not because the audit rejected the fixed
version, and 159 because the vulnerable record itself was rejected. Every eligible reference patch
scores as well formed, applicable, syntactically valid, exact and on the labelled span under its own
scorer.

Training examples are bound to the committed manifest (`supervision.patch_example_sha256`) like
detection examples, and inherit the governance gate: a pair is exported only when both records are
training-approved (same repository, so they are approved together). The trainer uses them only with
`data.tasks: [detection, patch]`; the default is detection only.

## Task contract (`sentinel/eval/patch.py`)

- Prompt `sentinel-patch-v1`: language, file, context, the finding (CWE and lines) and the advisory
  summary, then the numbered vulnerable excerpt. The reference fix and provenance are never shown.
- Output: a single unified diff, `--- a/<file>` / `+++ b/<file>`, hunk line numbers of the excerpt.
  Reasoning tags and a Markdown fence are tolerated; anything else (prose, wrong file, inconsistent
  hunk counts, lines without a prefix) is rejected, never repaired.

## Evaluation (`score_patch`, `summarize`)

Implemented and tested, all offline:

| Metric | Meaning |
| --- | --- |
| `well_formed` | the reply satisfies the output contract |
| `applies` | strict application to the vulnerable excerpt |
| `syntax_ok` | the patched excerpt parses (Python `ast`; tree-sitter for Go/JS/TS) |
| `exact_match` / `normalized_match` | equal to the upstream fixed excerpt, byte for byte / after the Garden normalization |
| `touches_span` | a changed line lies within 2 lines of the labelled span (diffs of one change can align differently) |
| `changed_lines` vs `reference_changed_lines` | size relative to the upstream fix (a minimality signal) |

Evaluation cases: `holdout_reserved.patch_cases.jsonl` (96 cases, Git-ignored because it contains
upstream code), next to the detection holdout export.

## Boundary: what is not measured

`security_verified` is always `false`. A patch that applies, parses and even equals the upstream fix
is not shown to be secure, and a different patch that does not match the upstream fix may be
correct. Functional, security and regression tests would need, per case, a buildable checkout of
the upstream project at the vulnerable revision, its test suite, an exploit or security test and an
isolated environment to run them (the coding sandbox, docs/CODING_SANDBOX.md, isolates execution
but has no per-project environments). None of that exists for these 941 cases, so it is not
implemented; results must be reported as "applies / matches the upstream fix", never as "fixes the
vulnerability". No generation runner is wired up yet; scoring takes model replies from any backend.
