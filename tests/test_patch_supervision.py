"""Patch supervision: verified ground truth, strict output contract, offline scoring boundaries."""

from __future__ import annotations

import json
import re

import pytest

from sentinel.eval.patch import PATCH_PROMPT_VERSION, PatchError, build_patch_messages, parse_patch, score_patch, summarize
from sentinel.garden import supervision
from tests.train_fixtures import _records


@pytest.fixture
def case():
    records = _records("example/webapp", "GHSA-2222-3333-4444", "holdout_reserved")
    cases = supervision.patch_cases(records)
    assert len(cases) == 1
    return cases[0]


def test_patch_case_comes_from_a_verified_pair(case) -> None:
    assert case["task"] == "patch" and case["finding"]["cwe"]
    assert case["provenance"]["fixed_record_id"].startswith("sg-py-post-fix-")
    from sentinel.garden.quality import apply_unified_patch

    assert apply_unified_patch(case["code"], case["reference_patch"]) == case["reference_fixed_code"]


def test_prompt_never_contains_the_reference(case) -> None:
    messages = build_patch_messages(case)
    text = json.dumps(messages)
    assert case["reference_patch"] not in text and "reference" not in text.lower()
    added = [line[1:] for line in case["reference_patch"].splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert not any(line.strip() and line in case["code"] for line in added if len(line.strip()) > 12)
    assert case["advisory_summary"] in messages[1]["content"]


def test_reference_scores_as_exact_and_never_as_secure(case) -> None:
    score = score_patch(case, case["reference_patch"])
    assert (score.well_formed, score.applies, score.syntax_ok, score.exact_match, score.normalized_match, score.touches_span) == (True,) * 6
    assert score.security_verified is False
    summary = summarize([score])
    assert summary["security_verified"] is False and "security tests" in " ".join(summary["not_measured"])


def test_fenced_and_reasoned_replies_are_accepted(case) -> None:
    reply = f"<think>plan</think>\n```diff\n{case['reference_patch']}```"
    assert score_patch(case, reply).exact_match


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: "Here is the fix:\n" + p, "must start"),
        (lambda p: p.replace("--- a/shop/orders.py", "--- a/other.py"), "name the case file"),
        (lambda p: re.sub(r"^@@ -(\d+),(\d+)", lambda m: f"@@ -{m[1]},{int(m[2]) + 1}", p, count=1, flags=re.M), "old and"),
        (lambda p: "\n".join(line if not line.startswith(" ") else line[1:] for line in p.splitlines()), "prefix"),
        (lambda p: p.split("@@")[0] + "context\n", "no hunk"),
    ],
)
def test_malformed_patches_are_rejected_not_repaired(case, mutate, message) -> None:
    with pytest.raises(PatchError, match=message):
        parse_patch(mutate(case["reference_patch"]), case["file"])
    score = score_patch(case, mutate(case["reference_patch"]))
    assert not score.well_formed and not score.applies and score.error


def test_a_patch_that_does_not_apply_is_scored_as_such(case) -> None:
    lines = case["reference_patch"].splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(" ") and line.strip())
    lines[index] = " " + lines[index][1:] + "  # drifted"
    score = score_patch(case, "\n".join(lines) + "\n")
    assert score.well_formed and not score.applies and not score.exact_match


def test_a_different_valid_patch_applies_but_is_not_exact(case) -> None:
    lines = case["code"].splitlines()
    n = len(lines)
    alternative = (
        f"--- a/{case['file']}\n+++ b/{case['file']}\n@@ -{n},1 +{n},2 @@\n {lines[-1]}\n+    # reviewed\n"
    )
    score = score_patch(case, alternative)
    assert score.applies and score.syntax_ok and not score.exact_match and score.changed_lines == 1


def test_patch_training_example_contract(case) -> None:
    example = supervision.patch_chat_example(case)
    assert example["prompt_version"] == PATCH_PROMPT_VERSION and example["task"] == "patch"
    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert score_patch(case, example["messages"][-1]["content"]).exact_match


def test_patch_eval_runner_with_the_reference_replay(case, tmp_path) -> None:
    from sentinel.eval import patch as patch_eval

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(case) + "\n")
    out = tmp_path / "patch-run.json"
    assert patch_eval.main(["--cases", str(cases_path), "--output", str(out), "--backend", "reference"]) == 0
    result = json.loads(out.read_text())
    assert result["applies_rate"] == result["exact_match_rate"] == 1.0
    assert result["security_verified"] is False and result["contract"]["prompt_version"] == "sentinel-patch-v1"


@pytest.mark.parametrize(
    "hunk, message",
    [
        ("@@ -999,0 +999,1 @@\n+    pass\n", "outside the excerpt"),  # review finding: appended past EOF
        ("@@ -1,0 +5,1 @@\n+    pass\n", "new start"),
    ],
)
def test_hunk_coordinates_must_fit_the_excerpt(case, hunk, message) -> None:
    from sentinel.eval.patch import check_coordinates

    reply = f"--- a/{case['file']}\n+++ b/{case['file']}\n{hunk}"
    _, hunks = parse_patch(reply, case["file"])
    with pytest.raises(PatchError, match=message):
        check_coordinates(hunks, len(case["code"].splitlines()))
    score = score_patch(case, reply)
    assert not score.applies and not score.exact_match and not score.well_formed


def test_overlapping_hunks_are_rejected(case) -> None:
    from sentinel.eval.patch import check_coordinates

    lines = case["code"].splitlines()
    reply = (
        f"--- a/{case['file']}\n+++ b/{case['file']}\n"
        f"@@ -1,1 +1,1 @@\n-{lines[0]}\n+{lines[0]}  # a\n"
        f"@@ -1,1 +1,1 @@\n-{lines[0]}\n+{lines[0]}  # b\n"
    )
    _, hunks = parse_patch(reply, case["file"])
    with pytest.raises(PatchError, match="overlap"):
        check_coordinates(hunks, len(lines))


@pytest.mark.parametrize(
    "language, original, broken",
    [
        ("python", "def f(x):\n    return x\n", "def f(x):\n    return +\n"),  # review finding: bracket fallback passed this
        ("python", "x = 1\n", "x =\n"),
        ("javascript", "function f(x) {\n  return x;\n}\n", "function f(x) {\n  return +;\n}\n"),
        ("go", "func f(x int) int {\n\treturn x\n}\n", "func f(x int) int {\n\treturn +\n}\n"),
    ],
)
def test_syntax_metric_is_strict(language, original, broken) -> None:
    from sentinel.eval.patch import syntax_errors

    assert syntax_errors(language, original) == 0 and syntax_errors(language, broken) > 0
    last = original.splitlines()
    new = broken.splitlines()
    changed = next(i for i, (a, b) in enumerate(zip(last, new)) if a != b)
    case = {
        "case_id": "c",
        "language": language,
        "file": "a.x",
        "code": original,
        "finding": {"cwe": "CWE-20", "start_line": changed + 1, "end_line": changed + 1},
        "reference_patch": f"--- a/a.x\n+++ b/a.x\n@@ -{changed + 1},1 +{changed + 1},1 @@\n-{last[changed]}\n+{last[changed]}\n".replace(f"+{last[changed]}\n", f"+{last[changed]} \n"),
        "reference_fixed_code": original,
    }
    reply = f"--- a/a.x\n+++ b/a.x\n@@ -{changed + 1},1 +{changed + 1},1 @@\n-{last[changed]}\n+{new[changed]}\n"
    score = score_patch(case, reply)
    assert score.applies and not score.syntax_ok
