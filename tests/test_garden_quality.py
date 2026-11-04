import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from sentinel.garden import code
from sentinel.garden.build import build_records
from sentinel.garden.quality import (
    EvalReference,
    QualitySettings,
    apply_unified_patch,
    check_record,
    contamination,
    deduplicate,
    load_eval_references,
    load_record_schema,
)
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = Draft202012Validator(load_record_schema())
QUALITY = QualitySettings(near_duplicate_jaccard=0.85, label_conflict_jaccard=0.92, contamination_jaccard=0.5, shingle_size=5)


def records(repo: str = "example/webapp", advisory: str = "GHSA-2222-3333-4444", old: str = PY_OLD, new: str = PY_NEW) -> list[dict]:
    return build_records(
        candidate(repo=repo, advisory=advisory),
        FakeRepo({"shop/orders.py": old}, {"shop/orders.py": new}),
        SETTINGS,
        "sentinel-garden-test",
        "ghsa-osv-fix-commits",
        ["osv"],
    )


def failed(record: dict) -> list[str]:
    return check_record(record, VALIDATOR, {"CWE-1395"}, {"CWE-20"}, 1000, 3)[0]


def rehash(record: dict) -> dict:
    record["provenance"]["content_sha256"] = hashlib.sha256(record["input"]["code"].encode()).hexdigest()
    record["provenance"]["normalized_sha256"] = code.normalized_sha256(record["input"]["language"], record["input"]["code"])
    return record


def test_tampered_code_fails_hash_check() -> None:
    record = copy.deepcopy(records()[0])
    record["input"]["code"] += "# tampered\n"
    assert "content_hash" in failed(record)


def test_span_outside_excerpt_and_trivial_span_fail() -> None:
    vulnerable = next(r for r in records() if r["kind"] == "vulnerable")
    outside = copy.deepcopy(vulnerable)
    outside["target"]["end_line"] = 99
    assert "localization_valid:span" in failed(outside)
    trivial = copy.deepcopy(vulnerable)
    trivial["input"]["code"] = "def list_orders(conn, status):\n    # note\n    return 1\n"
    trivial["target"]["start_line"] = trivial["target"]["end_line"] = 2
    assert "localization_valid:trivial_span" in failed(rehash(trivial))


def test_rejected_cwe_and_inconsistent_labels_fail() -> None:
    vulnerable = copy.deepcopy(next(r for r in records() if r["kind"] == "vulnerable"))
    vulnerable["target"]["cwe"] = "CWE-1395"
    assert "cwe_valid" in failed(vulnerable)
    fixed = copy.deepcopy(next(r for r in records() if r["kind"] == "post_fix"))
    fixed["target"]["cwe"] = "CWE-89"
    assert "localization_absent" in failed(fixed)
    fixed["target"]["decision"] = "finding"
    assert "kind_decision_consistent" in failed(fixed)


def test_secret_in_excerpt_rejects_record() -> None:
    record = copy.deepcopy(records()[0])
    record["input"]["code"] = record["input"]["code"].replace("SELECT", 'SELECT /* AKIAABCDEFGHIJKLMNOP */')
    assert "no_secrets:aws_access_key" in failed(rehash(record))


def test_test_files_are_not_accepted() -> None:
    record = copy.deepcopy(records()[0])
    record["input"]["file"] = record["provenance"]["file_path"] = "tests/test_orders.py"
    assert "non_test_source_file" in failed(record)


def test_patch_application_is_strict() -> None:
    assert apply_unified_patch("a\nb\nc\n", "--- a/x\n+++ b/x\n@@ -2,1 +2,1 @@\n-b\n+B\n") == "a\nB\nc\n"
    assert apply_unified_patch("a\nX\nc\n", "--- a/x\n+++ b/x\n@@ -2,1 +2,1 @@\n-b\n+B\n") is None
    assert apply_unified_patch("a\n", "--- a/x\n+++ b/x\n@@ -1,0 +2,1 @@\n+b\n") == "a\nb\n"


def test_exact_duplicates_across_advisories_keep_the_first() -> None:
    first = records(repo="example/webapp", advisory="GHSA-2222-3333-4444")
    fork = records(repo="fork/webapp", advisory="GHSA-5555-6666-7777")
    result = deduplicate(first + fork, QUALITY)
    rejected = {rid: reason for rid, (reason, _) in result.items()}
    kept_ids = {r["record_id"] for r in first + fork} - set(rejected)
    assert len(kept_ids) == 3 and set(rejected.values()) == {"exact_duplicate"}


def test_near_identical_code_with_opposite_labels_is_a_conflict() -> None:
    # Advisory B: the function that advisory A "fixed" is reported vulnerable again (incomplete fix).
    a = records(advisory="GHSA-2222-3333-4444")
    second_fix = PY_NEW.replace('"SELECT id FROM orders WHERE status = ?"', '"SELECT id FROM orders WHERE status = ? LIMIT 100"')
    b = records(repo="example/webapp2", advisory="GHSA-5555-6666-7777", old=PY_NEW, new=second_fix)
    result = deduplicate(a + b, QUALITY)
    a_fixed = next(r for r in a if r["kind"] == "post_fix")["record_id"]
    assert result[a_fixed][0] == "label_conflict_near"


def test_pairs_of_one_fix_are_not_duplicates_of_each_other() -> None:
    assert deduplicate(records(), QUALITY) == {}


def test_contamination_against_eval_cases() -> None:
    built = records()
    vulnerable = next(r for r in built if r["kind"] == "vulnerable")
    exact = [EvalReference("qe-x", "python", vulnerable["input"]["code"], None)]
    assert contamination([vulnerable], exact, QUALITY)[vulnerable["record_id"]].startswith("eval_contamination_exact")
    same_repo = [EvalReference("holdout-1", "python", "x = 1\n", "https://github.com/Example/WebApp")]
    assert contamination([vulnerable], same_repo, QUALITY)[vulnerable["record_id"]].startswith("eval_repository_overlap")
    embedded = [EvalReference("qe-y", "python", vulnerable["input"]["code"].replace("def list_orders", "def other"), None)]
    assert vulnerable["record_id"] in contamination([vulnerable], embedded, QUALITY)


def test_real_eval_sets_are_loaded_for_contamination_checks() -> None:
    references = load_eval_references([ROOT / "eval" / "sentinelbench"], [ROOT / "eval" / "coding" / "coding_v1.jsonl"])
    ids = {r.reference_id for r in references}
    assert sum(1 for line in (ROOT / "eval/sentinelbench/quickeval_v1.jsonl").read_text().splitlines() if line) <= len(ids)
    assert any(i.startswith("coding:") for i in ids)
    # a QuickEval case copied into a training record is caught
    case = json.loads((ROOT / "eval/sentinelbench/quickeval_v1.jsonl").read_text().splitlines()[0])
    record = copy.deepcopy(records()[0])
    record["input"]["code"] = case["code"]
    record["input"]["language"] = case["language"]
    rehash(record)
    assert record["record_id"] in contamination([record], references, QUALITY)
