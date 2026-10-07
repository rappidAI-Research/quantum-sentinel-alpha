import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from sentinel.eval.io import assert_publicly_committable, load_jsonl, validate_cases
from sentinel.foundation import public_case_files

ROOT = Path(__file__).resolve().parents[1]
QUICKEVAL = ROOT / "eval" / "sentinelbench" / "quickeval_v1.jsonl"
PUBLIC_CASE_FILES = public_case_files()


def quickeval() -> list[dict]:
    return load_jsonl(QUICKEVAL)


def test_quickeval_is_valid_and_covers_all_case_groups() -> None:
    cases = quickeval()
    validate_cases(cases)
    types = Counter(case["case_type"] for case in cases)
    assert 50 <= len(cases) <= 100
    assert types["vulnerable"] >= 25 and types["hard_negative"] >= 25 and types["abstention"] >= 10
    languages = Counter(case["language"] for case in cases)
    assert languages["python"] >= 15 and languages["go"] >= 15
    assert languages["javascript"] + languages["typescript"] >= 15


def test_every_quickeval_pair_is_complete() -> None:
    pairs = Counter(case["pair_id"] for case in quickeval() if "pair_id" in case)
    assert pairs and set(pairs.values()) == {2}


def test_committed_case_files_are_clean_room_synthetic() -> None:
    assert PUBLIC_CASE_FILES
    for path in PUBLIC_CASE_FILES:
        assert_publicly_committable(load_jsonl(path))


def test_duplicate_case_id_is_rejected() -> None:
    cases = quickeval()[:2]
    cases[1] = {**cases[1], "case_id": cases[0]["case_id"]}
    with pytest.raises(ValueError, match="duplicate case_id"):
        validate_cases(cases)


def test_case_type_must_match_expected_decision() -> None:
    case = deepcopy(next(c for c in quickeval() if c["case_type"] == "hard_negative"))
    case["case_type"] = "abstention"
    with pytest.raises(ValueError, match="contradicts"):
        validate_cases([case])


def test_ground_truth_span_must_lie_inside_the_code() -> None:
    case = deepcopy(next(c for c in quickeval() if c["case_type"] == "vulnerable"))
    case["ground_truth"]["end_line"] = len(case["code"].splitlines()) + 1
    with pytest.raises(ValueError, match="span"):
        validate_cases([case])


def test_edited_code_without_updated_hash_is_rejected() -> None:
    case = deepcopy(quickeval()[0])
    case["code"] += "# tampered\n"
    with pytest.raises(ValueError, match="content_sha256"):
        validate_cases([case])


def test_non_synthetic_case_needs_upstream_provenance() -> None:
    case = deepcopy(quickeval()[0])
    case["source_kind"] = "private_holdout"
    with pytest.raises(ValueError, match="disagree"):
        validate_cases([case])
    case["provenance"] = {
        "origin": "upstream_derived",
        "content_sha256": hashlib.sha256(case["code"].encode()).hexdigest(),
    }
    with pytest.raises(ValueError, match="upstream provenance"):
        validate_cases([case])
    with pytest.raises(ValueError, match="public repository"):
        assert_publicly_committable([case])
