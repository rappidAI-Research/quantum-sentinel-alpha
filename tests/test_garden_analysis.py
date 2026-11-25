"""Weakness analysis: composition, shortcut probes, development-run errors, cross-version contamination."""

from __future__ import annotations

import copy

from sentinel.garden import analysis
from tests.train_fixtures import GardenWorkdir


def _records() -> list[dict]:
    garden = GardenWorkdir()
    try:
        records = copy.deepcopy(garden.train + garden.validation)
    finally:
        garden.cleanup()
    for record in records:
        record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
    return records


def test_composition_and_probes_report_numbers_only() -> None:
    records = _records()
    result = analysis.analyze(records)
    assert result["accepted"]["records"] == len(records)
    assert set(result["accepted"]["by_decision"]) <= {"finding", "no_finding", "abstain"}
    text = repr(result)
    assert not any(r["input"]["code"].strip().splitlines()[1] in text for r in records)  # no code leaves the analysis
    assert "not available" in result["development_runs"]


def test_the_v3_abstention_sentence_is_measured_as_a_shortcut() -> None:
    records = _records()
    abstentions = [r for r in records if r["target"]["decision"] == "abstain"]
    for record in abstentions:
        record["input"]["context"] += f" `{record['localization']['withheld_callee']}` is defined elsewhere; its body is not included."
    train = [r for r in records if r["split"] == "train"]
    probe = analysis.shortcut_probes(train, train)["abstention_phrase"]["test"]
    assert probe["abstain_share_among_them"] == 1.0 and probe["abstain_recall_of_rule"] == 1.0


def test_naive_bayes_ranks_tokens_seen_with_a_label() -> None:
    model = analysis.NaiveBayes([(["escape", "html"], "no_finding"), (["innerhtml", "user"], "finding")])
    assert model.predict(["escape"]) == "no_finding" and model.predict(["innerhtml"]) == "finding"


def test_run_errors_split_missed_findings_into_no_finding_and_abstain() -> None:
    cases = {
        "a": {"language": "python", "ground_truth": {"cwe": "CWE-89"}},
        "b": {"language": "python", "ground_truth": {"cwe": "CWE-89"}},
        "c": {"language": "go", "ground_truth": {"cwe": None}},
    }
    artifact = {
        "model": {"id": "m", "adapter": None},
        "predictions": [
            {"case_id": "a", "expected_decision": "finding", "prediction": {"decision": "abstain"}},
            {"case_id": "b", "expected_decision": "finding", "prediction": {"decision": "finding"}},
            {"case_id": "c", "expected_decision": "no_finding", "prediction": {"decision": "finding"}},
        ],
    }
    errors = analysis.run_errors(artifact, cases)
    assert errors["confusion_expected_to_predicted"] == {"finding->abstain": 1, "finding->finding": 1, "no_finding->finding": 1}
    assert errors["recall_by_family"]["sql_injection"] == {"findings": 2, "detected": 1, "abstained": 1, "recall": 0.5}
    assert errors["by_language"]["go"]["false_positives"] == 1


def test_contamination_report_counts_shared_identities() -> None:
    records = _records()
    for record in records:
        record["evaluation_role"] = "training" if record["split"] == "train" else "development"
    clean = analysis.contamination_report(records, {}, {})
    assert clean["training_records"] == sum(1 for r in records if r["split"] == "train")
    # the fixtures share one CVE alias across the two repositories: reported, not hidden
    assert clean["training_records_sharing"].get("advisory_or_alias", 0) == clean["training_records"]
