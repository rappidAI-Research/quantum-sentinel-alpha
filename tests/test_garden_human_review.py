"""Independent human review: blind, reproducible packages; decisions kept apart and bound to packages."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from sentinel.garden import human_review
from tests.train_fixtures import _records


def population() -> list[dict]:
    records = []
    for index, (repo, language_suffix) in enumerate((("example/webapp", "a"), ("example/other", "b"), ("example/third", "c"))):
        for record in _records(repo, f"GHSA-2222-3333-44{index}{index}", "train" if index < 2 else "holdout_reserved"):
            records.append(record)
    return records


def test_sample_covers_only_the_exportable_population_and_is_deterministic() -> None:
    records = population()
    sample = human_review.draw_sample(records)
    assert sample == human_review.draw_sample(list(reversed(records)))
    assert {item["split"] for item in sample} <= {"train", "validation"}
    assert {item["item"] for item in sample} == {"pair", "abstention"}
    pair = next(i for i in sample if i["item"] == "pair")
    assert [r.split("-")[2] for r in pair["record_ids"]] == ["vulnerable", "post"]


def test_package_is_blind_complete_and_reproducible(tmp_path: Path) -> None:
    records = population()
    by_id = {r["record_id"]: r for r in records}
    manual = {"reviewer": "AI engineer", "items": [{"group_id": records[0]["group_id"], "item": "pair", "localization": "precise"}]}
    automated = [{"group_id": records[0]["group_id"], "span": {"broad": False}}]
    index = human_review.write_packages(records, tmp_path / "osv", automated, manual, tmp_path / "packages", tmp_path / "audit")
    again = human_review.write_packages(records, tmp_path / "osv", automated, manual, tmp_path / "packages2", tmp_path / "audit2")
    assert [i["package_sha256"] for i in index] == [i["package_sha256"] for i in again]
    item = next(i for i in index if i["item"] == "pair")
    package = (tmp_path / "packages" / item["item_id"] / "package.md").read_text()
    assert hashlib.sha256(package.encode()).hexdigest() == item["package_sha256"]
    vulnerable = by_id[item["record_ids"][0]]
    for expected in (
        f"https://github.com/advisories/{vulnerable['provenance']['advisory_id']}",
        f"/commit/{vulnerable['provenance']['fixed_revision']}",
        vulnerable["target"]["cwe"],
        "## Upstream patch",
        "## Fixed excerpt",
        "`hard_negative_safe`",
        "`cwe`",
        ">>",
    ):
        assert expected in package
    assert "AI engineer" not in package and "precise" not in package.split("## Checklist")[0]  # blind
    signals = (tmp_path / "packages" / item["item_id"] / "signals.json").read_text()
    assert "AI engineer" in signals
    template = yaml.safe_load((tmp_path / "audit" / "human_review.TEMPLATE.yaml").read_text())
    assert template["reviewer"]["name"] is None and all(v is None for v in template["items"][0].values() if not isinstance(v, (str, list)))


def _decisions(index: list[dict], **overrides) -> dict:
    item = next(i for i in index if i["item"] == "pair")
    entry = {
        "item_id": item["item_id"],
        "package_sha256": item["package_sha256"],
        "advisory_matches_code": "yes",
        "fix_addresses_weakness": "yes",
        "label_correct": "yes",
        "cwe": "defensible",
        "localization": "imprecise",
        "hard_negative_safe": "no",
        "pair_consistent": "yes",
        "leakage_or_duplicate": "none_seen",
        "exclude_records": [item["record_ids"][1]],
        "notes": "post-fix still vulnerable",
        **overrides,
    }
    return {
        "sample_id": human_review.SAMPLE_ID,
        "reviewer": {"name": "R. Reviewer", "affiliation": "external", "independent_of_implementation": True, "conflicts": "none"},
        "reviewed_at": "2026-11-10",
        "items": [entry],
    }


def test_decisions_are_validated_bound_to_the_package_and_summarised_separately(tmp_path: Path) -> None:
    index = human_review.write_packages(population(), tmp_path / "osv", [], {"items": []}, tmp_path / "p", tmp_path / "a")
    path = tmp_path / "human_review.yaml"

    assert human_review.summarize_human(human_review.load_human_review(path, index), index)["status"] == "not_performed"

    path.write_text(yaml.safe_dump(_decisions(index)))
    review = human_review.load_human_review(path, index)
    summary = human_review.summarize_human(review, index)
    assert summary["status"] == "independent" and summary["reviewed_items"] == 1
    assert summary["rates"]["hard_negative_not_safe"]["errors"] == 1
    assert summary["rates"]["localization_imprecise_or_wrong"]["errors"] == 1
    assert human_review.human_exclusions(review)[0]["reason"].startswith("human_review:")

    path.write_text(yaml.safe_dump(_decisions(index, package_sha256="0" * 64)))
    stale = human_review.load_human_review(path, index)
    assert stale["items"] == [] and len(stale["stale"]) == 1  # decided on another package version

    for bad in ({"cwe": "fine"}, {"exclude_records": ["sg-py-vulnerable-0000000000000000"]}):
        path.write_text(yaml.safe_dump(_decisions(index, **bad)))
        with pytest.raises(ValueError):
            human_review.load_human_review(path, index)

    document = _decisions(index)
    document["reviewer"]["name"] = None
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="reviewer"):
        human_review.load_human_review(path, index)


def test_an_unfinished_entry_is_not_counted(tmp_path: Path) -> None:
    index = human_review.write_packages(population(), tmp_path / "osv", [], {"items": []}, tmp_path / "p", tmp_path / "a")
    path = tmp_path / "human_review.yaml"
    path.write_text(yaml.safe_dump(_decisions(index, hard_negative_safe=None)))
    assert human_review.load_human_review(path, index)["items"] == []


def test_refreshing_the_sample_keeps_earlier_decisions_and_their_exclusions(tmp_path: Path) -> None:
    # review finding: an exclusion removes the item from the next draw; the decision must survive
    records = population()
    audit_dir = tmp_path / "a"
    index = human_review.write_packages(records, tmp_path / "osv", [], {"items": []}, tmp_path / "p", audit_dir)
    path = tmp_path / "human_review.yaml"
    path.write_text(yaml.safe_dump(_decisions(index)))
    decided = _decisions(index)["items"][0]

    excluded = decided["exclude_records"][0]
    for record in records:
        if record["record_id"] == excluded:
            record["quality"] = {"status": "rejected", "failed_checks": ["audit:human_review:x"], "flags": []}
    refreshed = human_review.write_packages(records, tmp_path / "osv", [], {"items": []}, tmp_path / "p", audit_dir)
    retired = [i for i in refreshed if i.get("retired")]
    assert [i["item_id"] for i in retired] == [decided["item_id"]]

    review = human_review.load_human_review(path, refreshed)  # no "unknown item"
    assert review["items"] == [] and [e["item_id"] for e in review["retired"]] == [decided["item_id"]]
    assert human_review.human_exclusions(review)[0]["record_id"] == excluded
    summary = human_review.summarize_human(review, refreshed)
    assert summary["retired_decisions"] == [decided["item_id"]] and summary["sample_items"] == len(refreshed) - 1
    template = yaml.safe_load((audit_dir / "human_review.TEMPLATE.yaml").read_text())
    assert decided["item_id"] not in {i["item_id"] for i in template["items"]}
