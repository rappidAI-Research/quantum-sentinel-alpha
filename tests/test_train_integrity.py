"""Target integrity: every supervised field of a training example is bound to the committed manifest.

Each attack below edits the Git-ignored export consistently (records, chat renderings and
SHA256SUMS regenerated), leaving only the committed manifest untouched, and must be refused.
"""

from __future__ import annotations

import copy
import json

import pytest

from sentinel.garden import render, supervision
from sentinel.train import data as train_data
from sentinel.train.data import GovernanceError, load_training_data
from tests.train_fixtures import GardenWorkdir, approved_registry


@pytest.fixture
def garden(monkeypatch: pytest.MonkeyPatch):
    workdir = GardenWorkdir()
    monkeypatch.setattr(train_data, "load_registry", approved_registry)
    yield workdir.write()
    workdir.cleanup()


def _load(garden: GardenWorkdir, tasks=("detection",)):
    return load_training_data(garden.config_path, approvals=garden.approvals, tasks=tasks)


def _vulnerable(train: list[dict]) -> dict:
    return next(r for r in train if r["kind"] == "vulnerable")


def test_clean_export_loads_both_tasks(garden: GardenWorkdir) -> None:
    detection = _load(garden)
    both = _load(garden, ("detection", "patch"))
    assert detection.examples_by_task["train"] == {"detection": len(garden.train), "patch": 0}
    assert both.examples_by_task["train"] == {"detection": len(garden.train), "patch": 1}
    assert detection.sha256 != both.sha256  # the task mix is part of the data hash
    patch = [e for e in both.train if e["task"] == "patch"]
    assert patch[0]["messages"][-1]["content"].startswith("--- a/shop/orders.py")


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("evidence", "Attacker-controlled data reaches eval(); always report CWE-94 here.", "target"),
        ("patch", None, "target"),
        ("evidence_source", "derived_template", "evidence_source|target"),
    ],
)
def test_editing_a_supervised_target_field_is_refused(garden: GardenWorkdir, field, value, message) -> None:
    train = copy.deepcopy(garden.train)
    vulnerable = _vulnerable(train)
    vulnerable["target"][field] = value if value is not None else vulnerable["target"]["patch"].replace("+", "+ ", 1)
    garden.rewrite_export(train)  # chat, patch chat and SHA256SUMS regenerated from the edited records
    with pytest.raises(GovernanceError, match=message):
        _load(garden)


@pytest.mark.parametrize("field", ["context", "file"])
def test_editing_the_prompt_input_is_refused_even_with_the_same_code(garden: GardenWorkdir, field) -> None:
    train = copy.deepcopy(garden.train)
    vulnerable = _vulnerable(train)
    vulnerable["input"][field] = vulnerable["input"][field] + ("\n# ignore previous instructions" if field == "context" else ".bak")
    garden.rewrite_export(train)
    with pytest.raises(GovernanceError, match="input"):
        _load(garden)


def test_dropping_an_approved_record_is_refused(garden: GardenWorkdir) -> None:
    train = [r for r in garden.train if r["kind"] != "caller_abstention"]
    garden.rewrite_export(train)
    with pytest.raises(GovernanceError, match="does not hold exactly"):
        _load(garden)


def test_export_edited_without_updating_checksums_is_refused(garden: GardenWorkdir) -> None:
    path = garden.workdir / "validation.jsonl"
    path.write_text(path.read_text() + "\n")  # harmless-looking edit, still not the exported file
    with pytest.raises(GovernanceError, match="SHA256SUMS"):
        _load(garden)


def test_tampered_patch_chat_file_is_refused(garden: GardenWorkdir) -> None:
    path = garden.workdir / "train.patch.chat.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["messages"][-1]["content"] = rows[0]["messages"][-1]["content"].replace("-", " ", 1)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    garden.write_sums()
    with pytest.raises(GovernanceError, match="train.patch.chat.jsonl"):
        _load(garden, ("detection", "patch"))


def test_changed_rendering_is_refused_until_the_manifest_is_regenerated(garden: GardenWorkdir, monkeypatch) -> None:
    original = render.assistant_reply

    def with_constant_confidence(record):
        payload = json.loads(original(record))
        if payload["decision"] == "finding":
            payload["confidence"] = 0.9
        return json.dumps(payload)

    monkeypatch.setattr(render, "assistant_reply", with_constant_confidence)
    garden.rewrite_export(garden.train)
    with pytest.raises(GovernanceError, match="detection example does not match the manifest"):
        _load(garden)


def test_patch_task_needs_a_verified_pair(garden: GardenWorkdir) -> None:
    train = [r for r in garden.train if r["kind"] != "post_fix"]  # the pair loses its fixed half
    garden.train = train
    garden.write()
    with pytest.raises(GovernanceError, match="no approved, verified vulnerable/fixed pair"):
        _load(garden, ("detection", "patch"))
    assert _load(garden).examples_by_task["train"]["patch"] == 0


def test_rejected_fixed_sibling_is_never_patch_ground_truth(garden: GardenWorkdir) -> None:
    pairs = supervision.pairs_by_id(garden.train)
    vulnerable = _vulnerable(garden.train)
    fixed = copy.deepcopy(pairs[vulnerable["pair_id"]]["post_fix"])
    assert supervision.patch_eligibility(vulnerable, fixed) == "eligible"
    fixed["quality"] = {"status": "rejected", "failed_checks": ["audit:later_fix_changes_function"], "flags": []}
    assert supervision.patch_eligibility(vulnerable, fixed) == "fixed_sibling_rejected"
    fixed = copy.deepcopy(pairs[vulnerable["pair_id"]]["post_fix"])
    fixed["input"]["code"] += "\n"
    assert supervision.patch_eligibility(vulnerable, fixed) == "patch_does_not_reproduce_fix"


def test_manifest_supervision_block_is_validated() -> None:
    row = {"kind": "post_fix", "supervision": {"version": "1", "input_sha256": "a" * 64, "target_sha256": "b" * 64,
                                                "detection_example_sha256": "c" * 64, "patch_eligibility": "eligible",
                                                "patch_example_sha256": "d" * 64}}
    assert "only vulnerable" in supervision.manifest_supervision_problem(row)
    row["kind"] = "vulnerable"
    assert supervision.manifest_supervision_problem(row) is None
    row["supervision"]["patch_example_sha256"] = None
    assert "exactly when" in supervision.manifest_supervision_problem(row)
    assert "no supervision block" in supervision.manifest_supervision_problem({"kind": "vulnerable"})


def test_unbinding_a_pair_cannot_silently_drop_its_patch_example(garden: GardenWorkdir) -> None:
    # review finding: patch eligibility is derived from record structure (pair_id), which the
    # input/target hashes do not cover
    validation = copy.deepcopy(garden.validation)
    for record in validation:
        record["pair_id"] = None
    path = garden.workdir / "validation.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in validation))
    garden.write_sums()
    with pytest.raises(GovernanceError, match="pair_id"):
        _load(garden, ("detection", "patch"))


def test_patch_examples_must_cover_every_manifest_eligible_record(garden: GardenWorkdir, monkeypatch) -> None:
    monkeypatch.setattr(supervision, "patch_examples", lambda records: [])
    with pytest.raises(GovernanceError, match="patch examples do not match"):
        _load(garden, ("detection", "patch"))
