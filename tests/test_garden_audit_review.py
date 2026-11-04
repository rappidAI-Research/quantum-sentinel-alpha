import copy
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from sentinel.garden import audit, review
from sentinel.garden.build import build_records
from sentinel.garden.gitsource import CommitInfo, GitError
from sentinel.garden.governance import load_approvals
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate, git_diff


def built() -> dict[str, dict]:
    records = build_records(candidate(), FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}), SETTINGS, "t", "ghsa-osv-fix-commits", ["osv"])
    out = {}
    for record in records:
        record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
        record["split"] = "train"
        out[record["kind"]] = record
    return out


# --- automated audit checks -----------------------------------------------------------------------


def test_correspondence_uses_advisory_text_and_commit_message() -> None:
    vulnerable = built()["vulnerable"]
    strong = audit.correspondence(vulnerable, "SQL injection in list_orders via the status parameter", "Fix GHSA-2222-3333-4444")
    assert strong["strength"] == "strong"
    assert {"function_named_in_advisory", "commit_message_cites_advisory"} <= set(strong["signals"])
    weak = audit.correspondence(vulnerable, "Unspecified issue", "refactor")
    assert weak["strength"] == "weak" and weak["signals"] == []


def test_fix_pattern_detects_parameterized_query() -> None:
    result = audit.fix_pattern(built()["vulnerable"])
    assert result["family"] == "sql_injection" and result["consistent"] is True


def test_constant_argument_abstention_is_invalid() -> None:
    group = built()
    ok = audit.abstention_validity(group["caller_abstention"], group["vulnerable"])
    assert ok["problems"] == []
    constant = copy.deepcopy(group["caller_abstention"])
    constant["input"]["code"] = 'def orders_view(conn):\n    x = 1\n    return {"orders": list_orders("db", "open")}\n'
    constant["localization"]["call_line"] = 3
    assert audit.abstention_validity(constant, group["vulnerable"])["problems"] == ["constant_arguments"]


class LaterFixRepo:
    """Upstream with a second, later fix that changes the same function again."""

    def __init__(self) -> None:
        third = PY_NEW.replace("WHERE status = ?", "WHERE status = ? AND visible = 1")
        self.files = {"p2": PY_NEW, "c2": third}

    def ensure_commit(self, sha: str) -> CommitInfo:
        if sha != "c" * 40:
            raise GitError("unknown")
        return CommitInfo("c" * 40, ("p2",), "2025-06-01T00:00:00+00:00", "2025-06-01T00:00:00+00:00")

    def changed_files(self, parent: str, sha: str) -> list[tuple[str, str]]:
        return [("M", "shop/orders.py")]

    def text(self, revision: str, path: str) -> str:
        return self.files["p2" if revision == "p2" else "c2"]

    def file_diff(self, parent: str, sha: str, path: str) -> str:
        return git_diff(self.files["p2"], self.files["c2"], path)


def test_later_fix_of_the_same_function_invalidates_the_hard_negative() -> None:
    group = built()
    later = {
        "id": "GHSA-6666-7777-8888",
        "summary": "Second SQL injection in webapp",
        "references": [{"type": "FIX", "url": f"https://github.com/example/webapp/commit/{'c' * 40}"}],
    }
    found = audit.later_fixes(group, [later], LaterFixRepo())
    assert [(f.advisory_id, f.reason) for f in found] == [("GHSA-6666-7777-8888", "later_fix_changes_same_function")]


def test_incomplete_fix_report_invalidates_the_hard_negative() -> None:
    group = built()
    report = {"id": "GHSA-9999-cccc-ffff", "details": "The fix for CVE-2025-0001 was incomplete and can be bypassed."}
    found = audit.later_fixes(group, [report], LaterFixRepo())
    assert [f.reason for f in found] == ["reported_incomplete_fix"]


def test_same_advisory_alias_in_two_splits_excludes_the_train_side() -> None:
    a = built()["vulnerable"]
    b = copy.deepcopy(a)
    b["record_id"] = "sg-py-vulnerable-ffffffffffffffff"
    b["split"] = "validation"
    b["input"]["code"] = "def other(x):\n    return x\n\n"
    b["provenance"]["advisory_id"] = "GHSA-aaaa-bbbb-cccc"
    b["provenance"]["fixed_revision"] = "d" * 40
    leaks = audit.split_leakage([a, b])
    assert leaks == {a["record_id"]: "shared_advisory_with_validation"}


def test_audit_population_ignores_earlier_audit_exclusions() -> None:
    record = built()["vulnerable"]
    record["quality"] = {"status": "rejected", "failed_checks": ["audit:split_leakage_x"], "flags": []}
    assert audit.audit_population([record]) == [record]
    record["quality"]["failed_checks"].append("exact_duplicate")
    assert audit.audit_population([record]) == []


def test_stratified_sample_is_deterministic_and_covers_languages() -> None:
    groups = {}
    for language, n in (("python", 5), ("go", 5)):
        for i in range(n):
            vulnerable = {"input": {"language": language}, "target": {"cwe": ["CWE-79", "CWE-22", "CWE-78"][i % 3]}}
            groups[f"{language}-{i}"] = {"vulnerable": vulnerable, "post_fix": {}, **({"caller_abstention": {}} if i < 2 else {})}
    first = audit.stratified_sample(groups, 3, 1, "s")
    assert first == audit.stratified_sample(groups, 3, 1, "s")
    assert sum(1 for s in first if s["language"] == "go" and s["item"] == "pair") == 3
    families = {audit.family_of(groups[s["group_id"]]["vulnerable"]["target"]["cwe"]) for s in first if s["language"] == "python" and s["item"] == "pair"}
    assert len(families) == 3  # round-robin over CWE families


def test_wilson_interval() -> None:
    low, high = audit.wilson_interval(0, 30)
    assert low == 0.0 and 0.1 < high < 0.13
    assert audit.wilson_interval(3, 10)[0] < 0.3 < audit.wilson_interval(3, 10)[1]


def test_manual_review_values_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump({"reviewer": "r", "items": [{"group_id": "g", "localization": "great"}]}))
    with pytest.raises(ValueError, match="localization"):
        audit.load_manual_review(path)


# --- rights review --------------------------------------------------------------------------------


def evidence(**overrides) -> dict:
    base = {
        "license_detected": "MIT",
        "license_category": "permissive",
        "license_files_identical_across_revisions": True,
        "registry_agreement": "agrees",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("overrides", "tier"),
    [
        ({}, "A"),
        ({"registry_agreement": "unavailable"}, "B"),
        ({"license_files_identical_across_revisions": False}, "B"),
        ({"license_detected": "Unlicense"}, "B"),
        ({"license_detected": "multiple:Apache-2.0|MIT"}, "B"),
        ({"license_detected": "MPL-2.0", "license_category": "weak_copyleft"}, "C"),
        ({"license_detected": "AGPL-3.0", "license_category": "strong_copyleft"}, "D"),
        ({"license_detected": "none", "license_category": "none"}, "D"),
        ({"license_detected": "multiple:GPL-3.0|MIT", "license_category": "unknown"}, "D"),
    ],
)
def test_review_tiers(overrides: dict, tier: str) -> None:
    assert review.tier_of(evidence(**overrides))[0] == tier


def test_registry_agreement() -> None:
    assert review.registry_agreement("MIT", {"npm:a": ["MIT"]}) == "agrees"
    assert review.registry_agreement("Apache-2.0", {"pypi:a": ["MIT"]}) == "disagrees"
    assert review.registry_agreement("MIT", {"pypi:a": None}) == "unavailable"
    assert review.registry_agreement("multiple:Apache-2.0|MIT", {"go:a": ["Apache-2.0"]}) == "agrees"
    assert review.registry_agreement("GPL-3.0", {"npm:a": ["LGPL-3.0"]}) == "disagrees"


def test_depsdev_lookup_is_cached(tmp_path: Path) -> None:
    calls = []

    def fetch(url: str):
        calls.append(url)
        if url.endswith("/versions/2.0.0"):
            return {"licenses": ["MIT"]}
        return {"versions": [{"versionKey": {"version": "1.0.0"}, "isDefault": False}, {"versionKey": {"version": "2.0.0"}, "isDefault": True}]}

    client = review.DepsDev(tmp_path, fetch=fetch)
    assert client.licenses("npm", "@scope/pkg") == ["MIT"]
    assert client.licenses("npm", "@scope/pkg") == ["MIT"]
    assert len(calls) == 2 and "%40scope%2Fpkg" in calls[0]


def test_approval_templates_cannot_be_loaded_until_completed(tmp_path: Path) -> None:
    row = {
        **evidence(),
        "repository_key": "example/webapp",
        "repository": "https://github.com/example/webapp",
        "license_files_latest": [{"path": "LICENSE", "sha256": "a" * 64}],
        "latest_fixed_revision": "b" * 40,
        "registry_licenses": {"PyPI:webapp": ["MIT"]},
        "records": {"train/vulnerable": 1},
        "record_count": 1,
        "advisories": ["GHSA-2222-3333-4444"],
        "tier": "A",
        "tier_reasons": [],
    }
    row["evidence_urls"] = review._evidence_urls(row)
    path = tmp_path / "approvals.yaml"
    path.write_text("approvals_version: '1.0'\n" + "repositories:\n" + review.approval_template(row) + "\n")
    with pytest.raises(ValidationError):
        load_approvals(path)


def test_manual_wrong_verdicts_become_exclusions_and_report_renders() -> None:
    manual = {
        "reviewer": "test",
        "items": [
            {"group_id": "g1", "item": "pair", "cwe": "wrong", "localization": "precise", "exclude_records": ["r1", "r2"], "notes": "CWE does not fit"},
            {"group_id": "g2", "item": "pair", "cwe": "defensible", "localization": "acceptable"},
        ],
    }
    assert audit.manual_exclusions(manual) == [{"record_id": "r1", "reason": "manual_cwe"}, {"record_id": "r2", "reason": "manual_cwe"}]
    group = built()
    result = {
        "audit_version": "1",
        "groups": [
            {
                "group_id": "g1",
                "language": "python",
                "advisory_id": "GHSA-2222-3333-4444",
                "repository_key": "example/webapp",
                "records": {k: v["record_id"] for k, v in group.items()},
                "correspondence": {"strength": "strong", "signals": [], "identifier_overlap": []},
                "fix_pattern": audit.fix_pattern(group["vulnerable"]),
                "span": audit.span_precision(group["vulnerable"]),
                "later_fixes": [],
                "later_fix_check": "done",
            }
        ],
        "exclusions": [],
        "sample": [{"group_id": "g1", "language": "python", "item": "pair"}, {"group_id": "g2", "language": "python", "item": "pair"}, {"group_id": "g3", "language": "python", "item": "abstention"}],
    }
    manual["items"].append({"group_id": "g9", "item": "pair", "cwe": "wrong"})  # no longer sampled: not counted
    summary = audit.summarize(result, manual)
    assert summary["manual"]["pending"] == ["g3:abstention"] and summary["manual"]["outside_sample"] == ["g9:pair"]
    assert summary["manual"]["rates"]["cwe"] == {"judged": 2, "errors": 1, "uncertain_or_partial": 0, "error_rate": 0.5, "wilson95": [round(audit.wilson_interval(1, 2)[0], 3), round(audit.wilson_interval(1, 2)[1], 3)]}
    text = audit.report_markdown(summary, result, manual)
    assert "No exploit, test or patch was executed" in text and "manual review | 2" in text


def test_multiline_and_argument_free_calls_are_not_constant_abstentions() -> None:
    group = built()
    for line in ('    return {"orders": list_orders(', '    return {"orders": list_orders()}'):
        abstention = copy.deepcopy(group["caller_abstention"])
        abstention["input"]["code"] = "def orders_view(conn):\n    x = 1\n" + line + "\n"
        abstention["localization"]["call_line"] = 3
        assert audit.abstention_validity(abstention, group["vulnerable"])["problems"] == [], line


def test_later_fix_of_an_unrelated_same_named_function_is_ignored() -> None:
    group = built()

    class Other(LaterFixRepo):
        def __init__(self) -> None:
            other = "def list_orders(a, b, c):\n    total = sum([a, b, c])\n    log(total)\n    return total\n"
            self.files = {"p2": other, "c2": other.replace("log(total)", "audit(total)")}

    later = {"id": "GHSA-6666-7777-8888", "references": [{"type": "FIX", "url": f"https://github.com/example/webapp/commit/{'c' * 40}"}]}
    assert audit.later_fixes(group, [later], Other()) == []


def test_broad_localization_span_excludes_the_vulnerable_record() -> None:
    results = [
        {"group_id": "g1", "records": {"vulnerable": "v1", "post_fix": "p1"}, "span": {"broad": True}},
        {"group_id": "g2", "records": {"vulnerable": "v2", "post_fix": "p2"}, "span": {"broad": False}},
    ]
    assert audit.audit_exclusions(results, {}) == [{"record_id": "v1", "reason": "broad_localization_span"}]


def test_rates_after_automated_exclusions_count_only_surviving_items() -> None:
    manual = {"reviewer": "t", "items": [
        {"group_id": "g1", "item": "pair", "localization": "imprecise"},
        {"group_id": "g2", "item": "pair", "localization": "precise"},
    ]}
    result = {
        "groups": [{"group_id": "g1", "records": {"vulnerable": "v1"}}, {"group_id": "g2", "records": {"vulnerable": "v2"}}],
        "exclusions": [{"record_id": "v1", "reason": "broad_localization_span"}],
        "sample": [{"group_id": "g1", "item": "pair"}, {"group_id": "g2", "item": "pair"}],
    }
    summary = audit.summarize(result, manual)
    assert summary["manual"]["rates"]["localization"]["errors"] == 1
    assert summary["manual"]["rates_after_automated_exclusions"]["localization"] == {"judged": 1, "errors": 0, "error_rate": 0.0, "wilson95": [0.0, round(audit.wilson_interval(0, 1)[1], 3)]}


def test_failed_license_lookup_is_missing_evidence_not_consistency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sentinel.garden.gitsource import GitError

    class NoLicenses:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def root_license_files(self, revision: str):
            raise GitError("unreachable")

    monkeypatch.setattr(review, "UpstreamRepo", NoLicenses)
    row = {
        "quality": {"status": "accepted"},
        "split": "train",
        "kind": "vulnerable",
        "provenance": {
            "repository_key": "example/webapp",
            "repository": "https://github.com/example/webapp",
            "fix_committed_at": "2025-01-01T00:00:00Z",
            "vulnerable_revision": "a" * 40,
            "fixed_revision": "b" * 40,
            "license": {"detected": "MIT", "category": "permissive", "files": [{"path": "LICENSE", "sha256": "c" * 64}]},
            "advisory_id": "GHSA-2222-3333-4444",
        },
    }
    depsdev = review.DepsDev(tmp_path, fetch=lambda url: {"versions": [{"isDefault": True, "licenses": ["MIT"]}]})
    candidates = [{"repository_key": "example/webapp", "ecosystem": "npm", "packages": ["webapp"]}]
    evidence = review.collect_evidence([row], candidates, tmp_path, depsdev, workers=1)[0]
    assert evidence["license_lookup_failed_revisions"] == ["a" * 40, "b" * 40]
    assert not evidence["license_files_identical_across_revisions"]
    assert evidence["tier"] == "B" and "license_lookup_failed" in evidence["tier_reasons"]
