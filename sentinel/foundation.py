from __future__ import annotations

from pathlib import Path

from sentinel.contracts import load_finding_schema
from sentinel.data.registry import load_registry, validate_registry
from sentinel.eval.bakeoff import load_bakeoff
from sentinel.eval.coding_regression import load_tasks
from sentinel.eval.sandbox import LocalProcessExecutor, load_policy
from sentinel.eval.io import PROJECT_ROOT, assert_publicly_committable, case_set_sha256, load_jsonl, validate_cases
from sentinel.foundation_model import load_foundation, validate_foundation
from sentinel.garden import pipeline as garden
from sentinel.garden.governance import load_approvals

BAKEOFF_CONFIG = PROJECT_ROOT / "configs" / "model" / "bakeoff_qwen35.yaml"


def public_case_files() -> list[Path]:
    """Case files in the public tree; the Git-ignored private holdout directory is skipped."""
    return [p for p in sorted((PROJECT_ROOT / "eval" / "sentinelbench").rglob("*.jsonl")) if "private" not in p.parts]


def main() -> int:
    finding_schema = load_finding_schema()
    registry = load_registry()
    validate_registry(registry)

    approved = [s["source_id"] for s in registry["sources"] if s["production_approved"]]
    print(f"finding schema: {finding_schema['$id']} OK")
    print(f"registry version: {registry['registry_version']} OK")
    print(f"registered sources: {len(registry['sources'])}")
    print(f"training-approved sources: {len(approved)}")

    for path in public_case_files():
        cases = load_jsonl(path)
        validate_cases(cases)
        assert_publicly_committable(cases)
        print(f"eval cases: {path.relative_to(PROJECT_ROOT)} {len(cases)} OK sha256={case_set_sha256(cases)[:12]}")

    tasks = load_tasks()
    for task in tasks:
        # Reference solutions are repository-authored (trusted) code; model output never runs here.
        passed, failure = LocalProcessExecutor().execute(task["canonical_solution"], task["tests"])
        if not passed:
            raise ValueError(f"coding task {task['task_id']} reference solution fails: {failure}")
    print(f"coding suite: coding-v1 {len(tasks)} tasks OK (reference solutions pass)")
    policy = load_policy()
    print(f"coding sandbox policy: {policy.runtime} {policy.image} network={policy.network} user={policy.user} OK")

    bakeoff = load_bakeoff(BAKEOFF_CONFIG)
    pinned = sum(1 for c in bakeoff.candidates.values() if c.revision)
    print(f"bake-off config (optional): {bakeoff.bakeoff_id} OK ({pinned}/{len(bakeoff.candidates)} revisions pinned)")

    foundation = load_foundation()
    validate_foundation(foundation)
    print(
        f"training foundation: {foundation.model_id}@{foundation.revision} OK "
        f"(selected, basis={foundation.basis}, baseline candidate {foundation.baseline_candidate} matches)"
    )

    approvals = load_approvals()
    approved_repos = sum(1 for a in approvals.values() if a["decision"] == "approved_for_training")
    print(f"garden approvals: {len(approvals)} entries, {approved_repos} repositories approved for training OK")
    config = garden.load_config(garden.DEFAULT_CONFIG_PATH)
    if (config.committed_dir / "manifest.jsonl").is_file():
        result = garden.check(config)
        print(
            f"garden dataset: {config.dataset_id} {result['records']} records, {result['accepted']} accepted, "
            f"{result['training_approved']} training-approved OK (metadata only, no code committed)"
        )
    else:
        print(f"garden dataset: {config.dataset_id} not generated yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
