from __future__ import annotations

from pathlib import Path

from sentinel.contracts import load_finding_schema
from sentinel.data.registry import load_registry, validate_registry
from sentinel.eval.bakeoff import load_bakeoff
from sentinel.eval.io import PROJECT_ROOT, assert_publicly_committable, case_set_sha256, load_jsonl, validate_cases

BAKEOFF_CONFIG = PROJECT_ROOT / "configs" / "model" / "bakeoff_qwen35.yaml"


def public_case_files() -> list[Path]:
    """Case files in the public tree; the Git-ignored private holdout directory is skipped."""
    return [p for p in sorted((PROJECT_ROOT / "eval").rglob("*.jsonl")) if "private" not in p.parts]


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

    bakeoff = load_bakeoff(BAKEOFF_CONFIG)
    pinned = sum(1 for c in bakeoff.candidates.values() if c.revision)
    print(f"bake-off config: {bakeoff.bakeoff_id} OK ({pinned}/{len(bakeoff.candidates)} revisions pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
