from __future__ import annotations

from sentinel.contracts import load_finding_schema
from sentinel.data.registry import load_registry, validate_registry


def main() -> int:
    finding_schema = load_finding_schema()
    registry = load_registry()
    validate_registry(registry)

    approved = [s["source_id"] for s in registry["sources"] if s["production_approved"]]
    print(f"finding schema: {finding_schema['$id']} OK")
    print(f"registry version: {registry['registry_version']} OK")
    print(f"registered sources: {len(registry['sources'])}")
    print(f"training-approved sources: {len(approved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
