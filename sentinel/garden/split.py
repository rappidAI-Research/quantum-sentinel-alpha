"""Repository-level split with a temporal safeguard.

1. Every repository is assigned to exactly one bucket - train, validation or holdout_reserved - by
   a salted hash of its repository key, so no repository contributes to two buckets.
2. A record whose fix was committed on/after the temporal cutoff never enters train or
   validation. If its repository is in train/validation it goes to `temporal_quarantine` (it can
   neither train nor serve as an independent holdout); in a holdout_reserved repository it stays a
   holdout candidate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SplitSettings:
    salt: str
    validation_per_mille: int
    holdout_reserved_per_mille: int
    temporal_cutoff: str
    temporal_cutoff_status: str


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def repository_bucket(repository_key: str, settings: SplitSettings) -> str:
    value = int(hashlib.sha256(f"{settings.salt}\x00{repository_key}".encode("utf-8")).hexdigest(), 16) % 1000
    if value < settings.validation_per_mille:
        return "validation"
    if value < settings.validation_per_mille + settings.holdout_reserved_per_mille:
        return "holdout_reserved"
    return "train"


def assign_split(record: dict[str, Any], settings: SplitSettings) -> str:
    bucket = repository_bucket(record["provenance"]["repository_key"], settings)
    after_cutoff = parse_time(record["provenance"]["fix_committed_at"]) >= parse_time(settings.temporal_cutoff)
    if after_cutoff and bucket in {"train", "validation"}:
        return "temporal_quarantine"
    return bucket


def check_split_invariants(records: list[dict[str, Any]], settings: SplitSettings) -> list[str]:
    """Violations of repository separation or the temporal cutoff (empty list = OK)."""
    problems = []
    repos: dict[str, set[str]] = {}
    cutoff = parse_time(settings.temporal_cutoff)
    for record in records:
        split = record["split"]
        if split in {"train", "validation", "holdout_reserved"}:
            repos.setdefault(record["provenance"]["repository_key"], set()).add(split)
        if split in {"train", "validation"} and parse_time(record["provenance"]["fix_committed_at"]) >= cutoff:
            problems.append(f"{record['record_id']}: {split} record after temporal cutoff")
    for repo, splits in sorted(repos.items()):
        if len(splits) > 1:
            problems.append(f"{repo}: repository in several splits {sorted(splits)}")
    return problems
