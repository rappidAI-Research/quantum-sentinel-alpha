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
from pathlib import Path
from typing import Any

import yaml


POLICY_KEYS = {"policy_id", "status", "frozen_at", "salt", "validation_per_mille", "holdout_reserved_per_mille", "temporal_cutoff"}


@dataclass(frozen=True)
class SplitSettings:
    salt: str
    validation_per_mille: int
    holdout_reserved_per_mille: int
    temporal_cutoff: str
    policy_id: str = "unversioned"
    status: str = "provisional"


def load_split_policy(path: Path) -> SplitSettings:
    """Load the shared split policy; dataset builds require it to be frozen."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != POLICY_KEYS:
        raise ValueError(f"split policy must define exactly {sorted(POLICY_KEYS)}")
    if document["status"] != "frozen":
        raise ValueError("split policy must be frozen before it is used to build a dataset")
    if not 0 < document["validation_per_mille"] + document["holdout_reserved_per_mille"] < 1000:
        raise ValueError("split buckets must leave room for train")
    parse_time(document["temporal_cutoff"])
    return SplitSettings(
        salt=document["salt"],
        validation_per_mille=document["validation_per_mille"],
        holdout_reserved_per_mille=document["holdout_reserved_per_mille"],
        temporal_cutoff=document["temporal_cutoff"],
        policy_id=document["policy_id"],
        status=document["status"],
    )


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
