"""Fixtures for training tests: a character-level chat tokenizer and an approved Garden workdir."""

from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml

from sentinel.data.registry import load_registry
from sentinel.garden import render
from sentinel.garden.build import build_records
from sentinel.garden.pipeline import PROJECT_ROOT, manifest_row
from tests.garden_fixtures import PY_NEW, PY_OLD, SETTINGS, FakeRepo, candidate

APPROVED_SOURCES = {"ghsa-osv-fix-commits", "osv", "github-advisory-db"}


class CharTokenizer:
    """Minimal stand-in for a HF chat tokenizer (no network, no torch)."""

    pad_token_id = 0
    eos_token_id = 1
    special = {"<|im_end|>": 2}

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = False, enable_thinking: bool = False) -> str:
        text = "".join(f"<{m['role']}>{m['content']}<|im_end|>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                ids.append(2)
                i += len("<|im_end|>")
            else:
                ids.append(3 + ord(text[i]) % 250)
                i += 1
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.special.get(token)


def approved_registry() -> dict[str, Any]:
    registry = load_registry()
    for source in registry["sources"]:
        if source["source_id"] in APPROVED_SOURCES:
            source.update(
                production_approved=True,
                revision="0" * 40,
                license_status="reviewed",
                terms_status="approved_for_project",
                redistribution_status="not_redistributed",
                reviewer="test-reviewer",
            )
    return registry


def approval_for(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository_key": record["provenance"]["repository_key"],
        "decision": "approved_for_training",
        "license_spdx": "MIT",
        "license_file": "LICENSE",
        "license_file_sha256": record["provenance"]["license"]["files"][0]["sha256"],
        "terms_status": "approved_for_project",
        "redistribution_status": "not_redistributed",
        "reviewer": "test-reviewer",
        "reviewed_at": "2026-11-04",
        "evidence_urls": ["https://example.com/LICENSE"],
        "notes": "",
    }


def _records(repo: str, advisory: str, split: str) -> list[dict[str, Any]]:
    records = build_records(candidate(repo=repo, advisory=advisory), FakeRepo({"shop/orders.py": PY_OLD}, {"shop/orders.py": PY_NEW}), SETTINGS, "t", "ghsa-osv-fix-commits", ["osv", "github-advisory-db"])
    for record in records:
        record["quality"] = {"status": "accepted", "failed_checks": [], "flags": []}
        record["split"] = split
        record["governance"] = {"approval_status": "training_approved", "blockers": []}
    return records


class GardenWorkdir:
    """A throw-away Garden config + workdir under artifacts/ with approved train/validation records."""

    def __init__(self) -> None:
        self.name = f"test-{uuid.uuid4().hex[:10]}"
        self.root = PROJECT_ROOT / "artifacts" / "garden" / self.name
        self.workdir = self.root / "work"
        self.committed = self.root / "committed"
        self.config_path = self.root / "garden.yaml"
        base = yaml.safe_load((PROJECT_ROOT / "configs" / "garden" / "seed_v1.yaml").read_text())
        base["workdir"] = f"artifacts/garden/{self.name}/work"
        base["committed_dir"] = f"artifacts/garden/{self.name}/committed"
        self.root.mkdir(parents=True)
        self.config_path.write_text(yaml.safe_dump(base))
        self.train = _records("example/webapp", "GHSA-2222-3333-4444", "train")
        self.validation = _records("example/other", "GHSA-5555-6666-7777", "validation")
        self.extra_manifest: list[dict[str, Any]] = []
        self.approvals = {r["provenance"]["repository_key"]: approval_for(r) for r in self.train + self.validation}

    def write(self) -> "GardenWorkdir":
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.committed.mkdir(parents=True, exist_ok=True)

        def dump(path: Path, rows: list[dict[str, Any]]) -> None:
            path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

        dump(self.workdir / "train.jsonl", self.train)
        dump(self.workdir / "validation.jsonl", self.validation)
        dump(self.workdir / "train.chat.jsonl", [render.to_chat_example(r) for r in self.train])
        dump(self.committed / "manifest.jsonl", [manifest_row(r) for r in self.train + self.validation] + self.extra_manifest)
        return self

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def copy_record(self, record: dict[str, Any], **changes: Any) -> dict[str, Any]:
        clone = copy.deepcopy(record)
        clone.update(changes)
        return clone
