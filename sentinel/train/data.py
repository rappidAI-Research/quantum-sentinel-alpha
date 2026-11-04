"""Governance-gated training data.

Fail-closed rules (any violation raises `GovernanceError`; nothing is skipped silently):
- train examples come only from `train.chat.jsonl` in the Garden work directory, and must be exactly
  the rendering of `train.jsonl` (no hand-edited chat file);
- every train/validation record must be `training_approved` in the committed manifest, must STILL
  pass the governance gate with today's registry and approvals (a revoked approval stops training),
  and its code hash must match the manifest;
- train records must be in split `train`, validation records in split `validation`; no repository
  may appear in both, and no record may come from a holdout/quarantine repository or an eval-only
  source;
- the prompt version must match the evaluation prompt;
- zero approved training records is an error, never an empty run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sentinel.data.registry import load_registry, validate_registry
from sentinel.eval.prompt import PROMPT_VERSION
from sentinel.garden import render
from sentinel.garden.governance import gate, load_approvals
from sentinel.garden.pipeline import PROJECT_ROOT, load_config

IGNORE_INDEX = -100


class GovernanceError(RuntimeError):
    """Training data is missing, unapproved or inconsistent with the governance records."""


@dataclass(frozen=True)
class TrainingData:
    train: list[dict[str, Any]]  # chat examples
    validation: list[dict[str, Any]]
    sha256: str  # canonical hash of train + validation examples
    record_ids: tuple[str, ...]
    repositories: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GovernanceError(f"missing training export {path} (run `python -m sentinel.garden curate` or `reconstruct`)")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_training_data(garden_config_path: Path, approvals: dict[str, Any] | None = None) -> TrainingData:
    config = load_config(garden_config_path)
    workdir = config.workdir
    manifest = {row["record_id"]: row for row in _read_jsonl(config.committed_dir / "manifest.jsonl")}
    registry = load_registry()
    validate_registry(registry)
    sources = {s["source_id"]: s for s in registry["sources"]}
    approvals = load_approvals() if approvals is None else approvals

    train_records = _read_jsonl(workdir / "train.jsonl")
    validation_records = _read_jsonl(workdir / "validation.jsonl")
    chat = _read_jsonl(workdir / "train.chat.jsonl")
    if not train_records:
        raise GovernanceError(
            "no training-approved records: train.jsonl is empty. Training requires repository approvals in "
            "data/garden/approvals.yaml and approved registry sources (docs/RIGHTS_REVIEW.md)."
        )

    held_out_repos = {
        row["provenance"]["repository_key"] for row in manifest.values() if row["split"] in {"holdout_reserved", "temporal_quarantine"}
    }

    def check(record: dict[str, Any], split: str) -> None:
        rid = record["record_id"]
        row = manifest.get(rid)
        if row is None:
            raise GovernanceError(f"{rid} is not in the committed manifest")
        if row["governance"]["approval_status"] != "training_approved" or record["governance"]["approval_status"] != "training_approved":
            raise GovernanceError(f"{rid} is not training_approved")
        status, blockers = gate(row, sources, approvals)
        if status != "training_approved":
            raise GovernanceError(f"{rid} no longer passes the governance gate: {blockers}")
        if record["split"] != split or row["split"] != split:
            raise GovernanceError(f"{rid} is in split {record['split']}, expected {split}")
        if row["provenance"]["repository_key"] in held_out_repos:
            raise GovernanceError(f"{rid} comes from a holdout/quarantine repository")
        label = row["label"]
        target = record["target"]
        for key in ("decision", "cwe", "alternate_cwes", "start_line", "end_line", "evidence_source"):
            if target.get(key) != label.get(key):
                raise GovernanceError(f"{rid} target {key}={target.get(key)!r} does not match the manifest ({label.get(key)!r})")
        if record["input"]["language"] != label.get("language"):
            raise GovernanceError(f"{rid} language does not match the manifest")
        code_hash = hashlib.sha256(record["input"]["code"].encode("utf-8")).hexdigest()
        if code_hash != row["provenance"]["content_sha256"] or code_hash != record["provenance"]["content_sha256"]:
            raise GovernanceError(f"{rid} code does not match the manifest content hash")

    for record in train_records:
        check(record, "train")
    for record in validation_records:
        check(record, "validation")

    expected_chat = [render.to_chat_example(r) for r in train_records]
    if chat != expected_chat:
        raise GovernanceError("train.chat.jsonl is not the rendering of train.jsonl (regenerate with curate)")
    for example in chat:
        if example["prompt_version"] != PROMPT_VERSION:
            raise GovernanceError(f"{example['record_id']}: prompt {example['prompt_version']} != {PROMPT_VERSION}")
        if [m["role"] for m in example["messages"]] != ["system", "user", "assistant"]:
            raise GovernanceError(f"{example['record_id']}: unexpected message roles")

    train_repos = {r["provenance"]["repository_key"] for r in train_records}
    validation_repos = {r["provenance"]["repository_key"] for r in validation_records}
    if train_repos & validation_repos:
        raise GovernanceError(f"repositories in both train and validation: {sorted(train_repos & validation_repos)[:5]}")
    ids = [r["record_id"] for r in train_records + validation_records]
    if len(ids) != len(set(ids)):
        raise GovernanceError("duplicate record ids across train and validation")

    validation = [render.to_chat_example(r) for r in validation_records]
    canonical = json.dumps({"train": chat, "validation": validation}, sort_keys=True, ensure_ascii=False)
    return TrainingData(
        train=chat,
        validation=validation,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        record_ids=tuple(sorted(ids)),
        repositories=len(train_repos | validation_repos),
    )


# --------------------------------------------------------------------------------------------
# Tokenization with prompt masking (no torch needed)
# --------------------------------------------------------------------------------------------


class ChatTokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...
    def __call__(self, text: str, **kwargs: Any) -> Any: ...
    def convert_tokens_to_ids(self, token: str) -> Any: ...


@dataclass(frozen=True)
class EncodedExample:
    record_id: str
    input_ids: list[int]
    labels: list[int]


def encode_example(tokenizer: ChatTokenizer, example: dict[str, Any], max_seq_len: int, end_of_turn: str) -> EncodedExample | None:
    """Prompt rendered exactly as at evaluation time; loss only on the assistant reply + end of turn.

    Returns None when the example does not fit: examples are dropped, never truncated, so a
    truncated code excerpt can never be paired with its full label.
    """
    *prompt_messages, reply = example["messages"]
    if reply["role"] != "assistant":
        raise ValueError(f"{example['record_id']}: last message must be the assistant reply")
    # Same call as sentinel.eval.backends.HFBackend.generate: thinking disabled, generation prompt on.
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    reply_ids = list(tokenizer(reply["content"] + end_of_turn, add_special_tokens=False)["input_ids"])
    if len(prompt_ids) + len(reply_ids) > max_seq_len:
        return None
    return EncodedExample(
        record_id=example["record_id"],
        input_ids=prompt_ids + reply_ids,
        labels=[IGNORE_INDEX] * len(prompt_ids) + reply_ids,
    )


def encode_all(tokenizer: ChatTokenizer, examples: list[dict[str, Any]], max_seq_len: int, end_of_turn: str) -> tuple[list[EncodedExample], list[str]]:
    eot = tokenizer.convert_tokens_to_ids(end_of_turn)
    if eot is None or (isinstance(eot, int) and eot < 0) or isinstance(eot, list):
        raise ValueError(f"end-of-turn marker {end_of_turn!r} is not a single token of this tokenizer")
    kept, dropped = [], []
    for example in examples:
        encoded = encode_example(tokenizer, example, max_seq_len, end_of_turn)
        if encoded is None:
            dropped.append(example["record_id"])
        else:
            kept.append(encoded)
    return kept, dropped


def batches(examples: list[EncodedExample], micro_batch_size: int, epoch: int, seed: int) -> list[list[EncodedExample]]:
    """Deterministic per-epoch shuffle (independent of torch), so a resumed run sees the same order."""
    order = sorted(examples, key=lambda e: hashlib.sha256(f"{seed}\x00{epoch}\x00{e.record_id}".encode()).hexdigest())
    return [order[i : i + micro_batch_size] for i in range(0, len(order), micro_batch_size)]


def pad_batch(batch: list[EncodedExample], pad_id: int) -> dict[str, list[list[int]]]:
    width = max(len(e.input_ids) for e in batch)
    return {
        "input_ids": [e.input_ids + [pad_id] * (width - len(e.input_ids)) for e in batch],
        "labels": [e.labels + [IGNORE_INDEX] * (width - len(e.labels)) for e in batch],
        "attention_mask": [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in batch],
    }


__all__ = [
    "GovernanceError",
    "IGNORE_INDEX",
    "PROJECT_ROOT",
    "TrainingData",
    "batches",
    "encode_all",
    "encode_example",
    "load_training_data",
    "pad_batch",
]
