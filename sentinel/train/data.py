"""Governance-gated training data.

Fail-closed rules (any violation raises `GovernanceError`; nothing is skipped or repaired):
- train examples come only from `train.chat.jsonl` (and, for the patch task, `train.patch.chat.jsonl`)
  in the Garden work directory, and must be exactly the rendering of `train.jsonl`;
- `train.jsonl` / `validation.jsonl` must contain exactly the records the committed manifest marks
  `training_approved` for that split (no record added, none dropped);
- every train/validation record must STILL pass the governance gate with today's registry and
  approvals (a revoked approval stops training);
- integrity against the committed manifest (`supervision` block, sentinel.garden.supervision): the
  canonical hashes of the full input (code, context, file, language) and the full target (decision,
  CWE, span, evidence text, patch) and of every rendered chat example must match. Editing any
  supervised field - not only the code or the label - is detected;
- the export files must match their `SHA256SUMS`;
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
from sentinel.garden import render, supervision
from sentinel.garden.governance import gate, load_approvals
from sentinel.garden.pipeline import PROJECT_ROOT, load_config

IGNORE_INDEX = -100


class GovernanceError(RuntimeError):
    """Training data is missing, unapproved or inconsistent with the governance records."""


TASKS = ("detection", "patch")


@dataclass(frozen=True)
class TrainingData:
    train: list[dict[str, Any]]  # chat examples (all selected tasks)
    validation: list[dict[str, Any]]
    sha256: str  # canonical hash of train + validation examples
    record_ids: tuple[str, ...]
    repositories: int
    tasks: tuple[str, ...] = ("detection",)
    examples_by_task: dict[str, dict[str, int]] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GovernanceError(f"missing training export {path} (run `python -m sentinel.garden curate` or `reconstruct`)")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_sums(workdir: Path, names: tuple[str, ...]) -> None:
    sums_path = workdir / "SHA256SUMS"
    if not sums_path.is_file():
        raise GovernanceError(f"missing {sums_path} (regenerate the exports with curate)")
    sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    for name in names:
        digest = hashlib.sha256((workdir / name).read_bytes()).hexdigest()
        if sums.get(name) != digest:
            raise GovernanceError(f"{name} does not match SHA256SUMS (modified after export; regenerate with curate)")


def load_training_data(
    garden_config_path: Path, approvals: dict[str, Any] | None = None, tasks: tuple[str, ...] = ("detection",)
) -> TrainingData:
    unknown = set(tasks) - set(TASKS)
    if not tasks or unknown or "detection" not in tasks:
        raise GovernanceError(f"tasks must include 'detection' and only use {TASKS}, got {tasks}")
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

    for split, records in (("train", train_records), ("validation", validation_records)):
        expected = {rid for rid, row in manifest.items() if row["split"] == split and row["governance"]["approval_status"] == "training_approved"}
        present = [r["record_id"] for r in records]
        if len(present) != len(set(present)):
            raise GovernanceError(f"{split}.jsonl contains duplicate records")
        if set(present) != expected:
            missing, extra = sorted(expected - set(present)), sorted(set(present) - expected)
            raise GovernanceError(
                f"{split}.jsonl does not hold exactly the manifest's approved {split} records "
                f"(missing {missing[:3]}, unexpected {extra[:3]}); regenerate with curate"
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
        for key in ("kind", "group_id", "pair_id"):
            # record structure decides which examples are derived (e.g. a patch example needs its pair)
            if record.get(key) != row.get(key):
                raise GovernanceError(f"{rid} {key}={record.get(key)!r} does not match the manifest ({row.get(key)!r})")
        if record["input"]["language"] != label.get("language"):
            raise GovernanceError(f"{rid} language does not match the manifest")
        code_hash = hashlib.sha256(record["input"]["code"].encode("utf-8")).hexdigest()
        if code_hash != row["provenance"]["content_sha256"] or code_hash != record["provenance"]["content_sha256"]:
            raise GovernanceError(f"{rid} code does not match the manifest content hash")
        bound = row.get("supervision")
        problem = supervision.manifest_supervision_problem(row)
        if problem:
            raise GovernanceError(f"{rid}: {problem}")
        if supervision.canonical_sha256(record["input"]) != bound["input_sha256"]:
            raise GovernanceError(f"{rid} input (code, context, file or language) does not match the manifest")
        if supervision.canonical_sha256(record["target"]) != bound["target_sha256"]:
            raise GovernanceError(f"{rid} target (label, span, evidence text or patch) does not match the manifest")
        if supervision.canonical_sha256(render.to_chat_example(record)) != bound["detection_example_sha256"]:
            raise GovernanceError(f"{rid} detection example does not match the manifest (prompt or rendering changed; rerun curate)")

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
        if supervision.canonical_sha256(example) != manifest[example["record_id"]]["supervision"]["detection_example_sha256"]:
            raise GovernanceError(f"{example['record_id']}: chat example does not match the manifest")
    _verify_sums(workdir, ("train.jsonl", "validation.jsonl", "train.chat.jsonl", "train.patch.chat.jsonl"))

    def bound_patch_examples(records: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        examples = supervision.patch_examples(records)
        expected = {
            rid for rid, row in manifest.items()
            if row["split"] == label
            and row["governance"]["approval_status"] == "training_approved"
            and (row.get("supervision") or {}).get("patch_eligibility") == "eligible"
        }
        produced = [e["record_id"] for e in examples]
        if set(produced) != expected or len(produced) != len(set(produced)):
            raise GovernanceError(
                f"{label} patch examples do not match the manifest's approved patch-eligible records "
                f"(missing {sorted(expected - set(produced))[:3]}, unexpected {sorted(set(produced) - expected)[:3]})"
            )
        for example in examples:
            digest = manifest[example["record_id"]]["supervision"]["patch_example_sha256"]
            if digest is None or supervision.canonical_sha256(example) != digest:
                raise GovernanceError(f"{example['record_id']}: {label} patch example is not the verified pair bound in the manifest")
        return examples

    train_patch = bound_patch_examples(train_records, "train")
    validation_patch = bound_patch_examples(validation_records, "validation")
    if _read_jsonl(workdir / "train.patch.chat.jsonl") != train_patch:
        raise GovernanceError("train.patch.chat.jsonl is not the rendering of the verified pairs in train.jsonl (regenerate with curate)")
    if "patch" in tasks and not train_patch:
        raise GovernanceError("task 'patch' selected but no approved, verified vulnerable/fixed pair exists in train.jsonl")

    train_repos = {r["provenance"]["repository_key"] for r in train_records}
    validation_repos = {r["provenance"]["repository_key"] for r in validation_records}
    if train_repos & validation_repos:
        raise GovernanceError(f"repositories in both train and validation: {sorted(train_repos & validation_repos)[:5]}")
    ids = [r["record_id"] for r in train_records + validation_records]
    if len(ids) != len(set(ids)):
        raise GovernanceError("duplicate record ids across train and validation")

    validation = [render.to_chat_example(r) for r in validation_records]
    train_examples, validation_examples = list(chat), list(validation)
    if "patch" in tasks:
        train_examples += train_patch
        validation_examples += validation_patch
    canonical = json.dumps({"tasks": list(tasks), "train": train_examples, "validation": validation_examples}, sort_keys=True, ensure_ascii=False)
    return TrainingData(
        train=train_examples,
        validation=validation_examples,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        record_ids=tuple(sorted(ids)),
        repositories=len(train_repos | validation_repos),
        tasks=tuple(tasks),
        examples_by_task={
            "train": {"detection": len(chat), "patch": len(train_patch) if "patch" in tasks else 0},
            "validation": {"detection": len(validation), "patch": len(validation_patch) if "patch" in tasks else 0},
        },
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
    task: str = "detection"


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
        task=example.get("task", "detection"),
    )


def encode_all(tokenizer: ChatTokenizer, examples: list[dict[str, Any]], max_seq_len: int, end_of_turn: str) -> tuple[list[EncodedExample], list[str]]:
    eot = tokenizer.convert_tokens_to_ids(end_of_turn)
    if eot is None or (isinstance(eot, int) and eot < 0) or isinstance(eot, list):
        raise ValueError(f"end-of-turn marker {end_of_turn!r} is not a single token of this tokenizer")
    kept, dropped = [], []
    for example in examples:
        encoded = encode_example(tokenizer, example, max_seq_len, end_of_turn)
        if encoded is None:
            task = example.get("task", "detection")
            dropped.append(example["record_id"] if task == "detection" else f"{task}:{example['record_id']}")
        else:
            kept.append(encoded)
    return kept, dropped


def batches(examples: list[EncodedExample], micro_batch_size: int, epoch: int, seed: int) -> list[list[EncodedExample]]:
    """Deterministic per-epoch shuffle (independent of torch), so a resumed run sees the same order."""
    # a vulnerable record can appear once per task, so the task is part of the key
    order = sorted(examples, key=lambda e: hashlib.sha256(f"{seed}\x00{epoch}\x00{e.task}\x00{e.record_id}".encode()).hexdigest())
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
