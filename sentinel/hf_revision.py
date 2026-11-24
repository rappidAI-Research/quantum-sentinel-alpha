"""Prove which Hugging Face revision a loaded model came from, and reset CUDA memory statistics safely.

A requested revision is not evidence of the loaded one, and transformers does not record a commit
hash for every checkpoint: the pinned Ornith-1.5-9B loads with ``config._commit_hash = None`` (also
on ``AutoConfig`` and ``text_config``). The proof used here is the content of the files that were
loaded:

1. resolve the files exactly as ``from_pretrained`` does, from the local Hugging Face cache
   (``local_files_only=True``): config, weight index, every weight shard the index names (or the
   single ``model.safetensors``) and the tokenizer files present in the snapshot;
2. all of them must lie in one ``snapshots/<commit>`` directory, and that commit must be the
   requested revision;
3. every one of them must match a reference digest *for that revision*: the pinned digests of the
   selected foundation (``configs/model/foundation.yaml`` ``files_sha256``) or, for any other model,
   the per-file sha256 (LFS) / git blob id that the Hub API reports for exactly that revision.

Anything missing, extra-unverifiable or different raises ``RevisionUnverified``. Callers fail
closed: training refuses to start, an evaluation stays non-reportable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG = PROJECT_ROOT / "configs" / "model" / "foundation.yaml"
_SHA = re.compile(r"^[0-9a-f]{40}$")
# Tokenizer files that AutoTokenizer may read; each one present in the snapshot is verified.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
)
_CHUNK = 16 * 2**20
# Any of these in the snapshot could be what the loader reads (model.safetensors takes precedence over
# a sharded index, .bin is a fallback), so every one present is verified, whichever the loader picks.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack")
_WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


class RevisionUnverified(RuntimeError):
    """The loaded files could not be proven to belong to the requested revision."""


@dataclass(frozen=True)
class Expected:
    """Reference digest of one file at one revision: sha256 (pinned or LFS) or a git blob id."""

    sha256: str | None = None
    git_blob_sha1: str | None = None
    size: int | None = None


def pinned_digests(model_id: str, revision: str, path: Path = FOUNDATION_CONFIG) -> dict[str, Expected] | None:
    """The selected foundation's committed digests, if (model_id, revision) is the pinned foundation."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("model_id") != model_id or document.get("revision") != revision:
        return None
    files = document.get("files_sha256") or {}
    return {name: Expected(sha256=str(digest)) for name, digest in files.items()} or None


def hub_digests(model_id: str, revision: str) -> dict[str, Expected] | None:
    """Per-file digests the Hub reports for exactly this revision (network); None when unavailable."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception:  # noqa: BLE001 - offline or unknown: the caller fails closed
        return None
    if getattr(info, "sha", None) != revision:
        return None
    digests: dict[str, Expected] = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            digests[sibling.rfilename] = Expected(sha256=lfs.sha256, size=lfs.size)
        elif getattr(sibling, "blob_id", None):
            digests[sibling.rfilename] = Expected(git_blob_sha1=sibling.blob_id, size=sibling.size)
    return digests or None


def _hub_resolve(model_id: str, filename: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename, revision=revision, local_files_only=True)


def _file_digests(path: Path) -> tuple[str, str, int]:
    """sha256 and git blob sha1 of a file, streamed (weight shards are several GB)."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())  # noqa: S324 - git object id, not a security hash
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest(), size


def _loaded_files(snapshot: Path, resolve: Callable[[str], Path]) -> list[str]:
    """Every file the loader may read: config, all weight files and indexes present, the shards an
    index names, and the tokenizer files present. Verifying all of them avoids depending on the
    loader's precedence (model.safetensors before a sharded index, .bin as fallback)."""
    files = ["config.json"]
    present = {path.name for path in snapshot.iterdir()}
    weights = sorted(name for name in present if name.endswith(_WEIGHT_SUFFIXES))
    indexes = [name for name in _WEIGHT_INDEXES if name in present]
    for index in indexes:
        weight_map = json.loads((snapshot / index).read_text(encoding="utf-8")).get("weight_map") or {}
        shards = sorted(set(weight_map.values()))
        if not shards:
            raise RevisionUnverified(f"{index} names no weight files")
        weights = sorted(set(weights) | set(shards))  # a shard the index names must exist and verify
    if not weights:
        raise RevisionUnverified(f"no weight files in the resolved snapshot {snapshot}")
    files += indexes + weights
    files += [name for name in TOKENIZER_FILES if name in present]
    for name in files:
        resolve(name)  # every file must resolve through the cache for this revision
    return files


def verify_loaded_revision(
    model_id: str,
    revision: str,
    *,
    expected: dict[str, Expected] | None = None,
    resolve: Callable[[str, str, str], str] = _hub_resolve,
    online: Callable[[str, str], dict[str, Expected] | None] = hub_digests,
) -> dict[str, Any]:
    """Verify the cached files `from_pretrained` loaded for (model_id, revision); raise RevisionUnverified."""
    if not _SHA.match(revision or ""):
        raise RevisionUnverified(f"requested revision {revision!r} is not a 40-character commit SHA")
    if Path(model_id).exists():
        # from_pretrained loads a local path with this name instead of the Hub snapshot; the cache
        # files verified below would then not be the files that were loaded.
        raise RevisionUnverified(f"{model_id!r} is a local path that shadows the Hub repository; remove or rename it")
    started = time.monotonic()

    def resolved(name: str) -> Path:
        try:
            path = Path(resolve(model_id, name, revision))
        except Exception as exc:  # noqa: BLE001 - not in the cache for this revision
            raise RevisionUnverified(f"{name} does not resolve from the local cache for {revision}: {exc}") from exc
        if path.parent != snapshot:
            raise RevisionUnverified(f"{name} resolves outside the snapshot {snapshot}: {path}")
        return path

    try:
        config_path = Path(resolve(model_id, "config.json", revision))
    except Exception as exc:  # noqa: BLE001
        raise RevisionUnverified(f"config.json does not resolve from the local cache for {revision}: {exc}") from exc
    snapshot = config_path.parent
    commit = snapshot.name
    if snapshot.parent.name != "snapshots" or not _SHA.match(commit):
        raise RevisionUnverified(f"config.json is not in a Hugging Face snapshot directory: {config_path}")
    if commit != revision:
        raise RevisionUnverified(f"the cache resolved snapshot {commit}, not the requested {revision}")
    files = _loaded_files(snapshot, resolved)

    method = "provided"
    if expected is None:
        expected, method = pinned_digests(model_id, revision), "pinned_digests"
    if expected is None:
        expected, method = online(model_id, revision), "hub_metadata"
    if not expected:
        raise RevisionUnverified(f"no reference digests for {model_id}@{revision} (not pinned, Hub metadata unavailable)")

    verified: dict[str, str] = {}
    for name in files:
        reference = expected.get(name)
        if reference is None:
            raise RevisionUnverified(f"{name} was loaded but has no reference digest for {revision}")
        sha256, blob, size = _file_digests(snapshot / name)
        if reference.size is not None and reference.size != size:
            raise RevisionUnverified(f"{name}: {size} bytes, expected {reference.size} for {revision}")
        if reference.sha256 is not None and reference.sha256 != sha256:
            raise RevisionUnverified(f"{name}: sha256 {sha256} does not match {reference.sha256} for {revision}")
        if reference.sha256 is None and reference.git_blob_sha1 != blob:
            raise RevisionUnverified(f"{name}: git blob {blob} does not match {reference.git_blob_sha1} for {revision}")
        verified[name] = sha256
    return {
        "revision": commit,
        "method": method,
        "snapshot": str(snapshot),
        "files_sha256": verified,
        "seconds": round(time.monotonic() - started, 3),
    }


def reset_cuda_peak_memory(torch_module: Any = None) -> None:
    """Reset peak-memory statistics on every visible GPU, initializing CUDA and each device's allocator first.

    `torch.cuda.reset_peak_memory_stats` fails on a device whose caching allocator has not been
    initialized yet ("Invalid device argument ... did you call init?"), which is the state before the
    model is loaded. A one-element allocation per device creates the context and allocator state.
    """
    torch = torch_module
    if torch is None:
        import torch  # noqa: PLC0415 - heavy, optional dependency
    if not torch.cuda.is_available():
        return
    torch.cuda.init()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.empty(1, device=f"cuda:{index}")
            torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats(index)


__all__ = [
    "Expected",
    "RevisionUnverified",
    "hub_digests",
    "pinned_digests",
    "reset_cuda_peak_memory",
    "verify_loaded_revision",
]
