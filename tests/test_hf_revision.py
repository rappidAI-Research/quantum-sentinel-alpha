"""Loaded-revision proof from cache snapshot files, and the CUDA peak-memory reset order (no GPU, no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sentinel.hf_revision import Expected, RevisionUnverified, pinned_digests, reset_cuda_peak_memory, verify_loaded_revision

MODEL = "org/model"
REVISION = "4" * 40
OTHER = "5" * 40


def _cache(root: Path, commit: str = REVISION, shards: int = 2) -> tuple[Path, dict[str, Expected]]:
    """A Hugging Face cache layout: snapshots/<commit>/<file> -> ../../blobs/<id>."""
    repo = root / "models--org--model"
    snapshot = repo / "snapshots" / commit
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True)
    names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
    files = {
        "config.json": b'{"model_type": "qwen3_5"}',
        "model.safetensors.index.json": json.dumps({"weight_map": {f"w{i}": n for i, n in enumerate(names)}}).encode(),
        **{name: f"weights {name}".encode() * 1000 for name in names},
        "tokenizer.json": b'{"model": {}}',
        "tokenizer_config.json": b'{"eos_token": "<|im_end|>"}',
    }
    expected = {}
    for name, content in files.items():
        digest = hashlib.sha256(content).hexdigest()
        blob = repo / "blobs" / digest
        blob.write_bytes(content)
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / digest)
        expected[name] = Expected(sha256=digest)
    return snapshot, expected


def _resolver(root: Path):
    def resolve(model_id: str, filename: str, revision: str) -> str:
        path = root / "models--org--model" / "snapshots" / revision / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    return resolve


def _offline(*_):
    return None


def test_loaded_files_matching_the_revision_digests_are_verified(tmp_path: Path) -> None:
    snapshot, expected = _cache(tmp_path)
    result = verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path), online=_offline)
    assert result["revision"] == REVISION and result["snapshot"] == str(snapshot)
    assert set(result["files_sha256"]) == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_tampered_weight_shard_is_refused(tmp_path: Path) -> None:
    snapshot, expected = _cache(tmp_path)
    shard = (snapshot / "model-00002-of-00002.safetensors").resolve()
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(RevisionUnverified, match="model-00002-of-00002.safetensors: sha256"):
        verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path), online=_offline)


def test_requested_revision_is_never_taken_as_evidence(tmp_path: Path) -> None:
    # A snapshot directory named after the request is not proof by itself: without reference
    # digests for that revision (pinned or from the Hub) the loaded revision stays unverified.
    _cache(tmp_path, commit=REVISION)
    with pytest.raises(RevisionUnverified, match="no reference digests"):
        verify_loaded_revision(MODEL, REVISION, resolve=_resolver(tmp_path), online=_offline)


def test_snapshot_of_another_commit_is_refused(tmp_path: Path) -> None:
    _cache(tmp_path, commit=OTHER)

    def resolve_main(model_id: str, filename: str, revision: str) -> str:  # e.g. a ref that moved
        return str(tmp_path / "models--org--model" / "snapshots" / OTHER / filename)

    with pytest.raises(RevisionUnverified, match="not the requested"):
        verify_loaded_revision(MODEL, REVISION, expected={}, resolve=resolve_main, online=_offline)


def test_missing_shard_or_unpinned_loaded_file_is_refused(tmp_path: Path) -> None:
    snapshot, expected = _cache(tmp_path)
    (snapshot / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(RevisionUnverified, match="does not resolve"):
        verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path), online=_offline)

    snapshot, expected = _cache(tmp_path / "second")
    del expected["tokenizer_config.json"]
    with pytest.raises(RevisionUnverified, match="no reference digest"):
        verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path / "second"), online=_offline)


def test_single_file_checkpoint_next_to_an_index_is_verified_too(tmp_path: Path) -> None:
    # transformers loads model.safetensors before a sharded index: it must be verified, not ignored.
    snapshot, expected = _cache(tmp_path)
    stray = b"other weights"
    blob = tmp_path / "models--org--model" / "blobs" / hashlib.sha256(stray).hexdigest()
    blob.write_bytes(stray)
    (snapshot / "model.safetensors").symlink_to(Path("..") / ".." / "blobs" / blob.name)
    with pytest.raises(RevisionUnverified, match="model.safetensors was loaded but has no reference digest"):
        verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path), online=_offline)
    expected["model.safetensors"] = Expected(sha256="0" * 64)
    with pytest.raises(RevisionUnverified, match="model.safetensors: sha256"):
        verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path), online=_offline)


def test_local_directory_shadowing_the_repository_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # from_pretrained would load ./org/model instead of the verified cache snapshot.
    _, expected = _cache(tmp_path / "cache")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "model").mkdir(parents=True)
    with pytest.raises(RevisionUnverified, match="local path"):
        verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=_resolver(tmp_path / "cache"), online=_offline)


def test_hub_metadata_digests_accept_git_blob_ids(tmp_path: Path) -> None:
    snapshot, pinned = _cache(tmp_path)
    online = {}
    for name, reference in pinned.items():
        content = (snapshot / name).read_bytes()
        if name.endswith(".safetensors"):
            online[name] = Expected(sha256=reference.sha256, size=len(content))  # LFS: sha256
        else:
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()  # noqa: S324
            online[name] = Expected(git_blob_sha1=blob, size=len(content))
    result = verify_loaded_revision(MODEL, REVISION, resolve=_resolver(tmp_path), online=lambda *_: online)
    assert result["method"] == "hub_metadata"
    online["config.json"] = Expected(git_blob_sha1="0" * 40)
    with pytest.raises(RevisionUnverified, match="git blob"):
        verify_loaded_revision(MODEL, REVISION, resolve=_resolver(tmp_path), online=lambda *_: online)


def test_real_huggingface_hub_cache_resolution(tmp_path: Path) -> None:
    hub = pytest.importorskip("huggingface_hub")
    _, expected = _cache(tmp_path)

    def resolve(model_id: str, filename: str, revision: str) -> str:
        return hub.hf_hub_download(model_id, filename, revision=revision, local_files_only=True, cache_dir=tmp_path)

    result = verify_loaded_revision(MODEL, REVISION, expected=expected, resolve=resolve, online=_offline)
    assert result["revision"] == REVISION
    with pytest.raises(RevisionUnverified):
        verify_loaded_revision(MODEL, OTHER, expected=expected, resolve=resolve, online=_offline)


def test_selected_foundation_pins_every_loaded_file() -> None:
    from sentinel.foundation_model import load_foundation

    foundation = load_foundation()
    digests = pinned_digests(foundation.model_id, foundation.revision)
    assert digests is not None
    shards = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    assert shards | required <= set(digests)
    assert all(len(d.sha256 or "") == 64 for d in digests.values())
    assert pinned_digests(foundation.model_id, "0" * 40) is None  # only the pinned revision has pinned digests


class _FakeCuda:
    def __init__(self, calls: list[str], devices: int = 2) -> None:
        self.calls, self.devices = calls, devices

    def is_available(self) -> bool:
        return True

    def init(self) -> None:
        self.calls.append("init")

    def device_count(self) -> int:
        return self.devices

    def device(self, index: int):
        calls = self.calls

        class _Context:
            def __enter__(self):
                calls.append(f"device{index}")

            def __exit__(self, *exc):
                return False

        return _Context()

    def synchronize(self, index: int) -> None:
        self.calls.append(f"sync{index}")

    def reset_peak_memory_stats(self, index: int) -> None:
        # The real call fails on a device whose allocator was never initialized.
        assert f"alloc{index}" in self.calls and "init" in self.calls, "reset before CUDA/allocator init"
        self.calls.append(f"reset{index}")


class _FakeTorch:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.cuda = _FakeCuda(calls)

    def empty(self, size: int, device: str):
        self.calls.append(f"alloc{device.split(':')[1]}")


def test_peak_memory_is_reset_only_after_cuda_and_allocator_init() -> None:
    calls: list[str] = []
    reset_cuda_peak_memory(_FakeTorch(calls))
    assert calls[0] == "init"
    assert calls.index("alloc0") < calls.index("reset0") and calls.index("alloc1") < calls.index("reset1")


def test_no_gpu_means_no_cuda_calls() -> None:
    calls: list[str] = []
    torch = _FakeTorch(calls)
    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    reset_cuda_peak_memory(torch)
    assert calls == []


def test_evaluator_reports_only_a_verified_revision() -> None:
    from sentinel.eval.backends import GenerationSettings, HFBackend, RuntimeSettings

    backend = HFBackend(MODEL, REVISION, RuntimeSettings(), GenerationSettings())
    assert backend.verified_revision() is None  # nothing verified yet: never the requested revision
    backend._revision_verification = {"error": "model-00001-of-00004.safetensors: sha256 mismatch"}
    assert backend.verified_revision() is None
    backend._revision_verification = {"revision": REVISION, "method": "pinned_digests"}
    assert backend.verified_revision() == REVISION
