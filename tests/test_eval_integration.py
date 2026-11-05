"""Combined behaviour of the coding sandbox (PR #7) and the versioned review prompt / adapter
contract (PR #8). Runs without Docker: isolation is made unavailable or replaced explicitly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.eval import run_model
from sentinel.eval.backends import BaselineBackend
from sentinel.eval.coding_regression import CodingV1
from sentinel.eval.prompt import PROMPT_VERSION
from sentinel.eval.sandbox import IsolationUnavailable, LocalProcessExecutor

BAKEOFF = "configs/model/bakeoff_qwen35.yaml"


@pytest.fixture
def no_sandbox(monkeypatch):
    def unavailable(self):
        raise IsolationUnavailable("docker daemon is not reachable")

    monkeypatch.setattr(CodingV1, "preflight", unavailable)


def _adapter(tmp_path: Path, base: str, revision: str, prompt: str | None) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "revision": revision}))
    if prompt:
        (adapter / "training_run.json").write_text(json.dumps({"prompt_version": prompt}))
    return adapter


def test_wrong_adapter_gets_its_own_error_even_without_a_sandbox(tmp_path, no_sandbox, capsys) -> None:
    # integration defect: the sandbox preflight ran first, so a mismatched adapter exited 3
    adapter = _adapter(tmp_path, "Qwen/Qwen3.5-4B", "1" * 40, None)
    with pytest.raises(SystemExit) as excinfo:
        run_model.main(["--bakeoff", BAKEOFF, "--candidate", "qwen35-9b", "--adapter", str(adapter), "--output", str(tmp_path / "r.json")])
    assert excinfo.value.code == 2 and "trained for" in capsys.readouterr().err


def test_adapter_trained_on_another_prompt_is_refused_before_the_sandbox(tmp_path, no_sandbox, capsys) -> None:
    from sentinel.foundation_model import load_foundation

    foundation = load_foundation()
    adapter = _adapter(tmp_path, foundation.model_id, foundation.revision, "quickeval-review-v1")
    with pytest.raises(SystemExit) as excinfo:
        run_model.main(["--bakeoff", BAKEOFF, "--candidate", "qwen35-9b", "--adapter", str(adapter), "--output", str(tmp_path / "r.json")])
    assert excinfo.value.code == 2 and "trained with prompt" in capsys.readouterr().err


def test_9b_baseline_security_only_without_docker_uses_the_v2_prompt(tmp_path, no_sandbox, monkeypatch) -> None:
    monkeypatch.setattr(run_model, "HFBackend", lambda *a, **kw: BaselineBackend())
    out = tmp_path / "run.json"
    assert run_model.main(["--bakeoff", BAKEOFF, "--candidate", "qwen35-9b", "--skip-coding-regression", "--output", str(out)]) == 0
    artifact = json.loads(out.read_text())
    assert artifact["contract"]["prompt_version"] == PROMPT_VERSION == "quickeval-review-v2"
    assert artifact["contract"]["parser_version"] == "3"
    assert artifact["coding_regression"]["status"] == "not_run"
    assert artifact["bakeoff"]["candidate"] == "qwen35-9b"


def test_without_skip_the_9b_baseline_refuses_to_run_unsandboxed(tmp_path, no_sandbox, monkeypatch) -> None:
    monkeypatch.setattr(run_model, "HFBackend", lambda *a, **kw: BaselineBackend())
    out = tmp_path / "run.json"
    assert run_model.main(["--bakeoff", BAKEOFF, "--candidate", "qwen35-9b", "--output", str(out)]) == 3
    assert not out.exists()


def test_runner_never_falls_back_to_the_local_executor() -> None:
    from sentinel.eval.backends import GenerationSettings, RuntimeSettings
    from sentinel.eval.io import PROJECT_ROOT, load_jsonl
    from sentinel.eval.runner import run_quickeval

    class Counting(BaselineBackend):
        calls = 0

        def generate(self, messages):
            Counting.calls += 1
            return super().generate(messages)

    cases = load_jsonl(PROJECT_ROOT / "eval/sentinelbench/quickeval_v1.jsonl")[:2]
    with pytest.raises(IsolationUnavailable, match="not an isolated executor"):
        run_quickeval(
            cases, Counting(), model_id="baseline/regex", revision=None, generation=GenerationSettings(), runtime=RuntimeSettings(),
            cases_path="x", full_case_set=False, coding_suite=CodingV1(executor=LocalProcessExecutor()),
        )
    assert Counting.calls == 0


def test_runs_with_different_prompts_or_sandbox_policies_are_not_compared(tmp_path, monkeypatch) -> None:
    from sentinel.eval import coding_regression, compare
    from sentinel.eval.sandbox import load_policy

    def stand_in(memory):
        class StandIn(LocalProcessExecutor):  # trusted: runs only the baseline backend's fixed answers
            isolated = True

            def describe(self):
                return {"executor": "container", "isolated": True, "policy": {**load_policy().__dict__, "memory_mib": memory}}

        return StandIn

    base = ["--backend", "baseline", "--cases", "eval/sentinelbench/quickeval_v1.jsonl", "--coding-suite", "coding-v1"]
    runs = {}
    for name, version, memory in (("v2", None, 1024), ("v1", "quickeval-review-v1", 1024), ("v2-other-policy", None, 2048)):
        monkeypatch.setattr(coding_regression, "ContainerExecutor", stand_in(memory))
        out = tmp_path / f"{name}.json"
        assert run_model.main(base + ["--output", str(out)] + (["--prompt-version", version] if version else [])) == 0
        runs[name] = json.loads(out.read_text())
    problems, _ = compare.comparability(runs["v2"], runs["v1"])
    assert any("prompt" in p for p in problems)
    problems, _ = compare.comparability(runs["v2"], runs["v2-other-policy"])
    assert any("contract_sha256" in p for p in problems)
    assert compare.comparability(runs["v2"], runs["v2"])[0] == []
