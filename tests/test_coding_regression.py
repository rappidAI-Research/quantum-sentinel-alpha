import dataclasses
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from sentinel.eval import sandbox
from sentinel.eval.backends import GenerationResult
from sentinel.eval.coding_regression import CodingV1, extract_code, load_tasks
from sentinel.eval.sandbox import ContainerExecutor, IsolationUnavailable, LocalProcessExecutor, check_probe, load_policy

ROOT = Path(__file__).resolve().parents[1]
TASKS = {task["prompt"]: task for task in load_tasks()}
POLICY = load_policy()


class ScriptedCoder:
    name = "fake"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return GenerationResult(text=self.answer(TASKS[messages[-1]["content"]]), generated_tokens=50)


class TrustedIsolatedStandIn(LocalProcessExecutor):
    """Test double: claims isolation so CodingV1 accepts it. Only ever runs repository-authored code."""

    isolated = True


def _container_ready() -> bool:
    try:
        ContainerExecutor().preflight()
        return True
    except IsolationUnavailable:
        return False


CONTAINER = pytest.mark.skipif(not _container_ready(), reason="container sandbox unavailable (docker daemon or pinned image missing)")
EXECUTORS = [pytest.param(LocalProcessExecutor(timeout_seconds=3), id="local")] + [
    pytest.param("container", id="container", marks=CONTAINER)
]


def _executor(kind):
    if kind == "container":
        return ContainerExecutor(dataclasses.replace(POLICY, timeout_seconds=8))
    return kind


# --- suite logic (trusted reference code only) ---------------------------------------------------


def test_reference_solutions_pass_and_wrong_code_fails() -> None:
    suite = CodingV1(executor=TrustedIsolatedStandIn())
    perfect = suite.run(ScriptedCoder(lambda t: f"```python\n{t['canonical_solution']}```")).as_dict()
    assert perfect["status"] == "completed"
    assert perfect["passed"] == perfect["tasks"] == len(TASKS) >= 12
    assert len(perfect["suite_sha256"]) == 64

    broken = suite.run(ScriptedCoder(lambda t: "```python\ndef nothing():\n    return None\n```")).as_dict()
    assert broken["passed"] == 0
    assert all(not d["passed"] for d in broken["details"]["tasks"].values())


def test_truncated_reasoning_yields_no_code() -> None:
    assert extract_code("<think>let me write def f(): ...") is None
    assert extract_code("<think>plan</think>\n```python\nx = 1\n```") == "x = 1"


def test_pin_script_only_fills_unpinned_revisions() -> None:
    spec = importlib.util.spec_from_file_location("pin", ROOT / "scripts" / "pin_hf_revisions.py")
    pin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pin)
    text = re.sub(r"revision: [0-9a-f]{40}", "revision: null", (ROOT / "configs" / "model" / "bakeoff_qwen35.yaml").read_text())
    updated = pin.set_revisions(text, {"Qwen/Qwen3.5-4B": "a" * 40, "Qwen/Qwen3.5-9B": "b" * 40})
    assert updated.count("revision: " + "a" * 40) == 1 and updated.count("revision: " + "b" * 40) == 1
    assert updated.replace("a" * 40, "null").replace("b" * 40, "null") == text


# --- fail-closed execution policy ----------------------------------------------------------------


def test_generated_code_never_runs_without_isolation() -> None:
    coder = ScriptedCoder(lambda t: f"```python\n{t['canonical_solution']}```")
    suite = CodingV1(executor=LocalProcessExecutor())
    with pytest.raises(IsolationUnavailable, match="not an isolated executor"):
        suite.run(coder)
    assert coder.calls == 0  # refused before any generation


def test_missing_runtime_daemon_or_image_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    with pytest.raises(IsolationUnavailable, match="not installed"):
        ContainerExecutor(POLICY).preflight()

    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/" + name)
    down = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Cannot connect to the Docker daemon")  # noqa: E731
    with pytest.raises(IsolationUnavailable, match="daemon is not reachable"):
        ContainerExecutor(POLICY, run=down).preflight()

    def no_image(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0 if cmd[1] == "version" else 1, "29.0" if cmd[1] == "version" else "", "")

    with pytest.raises(IsolationUnavailable, match="not present locally"):
        ContainerExecutor(POLICY, run=no_image).preflight()


def _fake_runtime(probe_report):
    digest = POLICY.image.split("@")[1]

    def run(cmd, **kw):
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "29.0\n", "")
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, json.dumps([f"python@{digest}"]), "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps(probe_report) + "\n", "")

    return run


GOOD_PROBE = {
    "env": {"PATH": "/usr/local/bin:/usr/bin", "HOME": "/", "HTTP_PROXY": "", "no_proxy": "", "GPG_KEY": "7169605F"},
    "uid": 65534,
    "gid": 65534,
    "interfaces": ["lo"],
    "root_writable": False,
    "work_writable": False,
    "root_home_readable": False,
    "mounts": ["/", "/dev", "/dev/shm", "/etc/hosts", "/proc", "/proc/sys", "/sys", "/sys/fs/cgroup", "/tmp", "/work"],
}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"interfaces": ["eth0", "lo"]}, "network interfaces"),
        ({"uid": 0, "gid": 0}, "runs as 0:0"),
        ({"root_writable": True}, "root file system is writable"),
        ({"work_writable": True}, "task directory is writable"),
        ({"mounts": GOOD_PROBE["mounts"] + ["/home/user/.aws"]}, "unexpected mounts"),
        ({"env": {**GOOD_PROBE["env"], "HTTPS_PROXY": "http://user:secret@proxy:3128"}}, "proxy variables"),
        ({"env": {**GOOD_PROBE["env"], "HF_TOKEN": "hf_x"}}, "credential-like"),
    ],
)
def test_probe_must_observe_the_required_isolation(monkeypatch, change, message) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/" + name)
    assert check_probe(GOOD_PROBE, POLICY) == []
    with pytest.raises(IsolationUnavailable, match=message):
        ContainerExecutor(POLICY, run=_fake_runtime({**GOOD_PROBE, **change})).preflight()


def test_container_command_enforces_the_policy(tmp_path) -> None:
    args = ContainerExecutor(POLICY).container_args(tmp_path, "qsa-test")
    joined = " ".join(args)
    for flag in (
        "--network none",
        "--log-driver none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--pull never",
        f"type=bind,source={tmp_path},target=/work,readonly",
    ):
        assert flag in joined
    assert args[-1] == POLICY.image and "@sha256:" in POLICY.image
    assert not any(a in {"-e", "--env-file", "--privileged", "-v", "--volume"} for a in args)
    # the only --env flags blank the proxy variables the Docker CLI would otherwise inject
    env_values = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert env_values and all(v.endswith("=") and v[:-1] in sandbox.PROXY_VARIABLES for v in env_values)
    assert {v[:-1] for v in env_values} == set(sandbox.PROXY_VARIABLES)


def test_runtime_start_failure_is_an_infrastructure_error_not_a_task_result(monkeypatch) -> None:
    class Process:
        def __init__(self, *a, **kw):
            import io

            self.stdin, self.stdout, self.returncode = io.BytesIO(), io.BytesIO(b""), 125

        def wait(self, timeout=None):
            return 125

    executor = ContainerExecutor(POLICY, popen=Process)
    executor._ready = True
    with pytest.raises(IsolationUnavailable, match="exit status 125"):
        executor.execute("x = 1", "assert x == 1")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"image": "python:3.12-slim"}, "pinned by digest"),
        ({"user": "0:0"}, "non-root"),
        ({"network": "bridge"}, "network must be 'none'"),
        ({"runtime": "chroot"}, "runtime must be one of"),
    ],
)
def test_policy_rejects_weak_settings(tmp_path, change, message) -> None:
    import yaml

    document = yaml.safe_load((ROOT / "configs" / "eval" / "coding_sandbox.yaml").read_text())
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({**document, **change}))
    with pytest.raises(ValueError, match=message):
        load_policy(path)


def test_run_model_stops_before_loading_the_model_without_isolation(monkeypatch, tmp_path, capsys) -> None:
    from sentinel.eval import run_model

    def unavailable(self):
        raise IsolationUnavailable("docker daemon is not reachable")

    monkeypatch.setattr(CodingV1, "preflight", unavailable)
    monkeypatch.setattr(run_model, "HFBackend", lambda *a, **kw: pytest.fail("model must not be loaded"))
    out = tmp_path / "run.json"
    code = run_model.main(
        ["--bakeoff", "configs/model/bakeoff_qwen35.yaml", "--candidate", "qwen35-9b", "--output", str(out)]
    )
    assert code == 3 and not out.exists()
    assert "--skip-coding-regression" in capsys.readouterr().err


def test_skip_flag_contradicts_an_explicit_suite(tmp_path) -> None:
    from sentinel.eval import run_model

    out = tmp_path / "run.json"
    base = ["--backend", "baseline", "--cases", "eval/sentinelbench/quickeval_v1.jsonl", "--output", str(out)]
    with pytest.raises(SystemExit):
        run_model.main(base + ["--coding-suite", "coding-v1", "--skip-coding-regression"])
    assert run_model.main(base) == 0
    assert json.loads(out.read_text())["coding_regression"] == {"status": "not_run", "reason": "no coding-regression suite selected"}


def test_bakeoff_skip_records_the_reason(monkeypatch, tmp_path) -> None:
    from sentinel.eval import run_model
    from sentinel.eval.backends import BaselineBackend

    monkeypatch.setattr(CodingV1, "preflight", lambda self: pytest.fail("skipped suites need no sandbox"))
    monkeypatch.setattr(run_model, "HFBackend", lambda *a, **kw: BaselineBackend())
    out = tmp_path / "run.json"
    code = run_model.main(
        ["--bakeoff", "configs/model/bakeoff_qwen35.yaml", "--candidate", "qwen35-9b", "--skip-coding-regression", "--output", str(out)]
    )
    assert code == 0
    coding = json.loads(out.read_text())["coding_regression"]
    assert coding["status"] == "not_run" and "--skip-coding-regression" in coding["reason"]


# --- execution behaviour (local: trusted code only; container: the real boundary) ---------------


@pytest.mark.parametrize("kind", EXECUTORS)
def test_successful_early_exit_cannot_fake_a_pass(kind) -> None:
    executor = _executor(kind)
    assert executor.execute("import sys\nsys.exit(0)\n", "assert True") == (False, "tests_not_completed")
    assert executor.execute("import os\nos._exit(0)\n", "assert True") == (False, "tests_not_completed")


@pytest.mark.parametrize("kind", EXECUTORS)
def test_generated_output_is_bounded_and_does_not_hide_the_result(kind) -> None:
    code = "print('x' * 2_000_000)\ndef answer():\n    return 42\n"
    assert _executor(kind).execute(code, "assert answer() == 42") == (True, None)


@pytest.mark.parametrize("kind", EXECUTORS)
def test_runaway_code_is_stopped(kind) -> None:
    ok, failure = _executor(kind).execute("while True:\n    pass\n", "assert True")
    assert ok is False and failure in {"timeout", "resource_limit"}


@CONTAINER
def test_container_has_no_network_host_files_or_host_environment(tmp_path, monkeypatch) -> None:
    secret = tmp_path / "host-secret.txt"
    secret.write_text("do-not-read")
    monkeypatch.setenv("QSA_FAKE_TOKEN", "do-not-leak")
    executor = ContainerExecutor(dataclasses.replace(POLICY, timeout_seconds=15))
    probe = (
        "import os, socket\n"
        "def isolated():\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "        return 'network'\n"
        "    except OSError:\n"
        "        pass\n"
        f"    if os.path.exists({str(secret)!r}):\n"
        "        return 'host file'\n"
        "    if 'QSA_FAKE_TOKEN' in os.environ:\n"
        "        return 'host env'\n"
        "    if os.getuid() == 0:\n"
        "        return 'root'\n"
        "    return 'ok'\n"
    )
    assert executor.execute(probe, "assert isolated() == 'ok', isolated()") == (True, None)


@pytest.mark.parametrize("kind", EXECUTORS)
def test_generated_code_exiting_125_to_127_is_a_task_failure(kind) -> None:
    # review finding: these statuses are also Docker's own start-up errors
    for status in (125, 126, 127):
        assert _executor(kind).execute(f"import sys\nsys.exit({status})\n", "assert True") == (False, f"exit_{status}")


@CONTAINER
def test_client_proxy_configuration_does_not_reach_generated_code(tmp_path, monkeypatch) -> None:
    config = tmp_path / "docker-config"
    config.mkdir()
    (config / "config.json").write_text(json.dumps({"proxies": {"default": {"httpProxy": "http://user:secret@proxy.example:3128"}}}))
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    executor = ContainerExecutor(dataclasses.replace(POLICY, timeout_seconds=15))
    executor.preflight()
    code = "import os\ndef leaked():\n    return sorted(k for k, v in os.environ.items() if 'proxy' in k.lower() and v)\n"
    assert executor.execute(code, "assert leaked() == [], leaked()") == (True, None)


@CONTAINER
def test_reference_solutions_pass_in_the_container() -> None:
    suite = CodingV1(executor=ContainerExecutor())
    result = suite.run(ScriptedCoder(lambda t: f"```python\n{t['canonical_solution']}```")).as_dict()
    assert result["passed"] == result["tasks"]
    assert result["isolation"]["executor"] == "container" and result["isolation"]["policy"]["network"] == "none"
