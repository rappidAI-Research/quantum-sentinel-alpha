"""Execution boundary for model-generated code (coding-regression suites).

Model-generated code is untrusted. It only runs through an executor whose `isolated` attribute is
True; `CodingV1.run` refuses any other executor, so untrusted code can never run silently without
isolation.

- `ContainerExecutor` runs each task in a disposable container (docker or podman) started from a
  digest-pinned image with no network, no host environment or credentials, a read-only root file
  system, an unprivileged user, dropped capabilities, `no-new-privileges`, and CPU, memory, process
  and file-size limits. The only host path it sees is the per-task directory, mounted read-only.
  `preflight()` checks the runtime, the daemon and the pinned image, then runs a probe container
  that must observe exactly that isolation; any failure raises `IsolationUnavailable` (fail closed).
- `LocalProcessExecutor` is the old resource-limited child process. It is NOT isolated (same user,
  same file system, same network) and exists only to execute trusted, repository-authored code such
  as the reference solutions in the test suite.

Isolation reduces what generated code can reach; it does not make arbitrary code safe to run on a
host that holds valuable data. Use a disposable evaluation host as well (docs/CODING_SANDBOX.md).
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from sentinel.eval.io import PROJECT_ROOT

DEFAULT_POLICY_PATH = PROJECT_ROOT / "configs" / "eval" / "coding_sandbox.yaml"
RUNTIMES = ("docker", "podman")
_DIGEST_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
_USER = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")  # never root (uid/gid 0)
POLICY_KEYS = {
    "schema_version",
    "runtime",
    "image",
    "user",
    "network",
    "cpus",
    "memory_mib",
    "pids_limit",
    "tmpfs_mib",
    "file_size_mib",
    "timeout_seconds",
    "max_output_kib",
}
# Mount points a container may have: the runtime's own pseudo file systems and per-container files,
# the tmpfs and the read-only task directory. Anything else is an extra host mount.
_ALLOWED_MOUNTS = {"/", "/tmp", "/work", "/dev", "/proc", "/sys", "/etc/hostname", "/etc/hosts", "/etc/resolv.conf"}
_SYSTEM_MOUNT_PREFIXES = ("/dev/", "/proc/", "/sys/")
# Exit statuses of `docker run`/`podman run` itself (the container never ran the code).
_RUNTIME_ERRORS = {125, 126, 127}
_TAIL_BYTES = 4096

# Runs inside the container. The completion marker arrives on stdin before any generated code is
# imported and is printed only after every hidden test passed, so an early `sys.exit(0)` or
# `os._exit(0)` in the solution cannot fake a pass.
RUNNER = """import importlib.util, sys
marker = sys.stdin.readline().strip()
spec = importlib.util.spec_from_file_location('_qsa_solution', '/work/solution.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
namespace = vars(module)
exec(compile(open('/work/tests.py', encoding='utf-8').read(), '/work/tests.py', 'exec'), namespace, namespace)
sys.stdout.flush()
sys.stdout.write('\\n' + marker + '\\n')
sys.stdout.flush()
"""

# Must hold inside every container before any generated code runs (checked by preflight).
PROBE = """import json, os
interfaces = [line.split(':')[0].strip() for line in open('/proc/net/dev').read().splitlines()[2:] if ':' in line]
try:
    open('/probe-write', 'w').close()
    root_writable = True
except OSError:
    root_writable = False
try:
    open('/work/probe-write', 'w').close()
    work_writable = True
except OSError:
    work_writable = False
try:
    os.listdir('/root')
    root_home_readable = True
except OSError:
    root_home_readable = False
print(json.dumps({
    'uid': os.getuid(),
    'gid': os.getgid(),
    'interfaces': sorted(interfaces),
    'root_writable': root_writable,
    'work_writable': work_writable,
    'root_home_readable': root_home_readable,
    'mounts': sorted(line.split()[1] for line in open('/proc/self/mounts')),
}))
"""


class IsolationUnavailable(RuntimeError):
    """The required execution isolation cannot be established; untrusted code must not run."""


@dataclass(frozen=True)
class SandboxPolicy:
    runtime: str
    image: str
    user: str
    network: str
    cpus: float
    memory_mib: int
    pids_limit: int
    tmpfs_mib: int
    file_size_mib: int
    timeout_seconds: int
    max_output_kib: int

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> SandboxPolicy:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != POLICY_KEYS:
        raise ValueError(f"{path}: sandbox policy must have exactly the keys {sorted(POLICY_KEYS)}")
    if document["schema_version"] != "1.0":
        raise ValueError(f"{path}: unsupported schema_version {document['schema_version']!r}")
    values = {k: v for k, v in document.items() if k != "schema_version"}
    policy = SandboxPolicy(**values)
    if policy.runtime not in RUNTIMES:
        raise ValueError(f"{path}: runtime must be one of {RUNTIMES}")
    if not _DIGEST_IMAGE.match(policy.image):
        raise ValueError(f"{path}: image must be pinned by digest (name@sha256:<64 hex>), got {policy.image!r}")
    if not _USER.match(str(policy.user)):
        raise ValueError(f"{path}: user must be a non-root uid:gid, got {policy.user!r}")
    if policy.network != "none":
        raise ValueError(f"{path}: network must be 'none'")
    for key in ("memory_mib", "pids_limit", "tmpfs_mib", "file_size_mib", "timeout_seconds", "max_output_kib"):
        value = getattr(policy, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
    if not isinstance(policy.cpus, (int, float)) or isinstance(policy.cpus, bool) or policy.cpus <= 0:
        raise ValueError(f"{path}: cpus must be a positive number")
    return policy


class Executor(Protocol):
    isolated: bool

    def preflight(self) -> None: ...
    def execute(self, code: str, tests: str) -> tuple[bool, str | None]: ...
    def describe(self) -> dict[str, Any]: ...


Run = Callable[..., subprocess.CompletedProcess]


def _drain(stream: Any, keep: int, sink: dict[str, Any]) -> None:
    """Read a pipe to EOF, keeping only the last `keep` bytes, so output size is bounded."""
    tail = b""
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        total += len(chunk)
        tail = (tail + chunk)[-keep:]
    sink["tail"], sink["bytes"] = tail, total


class ContainerExecutor:
    """Disposable, network-less container per task (fail closed when it cannot be established)."""

    isolated = True

    def __init__(self, policy: SandboxPolicy | None = None, run: Run = subprocess.run, popen: Callable[..., Any] = subprocess.Popen):
        self.policy = policy or load_policy()
        self._run = run
        self._popen = popen
        self._ready = False

    # -- isolation checks ------------------------------------------------------------------------

    def _cli(self, *args: str, timeout: int = 60, stdin: str | None = None) -> subprocess.CompletedProcess:
        try:
            return self._run(
                [self.policy.runtime, *args], input=stdin, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise IsolationUnavailable(f"{self.policy.runtime} {args[0]} failed: {exc}") from exc

    def container_args(self, workdir: Path, name: str) -> list[str]:
        p = self.policy
        return [
            p.runtime,
            "run",
            "--rm",
            "--interactive",  # stdin carries the completion marker
            "--name",
            name,
            "--network",
            "none",
            # stdout is read through the attached pipe; the daemon must not also write it to a host-side
            # log file (json-file has no rotation by default), or a flood could fill the host disk
            "--log-driver",
            "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={p.tmpfs_mib}m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(p.pids_limit),
            "--memory",
            f"{p.memory_mib}m",
            "--memory-swap",
            f"{p.memory_mib}m",
            "--cpus",
            str(p.cpus),
            "--ulimit",
            f"fsize={p.file_size_mib * 1024 * 1024}",
            "--user",
            p.user,
            "--workdir",
            "/tmp",
            "--mount",
            f"type=bind,source={workdir},target=/work,readonly",
            "--pull",
            "never",
            p.image,
        ]

    def preflight(self) -> None:
        """Raise IsolationUnavailable unless a probe container shows exactly the required isolation."""
        if self._ready:
            return
        p = self.policy
        if shutil.which(p.runtime) is None:
            raise IsolationUnavailable(f"container runtime {p.runtime!r} is not installed")
        version = self._cli("version", "--format", "{{.Server.Version}}", timeout=30)
        if version.returncode != 0:
            raise IsolationUnavailable(f"{p.runtime} daemon is not reachable: {version.stderr.strip()[:200]}")
        image = self._cli("image", "inspect", "--format", "{{json .RepoDigests}}", p.image, timeout=30)
        digest = p.image.split("@", 1)[1]
        if image.returncode != 0 or digest not in image.stdout:
            raise IsolationUnavailable(f"pinned image {p.image} is not present locally; prepare the host with `{p.runtime} pull {p.image}`")
        with tempfile.TemporaryDirectory(prefix="qsa-probe-") as directory:
            workdir = Path(directory)
            workdir.chmod(0o755)
            command = self.container_args(workdir, f"qsa-probe-{secrets.token_hex(6)}") + ["python", "-I", "-c", PROBE]
            try:
                done = self._run(command, capture_output=True, text=True, timeout=60, check=False, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError) as exc:
                raise IsolationUnavailable(f"probe container failed: {exc}") from exc
        if done.returncode != 0:
            raise IsolationUnavailable(f"probe container failed (exit {done.returncode}): {done.stderr.strip()[:300]}")
        try:
            observed = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise IsolationUnavailable(f"probe container printed no report: {done.stdout[:200]!r}") from exc
        problems = check_probe(observed, p)
        if problems:
            raise IsolationUnavailable("container isolation is not as required: " + "; ".join(problems))
        self._ready = True

    # -- execution -------------------------------------------------------------------------------

    def execute(self, code: str, tests: str) -> tuple[bool, str | None]:
        if not self._ready:
            self.preflight()
        p = self.policy
        marker = f"qsa-complete-{secrets.token_hex(16)}"
        name = f"qsa-coding-{secrets.token_hex(8)}"
        with tempfile.TemporaryDirectory(prefix="qsa-coding-") as directory:
            workdir = Path(directory)
            workdir.chmod(0o755)  # readable by the unprivileged container user, never writable by it
            for filename, content in (("solution.py", code + "\n"), ("tests.py", tests + "\n"), ("runner.py", RUNNER)):
                (workdir / filename).write_text(content, encoding="utf-8")
                (workdir / filename).chmod(0o644)
            command = self.container_args(workdir, name) + ["python", "-I", "-B", "/work/runner.py"]
            try:
                process = self._popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except OSError as exc:
                raise IsolationUnavailable(f"cannot start {p.runtime}: {exc}") from exc
            sink: dict[str, Any] = {}
            reader = threading.Thread(target=_drain, args=(process.stdout, _TAIL_BYTES, sink), daemon=True)
            reader.start()
            try:
                process.stdin.write((marker + "\n").encode("utf-8"))
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                returncode = process.wait(timeout=p.timeout_seconds)
            except subprocess.TimeoutExpired:
                self._cli("kill", name, timeout=30)
                process.kill()
                process.wait()
                reader.join(timeout=10)
                return False, "timeout"
            reader.join(timeout=10)
        if returncode in _RUNTIME_ERRORS:
            # The runtime could not start the container: an infrastructure failure, never a task result.
            raise IsolationUnavailable(f"{p.runtime} run failed with exit status {returncode}")
        if returncode == 137:
            return False, "resource_limit"
        if returncode != 0:
            return False, f"exit_{returncode}"
        lines = sink.get("tail", b"").decode("utf-8", errors="replace").strip().splitlines()
        if not lines or lines[-1].strip() != marker:
            return False, "tests_not_completed"
        return True, None

    def describe(self) -> dict[str, Any]:
        return {"executor": "container", "isolated": True, "policy": asdict(self.policy), "policy_sha256": self.policy.sha256()}


def check_probe(observed: dict[str, Any], policy: SandboxPolicy) -> list[str]:
    """What the probe container saw versus what the policy requires."""
    uid, gid = (int(part) for part in policy.user.split(":"))
    problems = []
    if observed.get("uid") != uid or observed.get("gid") != gid:
        problems.append(f"runs as {observed.get('uid')}:{observed.get('gid')}, expected {policy.user}")
    if observed.get("interfaces") != ["lo"]:
        problems.append(f"network interfaces {observed.get('interfaces')} (expected only lo)")
    if observed.get("root_writable") is not False:
        problems.append("root file system is writable")
    if observed.get("work_writable") is not False:
        problems.append("the task directory is writable")
    if observed.get("root_home_readable") is not False:
        problems.append("/root is readable by the container user")
    mounts = observed.get("mounts")
    if not isinstance(mounts, list):
        problems.append("mount table not reported")
    else:
        unexpected = [m for m in mounts if m not in _ALLOWED_MOUNTS and not m.startswith(_SYSTEM_MOUNT_PREFIXES)]
        if unexpected:
            problems.append(f"unexpected mounts {unexpected[:5]} (only the read-only task directory may come from the host)")
    return problems


class LocalProcessExecutor:
    """Resource-limited child process. NOT isolated: only for trusted, repository-authored code."""

    isolated = False

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def preflight(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"executor": "local_process", "isolated": False}

    def execute(self, code: str, tests: str) -> tuple[bool, str | None]:
        timeout = self.timeout_seconds

        def limit() -> None:  # pragma: no cover - runs in the child process
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))

        marker = f"qsa-complete-{secrets.token_hex(16)}"
        with tempfile.TemporaryDirectory(prefix="qsa-coding-") as directory:
            workdir = Path(directory)
            (workdir / "solution.py").write_text(code + "\n", encoding="utf-8")
            (workdir / "tests.py").write_text(tests + "\n", encoding="utf-8")
            (workdir / "runner.py").write_text(RUNNER.replace("/work/", str(workdir) + "/"), encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", str(workdir / "runner.py")],
                cwd=workdir,
                env={},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                preexec_fn=limit if sys.platform != "win32" else None,
            )
            sink: dict[str, Any] = {}
            reader = threading.Thread(target=_drain, args=(process.stdout, _TAIL_BYTES, sink), daemon=True)
            reader.start()
            try:
                process.stdin.write((marker + "\n").encode("utf-8"))
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                reader.join(timeout=10)
                return False, "timeout"
            reader.join(timeout=10)
        if returncode < 0:
            return False, "resource_limit"
        if returncode != 0:
            return False, f"exit_{returncode}"
        lines = sink.get("tail", b"").decode("utf-8", errors="replace").strip().splitlines()
        if not lines or lines[-1].strip() != marker:
            return False, "tests_not_completed"
        return True, None
