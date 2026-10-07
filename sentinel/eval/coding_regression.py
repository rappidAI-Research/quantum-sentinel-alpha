"""General coding-regression check for the 4B vs 9B bake-off.

`coding-v1` is a small clean-room Python suite (eval/coding/coding_v1.jsonl): each task gives a
signature + docstring, the model writes the function, hidden assert-tests decide pass/fail.

Model-generated code is executed. It runs in a separate `python -I` process with CPU, memory,
file-size and wall-clock limits, an empty environment and a throwaway working directory. That is
containment, not a sandbox: run the suite only on a disposable evaluation host (see
docs/THREAT_MODEL.md, docs/BAKEOFF.md).
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from sentinel.eval.backends import ModelBackend
from sentinel.eval.io import PROJECT_ROOT, load_jsonl
from sentinel.eval.parse import strip_reasoning

CODING_V1_PATH = PROJECT_ROOT / "eval" / "coding" / "coding_v1.jsonl"
TASK_KEYS = {"task_id", "entry_point", "prompt", "tests", "canonical_solution"}
TIMEOUT_SECONDS = 10
MAX_STORED_CODE_CHARS = 4000
CODING_RUNNER_VERSION = "2"

CODING_SYSTEM_PROMPT = """You are a careful Python programmer.
Implement the requested function or class exactly as specified, using only the Python standard library.
Reply with the complete implementation (including any imports and helpers) in a single ```python code block and nothing else."""

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def coding_contract_sha256() -> str:
    payload = "\\x00".join(
        (
            CODING_RUNNER_VERSION,
            CODING_SYSTEM_PROMPT,
            _CODE_BLOCK.pattern,
            "separate-solution-hidden-tests;completion-marker;stdout-stderr-devnull",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CodingRegressionResult:
    suite: str
    suite_version: str
    suite_sha256: str
    runner_version: str
    contract_sha256: str
    tasks: int
    passed: int
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = "completed"
        result["pass_rate"] = self.passed / self.tasks if self.tasks else None
        return result


class CodingRegressionSuite(Protocol):
    name: str
    version: str

    def run(self, backend: ModelBackend) -> CodingRegressionResult: ...


def not_run() -> dict[str, Any]:
    return {"status": "not_run", "reason": "no coding-regression suite selected"}


def load_tasks(path: Path = CODING_V1_PATH) -> list[dict[str, Any]]:
    tasks = load_jsonl(path)
    seen: set[str] = set()
    for task in tasks:
        if set(task) != TASK_KEYS or not all(isinstance(task[key], str) and task[key] for key in TASK_KEYS):
            raise ValueError(f"coding task must have exactly the non-empty string fields {sorted(TASK_KEYS)}")
        if task["task_id"] in seen:
            raise ValueError(f"duplicate coding task_id: {task['task_id']}")
        seen.add(task["task_id"])
    return tasks


def tasks_sha256(tasks: list[dict[str, Any]]) -> str:
    canonical = sorted(json.dumps(task, sort_keys=True, ensure_ascii=False) for task in tasks)
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def extract_code(text: str) -> str | None:
    visible, truncated = strip_reasoning(text)
    if truncated:
        return None
    block = _CODE_BLOCK.search(visible)
    return (block.group(1) if block else visible).strip() or None


def _limit_resources() -> None:  # pragma: no cover - runs in the child process
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SECONDS, TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def execute(code: str, tests: str) -> tuple[bool, str | None]:
    """Run generated code and hidden tests in a contained child process.

    The generated solution and hidden-test controller live in separate files. A zero
    process exit is not enough to pass: the controller must write a random completion
    marker only after all hidden tests finish. stdout/stderr go to DEVNULL so untrusted
    code cannot make the parent buffer unbounded output.
    """
    with tempfile.TemporaryDirectory(prefix="qsa-coding-") as workdir:
        solution = Path(workdir) / "solution.py"
        runner = Path(workdir) / "runner.py"
        marker = Path(workdir) / f".complete-{secrets.token_hex(16)}"
        solution.write_text(code + "\n", encoding="utf-8")
        runner.write_text(
            "import importlib.util\n"
            "from pathlib import Path\n"
            f"_solution_path = {str(solution)!r}\n"
            f"_marker_path = {str(marker)!r}\n"
            f"_tests = {tests!r}\n"
            "_spec = importlib.util.spec_from_file_location('_qsa_solution', _solution_path)\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "assert _spec.loader is not None\n"
            "_spec.loader.exec_module(_module)\n"
            "_namespace = vars(_module)\n"
            "exec(_tests, _namespace, _namespace)\n"
            "Path(_marker_path).write_text('complete', encoding='utf-8')\n",
            encoding="utf-8",
        )
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(runner)],
                cwd=workdir,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TIMEOUT_SECONDS,
                preexec_fn=_limit_resources if sys.platform != "win32" else None,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"

        if done.returncode < 0:
            return False, "resource_limit"
        if done.returncode != 0:
            return False, f"exit_{done.returncode}"
        if not marker.is_file() or marker.read_text(encoding="utf-8") != "complete":
            return False, "tests_not_completed"
        return True, None


class CodingV1:
    name = "coding-v1"
    version = "1"

    def __init__(self, path: Path = CODING_V1_PATH):
        self.tasks = load_tasks(path)

    def run(self, backend: ModelBackend) -> CodingRegressionResult:
        details: dict[str, Any] = {}
        passed = 0
        started = time.perf_counter()
        for task in self.tasks:
            messages = [
                {"role": "system", "content": CODING_SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ]
            result = backend.generate(messages)
            code = extract_code(result.text)
            ok, failure = execute(code, task["tests"]) if code else (False, "no code in output")
            passed += int(ok)
            details[task["task_id"]] = {
                "passed": ok,
                "failure": failure,
                "generated_tokens": result.generated_tokens,
                "code": (code or "")[:MAX_STORED_CODE_CHARS],
            }
        return CodingRegressionResult(
            suite=self.name,
            suite_version=self.version,
            suite_sha256=tasks_sha256(self.tasks),
            runner_version=CODING_RUNNER_VERSION,
            contract_sha256=coding_contract_sha256(),
            tasks=len(self.tasks),
            passed=passed,
            details={"seconds": time.perf_counter() - started, "tasks": details},
        )


SUITES: dict[str, Callable[[], CodingRegressionSuite]] = {CodingV1.name: CodingV1}
