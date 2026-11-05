"""General coding-regression check for the 4B vs 9B bake-off.

`coding-v1` is a small clean-room Python suite (eval/coding/coding_v1.jsonl): each task gives a
signature + docstring, the model writes the function, hidden assert-tests decide pass/fail.

Model-generated code is executed, and only through an isolated executor
(`sentinel.eval.sandbox.ContainerExecutor`: a disposable, network-less, credential-free container
from a digest-pinned image). `CodingV1.run` refuses a non-isolated executor and the container
executor fails closed when isolation cannot be established (docs/CODING_SANDBOX.md).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from sentinel.eval.backends import ModelBackend
from sentinel.eval.io import PROJECT_ROOT, load_jsonl
from sentinel.eval.parse import strip_reasoning
from sentinel.eval.sandbox import ContainerExecutor, Executor, IsolationUnavailable

CODING_V1_PATH = PROJECT_ROOT / "eval" / "coding" / "coding_v1.jsonl"
TASK_KEYS = {"task_id", "entry_point", "prompt", "tests", "canonical_solution"}
MAX_STORED_CODE_CHARS = 4000
CODING_RUNNER_VERSION = "3"  # 3: isolated container executor; marker on stdin; hidden tests in their own file

CODING_SYSTEM_PROMPT = """You are a careful Python programmer.
Implement the requested function or class exactly as specified, using only the Python standard library.
Reply with the complete implementation (including any imports and helpers) in a single ```python code block and nothing else."""

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def coding_contract_sha256(executor: dict[str, Any] | None = None) -> str:
    """Hash of everything that decides a coding result, including the execution policy."""
    payload = "\\x00".join(
        (
            CODING_RUNNER_VERSION,
            CODING_SYSTEM_PROMPT,
            _CODE_BLOCK.pattern,
            "separate-solution-hidden-tests;stdin-completion-marker;bounded-stdout",
            json.dumps(executor or {}, sort_keys=True),
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
    isolation: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = "completed"
        result["pass_rate"] = self.passed / self.tasks if self.tasks else None
        return result


class CodingRegressionSuite(Protocol):
    name: str
    version: str

    def preflight(self) -> None: ...
    def run(self, backend: ModelBackend) -> CodingRegressionResult: ...


def not_run(reason: str = "no coding-regression suite selected") -> dict[str, Any]:
    return {"status": "not_run", "reason": reason}


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


class CodingV1:
    name = "coding-v1"
    version = "1"

    def __init__(self, path: Path = CODING_V1_PATH, executor: Executor | None = None):
        self.tasks = load_tasks(path)
        self.executor: Executor = executor if executor is not None else ContainerExecutor()

    def preflight(self) -> None:
        """Fail closed before any model work when generated code could not run isolated."""
        if not getattr(self.executor, "isolated", False):
            raise IsolationUnavailable(
                f"{type(self.executor).__name__} is not an isolated executor; model-generated code only runs in the container sandbox"
            )
        self.executor.preflight()

    def run(self, backend: ModelBackend) -> CodingRegressionResult:
        self.preflight()
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
            ok, failure = self.executor.execute(code, task["tests"]) if code else (False, "no code in output")
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
            contract_sha256=coding_contract_sha256(self.executor.describe()),
            tasks=len(self.tasks),
            passed=passed,
            isolation=self.executor.describe(),
            details={"seconds": time.perf_counter() - started, "tasks": details},
        )


SUITES: dict[str, Callable[[], CodingRegressionSuite]] = {CodingV1.name: CodingV1}
