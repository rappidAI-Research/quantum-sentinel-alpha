"""Extension point for general coding-regression evaluation in the 4B vs 9B bake-off.

No suite ships yet. A suite generates code through the same backend and must run any
generated code through an isolated execution layer (see docs/THREAT_MODEL.md), never in
the evaluation process itself. Register a factory in SUITES to make it selectable via
``run_model --coding-suite``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from sentinel.eval.backends import ModelBackend


@dataclass(frozen=True)
class CodingRegressionResult:
    suite: str
    suite_version: str
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


SUITES: dict[str, Callable[[], CodingRegressionSuite]] = {}


def not_run() -> dict[str, Any]:
    return {"status": "not_run", "reason": "no coding-regression suite selected"}
