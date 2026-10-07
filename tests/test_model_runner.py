import json
from copy import deepcopy
from pathlib import Path

import pytest

from sentinel.eval import runner
from sentinel.eval.backends import GenerationResult, GenerationSettings, RuntimeSettings
from sentinel.eval.io import load_jsonl
from sentinel.eval.prompt import build_messages
from sentinel.eval.runner import run_quickeval, validate_run_artifact

ROOT = Path(__file__).resolve().parents[1]
CASES = load_jsonl(ROOT / "eval" / "sentinelbench" / "quickeval_v1.jsonl")
PINNED = "0123456789abcdef0123456789abcdef01234567"


class FakeBackend:
    """Stands in for a real model: answers from a script keyed by file + first code line."""

    name = "hf"

    def __init__(
        self,
        answers: dict[str, str],
        default: str = '{"decision": "no_finding"}',
        resolved: str | None = PINNED,
        loaded_dtype: str = "bfloat16",
    ):
        self.answers = answers
        self.default = default
        self.resolved = resolved
        self.loaded_dtype = loaded_dtype
        self.loaded = False
        self.seen: list[list[dict[str, str]]] = []

    def load(self) -> None:
        self.loaded = True

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        assert self.loaded
        self.seen.append(messages)
        for marker, answer in self.answers.items():
            if marker in messages[-1]["content"]:
                return GenerationResult(text=answer, generated_tokens=20, prompt_tokens=300)
        return GenerationResult(text=self.default, generated_tokens=10, prompt_tokens=300)

    def describe(self) -> dict:
        return {"backend": self.name, "resolved_revision": self.resolved, "loaded_dtype": self.loaded_dtype}

    def peak_vram_gib(self) -> float | None:
        return 9.5


def run(backend: FakeBackend, cases: list[dict] = CASES, revision: str | None = PINNED, **kwargs) -> dict:
    return run_quickeval(
        cases,
        backend,
        model_id="Qwen/Qwen3.5-4B",
        revision=revision,
        generation=GenerationSettings(),
        runtime=RuntimeSettings(),
        cases_path="eval/sentinelbench/quickeval_v1.jsonl",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "f" * 40, "dirty": False})


def test_fake_model_run_produces_valid_complete_artifact() -> None:
    sqli = next(c for c in CASES if c["case_id"] == "qe-py-sqli-fstring")
    truth = sqli["ground_truth"]
    answer = json.dumps(
        {"decision": "finding", "cwe": "CWE-89", "start_line": truth["start_line"], "end_line": truth["end_line"],
         "confidence": 0.9, "evidence": "status flows into an f-string SQL query."}
    )
    backend = FakeBackend({"status = '{status}'": answer, "Pipeline.from_dict": "not json at all"})
    artifact = run(backend)

    validate_run_artifact(json.loads(json.dumps(artifact)))
    assert artifact["reportable"] is True and artifact["reportable_blockers"] == []
    assert artifact["cases"]["count"] == len(CASES) == len(artifact["predictions"])
    assert artifact["metrics"]["true_positive"] == 1
    # Both the unsafe yaml.load and the safe_load neighbour share the marker; neither may be repaired.
    assert artifact["metrics"]["invalid_predictions"] == 2
    efficiency = artifact["efficiency"]
    assert efficiency["generated_tokens"] > 0 and efficiency["tokens_per_second"] > 0
    assert efficiency["peak_vram_gib"] == 9.5
    assert artifact["coding_regression"]["status"] == "not_run"


def test_unpinned_or_mismatched_revision_is_not_reportable() -> None:
    artifact = run(FakeBackend({}, resolved=None), revision="main")
    assert artifact["reportable"] is False
    assert any("immutable" in blocker for blocker in artifact["reportable_blockers"])

    mismatch = run(FakeBackend({}, resolved="a" * 40))
    assert any("differs from requested" in blocker for blocker in mismatch["reportable_blockers"])


def test_loaded_dtype_must_match_requested_dtype() -> None:
    artifact = run(FakeBackend({}, loaded_dtype="float32"))
    assert artifact["reportable"] is False
    assert any("dtype float32 differs from requested bfloat16" in b for b in artifact["reportable_blockers"])


def test_subset_runs_are_not_reportable() -> None:
    artifact = run(FakeBackend({}), cases=CASES[:5], full_case_set=False)
    assert artifact["reportable"] is False


def test_prompt_never_contains_ground_truth() -> None:
    case = next(c for c in CASES if c["case_type"] == "vulnerable")
    altered = deepcopy(case)
    altered.update(expected_decision="no_finding", case_type="hard_negative", case_id="something-else")
    altered["ground_truth"] = {"cwe": "CWE-1", "start_line": 1, "end_line": 1}
    assert build_messages(case) == build_messages(altered)
