import importlib.util
import re
from pathlib import Path

from sentinel.eval import coding_regression
from sentinel.eval.backends import GenerationResult
from sentinel.eval.coding_regression import CodingV1, extract_code, load_tasks

ROOT = Path(__file__).resolve().parents[1]
TASKS = {task["prompt"]: task for task in load_tasks()}


class ScriptedCoder:
    name = "fake"

    def __init__(self, answer):
        self.answer = answer

    def generate(self, messages):
        return GenerationResult(text=self.answer(TASKS[messages[-1]["content"]]), generated_tokens=50)


def test_reference_solutions_pass_and_wrong_code_fails() -> None:
    perfect = CodingV1().run(ScriptedCoder(lambda t: f"```python\n{t['canonical_solution']}```")).as_dict()
    assert perfect["status"] == "completed"
    assert perfect["passed"] == perfect["tasks"] == len(TASKS) >= 12
    assert len(perfect["suite_sha256"]) == 64

    broken = CodingV1().run(ScriptedCoder(lambda t: "```python\ndef nothing():\n    return None\n```")).as_dict()
    assert broken["passed"] == 0
    assert all(not d["passed"] for d in broken["details"]["tasks"].values())


def test_runaway_code_is_stopped(monkeypatch) -> None:
    monkeypatch.setattr(coding_regression, "TIMEOUT_SECONDS", 2)
    hang = CodingV1().run(ScriptedCoder(lambda t: "```python\nwhile True:\n    pass\n```" if t["task_id"] == "rle-encode" else ""))
    result = hang.details["tasks"]["rle-encode"]
    assert result["passed"] is False and result["failure"] in {"timeout", "resource_limit"}


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


def test_successful_early_exit_cannot_fake_a_pass() -> None:
    ok, failure = coding_regression.execute("import sys\nsys.exit(0)\n", "assert True")
    assert ok is False and failure == "tests_not_completed"

    ok, failure = coding_regression.execute("import os\nos._exit(0)\n", "assert True")
    assert ok is False and failure == "tests_not_completed"


def test_generated_output_is_not_buffered_by_the_parent() -> None:
    code = "print('x' * 2_000_000)\ndef answer():\n    return 42\n"
    ok, failure = coding_regression.execute(code, "assert answer() == 42")
    assert ok is True and failure is None
