from pathlib import Path

import pytest
import yaml

from sentinel.eval import compare, run_model, runner
from sentinel.eval.bakeoff import load_bakeoff

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "model" / "bakeoff_qwen35.yaml"


def write_config(tmp_path: Path, mutate) -> Path:
    document = yaml.safe_load(CONFIG.read_text())
    mutate(document)
    path = tmp_path / "bakeoff.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_shipped_bakeoff_compares_qwen35_4b_and_9b_under_one_setup() -> None:
    config = load_bakeoff(CONFIG)
    models = {c.model_id for c in config.candidates.values()}
    assert models == {"Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}
    runtimes = {c.runtime for c in config.candidates.values()}
    assert len(runtimes) == 1, "candidates must share runtime unless an override is justified"
    assert config.generation.do_sample is False and config.generation.enable_thinking is False
    assert config.coding_suite == "coding-v1"


def test_bakeoff_rejects_per_candidate_generation_and_unjustified_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        load_bakeoff(write_config(tmp_path, lambda d: d["candidates"][0].update(generation={"max_new_tokens": 2048})))
    with pytest.raises(ValueError, match="override_reason"):
        load_bakeoff(write_config(tmp_path, lambda d: d["candidates"][1].update(runtime_overrides={"quantization": "bnb-4bit"})))
    with pytest.raises(ValueError, match="40-character"):
        load_bakeoff(write_config(tmp_path, lambda d: d["candidates"][0].update(revision="main")))


def test_cli_refuses_to_change_shared_setup_or_run_unpinned(tmp_path: Path) -> None:
    out = str(tmp_path / "run.json")
    with pytest.raises(SystemExit):
        run_model.main(["--bakeoff", str(CONFIG), "--candidate", "qwen35-4b", "--max-new-tokens", "2048", "--output", out])
    with pytest.raises(SystemExit):
        run_model.main(["--bakeoff", str(CONFIG), "--candidate", "qwen35-4b", "--coding-suite", "coding-v1", "--output", out])
    with pytest.raises(SystemExit):
        run_model.main(["--bakeoff", str(CONFIG), "--candidate", "qwen35-9b", "--output", out])


def _baseline_run(tmp_path: Path, name: str, *extra: str) -> Path:
    out = tmp_path / f"{name}.json"
    assert run_model.main(["--bakeoff", str(CONFIG), "--candidate", "qwen35-4b", "--backend", "baseline", "--output", str(out), *extra]) == 0
    return out


def test_compare_reports_side_by_side_without_a_winner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _baseline_run(tmp_path, "a")
    b = _baseline_run(tmp_path, "b")
    report = compare.build_report(runner.load_artifact(a), runner.load_artifact(b))
    assert report["comparable"] is True
    assert {row["metric"] for row in report["rows"]} >= {"precision", "false positive rate", "peak VRAM (GiB)", "tokens/s"}
    assert not any("winner" in key or "score" in key for key in report)

    assert compare.main([str(a), str(b)]) == 0
    assert "No winner is computed" in capsys.readouterr().out


def test_compare_flags_different_case_sets(tmp_path: Path) -> None:
    a = _baseline_run(tmp_path, "full")
    b = _baseline_run(tmp_path, "subset", "--limit", "10")
    assert compare.main([str(a), str(b)]) == 2
