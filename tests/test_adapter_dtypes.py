"""Foundation and adapter parameters are verified separately, by identity (no GPU needed)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from sentinel.eval.backends import adapter_file_tensors, verify_parameters
from tests.test_model_runner import FakeBackend, run

BF16_FOUNDATION = 8_953_803_264  # measured on the g6e run: every Ornith text parameter in bfloat16
LORA_PARAMETERS = 40_108_032  # the 200 rank-16 LoRA targets


class P:
    """Stand-in for a torch parameter."""

    def __init__(self, dtype: str, shape: list[int]):
        self.dtype = f"torch.{dtype}"
        self.shape = shape

    def numel(self) -> int:
        size = 1
        for dimension in self.shape:
            size *= dimension
        return size


def lora(module: str, rank: int = 4, fan_in: int = 8, fan_out: int = 16) -> dict[str, dict]:
    return {
        f"base_model.model.{module}.lora_A.weight": {"dtype": "float32", "shape": [rank, fan_in]},
        f"base_model.model.{module}.lora_B.weight": {"dtype": "float32", "shape": [fan_out, rank]},
    }


FILE = {**lora("model.layers.0.self_attn.q_proj"), **lora("model.layers.1.linear_attn.out_proj")}


def loaded(file: dict[str, dict], dtype: str | None = None) -> list[tuple[str, P]]:
    params = [
        ("base_model.model.model.embed_tokens.weight", P("bfloat16", [32, 8])),
        ("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight", P("bfloat16", [16, 8])),
        ("base_model.model.model.layers.1.linear_attn.out_proj.base_layer.weight", P("bfloat16", [16, 8])),
    ]
    for name, meta in file.items():
        peft_name = name.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default.")
        params.append((peft_name, P(dtype or meta["dtype"], meta["shape"])))
    return params


def test_bf16_foundation_and_fp32_adapter_verify_separately() -> None:
    result = verify_parameters(loaded(FILE), FILE)
    assert result["foundation_parameter_dtypes"] == {"bfloat16": 32 * 8 + 16 * 8 * 2}
    assert result["adapter_parameter_dtypes"] == {"float32": 2 * (4 * 8 + 16 * 4)}
    verification = result["adapter_verification"]
    assert verification["ok"] and verification["expected_tensors"] == verification["loaded_tensors"] == 4
    assert verification["expected_parameters"] == verification["loaded_parameters"] == 2 * (4 * 8 + 16 * 4)
    assert verification["file_dtypes"] == {"float32": 2 * (4 * 8 + 16 * 4)}


def test_without_adapter_everything_is_foundation() -> None:
    result = verify_parameters(loaded({}), None)
    assert set(result) == {"foundation_parameter_dtypes"}


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda f: {k: v for k, v in f.items() if "q_proj.lora_B" not in k}, "missing"),
        (lambda f: {**f, **lora("model.layers.2.mlp.up_proj")}, "unexpected"),
        (lambda f: {k: ({"dtype": "float32", "shape": [8, 8]} if "q_proj.lora_A" in k else v) for k, v in f.items()}, "shape_mismatches"),
    ],
)
def test_adapter_identity_mismatch_fails(change, field) -> None:
    # the adapter file is `FILE`; the loaded LoRA tensors differ from it
    verification = verify_parameters(loaded(change(FILE)), FILE)["adapter_verification"]
    assert not verification["ok"] and verification[field]


def test_adapter_dtype_must_be_the_file_dtype_or_peft_fp32_upcast() -> None:
    half_file = {k: {**v, "dtype": "bfloat16"} for k, v in FILE.items()}
    assert verify_parameters(loaded(half_file, "float32"), half_file)["adapter_verification"]["ok"]  # PEFT upcast
    verification = verify_parameters(loaded(FILE, "float16"), FILE)["adapter_verification"]
    assert not verification["ok"] and verification["dtype_mismatches"]


def test_a_non_lora_fp32_parameter_still_counts_against_the_foundation() -> None:
    params = loaded(FILE) + [("base_model.model.model.norm.weight", P("float32", [8]))]
    result = verify_parameters(params, FILE)
    assert result["foundation_parameter_dtypes"]["float32"] == 8 and result["adapter_verification"]["ok"]


def test_safetensors_header_is_read_without_loading_tensors(tmp_path: Path) -> None:
    header = {"__metadata__": {"format": "pt"}, "a.lora_A.weight": {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 128]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 128)
    assert adapter_file_tensors(path) == {"a.lora_A.weight": {"dtype": "float32", "shape": [4, 8]}}


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    from sentinel.eval import runner

    monkeypatch.setattr(runner, "git_state", lambda: {"commit": "f" * 40, "dirty": False})


class AdapterBackend(FakeBackend):
    def __init__(self, runtime_extra: dict, **kwargs):
        super().__init__({}, **kwargs)
        self.adapter_info = {"path": "runs/train/x/adapter", "weights_sha256": "a" * 64, "config_sha256": "b" * 64, "base_revision": None, "training_run": None}
        self.runtime_extra = runtime_extra

    def describe(self) -> dict:
        return {**super().describe(), **self.runtime_extra}


VERIFIED = {
    "ok": True,
    "expected_tensors": 400,
    "loaded_tensors": 400,
    "expected_parameters": LORA_PARAMETERS,
    "loaded_parameters": LORA_PARAMETERS,
}


def test_measured_adapter_run_is_reportable_with_separate_verification() -> None:
    combined = {"bfloat16": BF16_FOUNDATION, "float32": LORA_PARAMETERS}
    backend = AdapterBackend(
        {
            "foundation_parameter_dtypes": {"bfloat16": BF16_FOUNDATION},
            "adapter_parameter_dtypes": {"float32": LORA_PARAMETERS},
            "adapter_verification": VERIFIED,
        },
        parameter_dtypes=combined,
    )
    artifact = run(backend)
    assert artifact["reportable"] is True, artifact["reportable_blockers"]
    runtime = artifact["runtime"]
    assert runtime["parameter_dtypes"] == combined  # both distributions are recorded
    assert runtime["foundation_parameter_dtypes"] == {"bfloat16": BF16_FOUNDATION}
    assert runtime["adapter_parameter_dtypes"] == {"float32": LORA_PARAMETERS}


def test_adapter_run_without_identity_verification_is_not_reportable() -> None:
    # An artifact written before the split: only the combined distribution exists. Counts alone are
    # not trusted, so it stays non-reportable (as recorded), instead of being reinterpreted.
    backend = AdapterBackend({}, parameter_dtypes={"bfloat16": BF16_FOUNDATION, "float32": LORA_PARAMETERS})
    artifact = run(backend)
    assert artifact["reportable"] is False
    assert any("not verified against the adapter file" in b for b in artifact["reportable_blockers"])
    assert any("foundation parameter dtypes could not be confirmed" in b for b in artifact["reportable_blockers"])


def test_failed_adapter_verification_or_fp32_foundation_is_not_reportable() -> None:
    failed = AdapterBackend(
        {
            "foundation_parameter_dtypes": {"bfloat16": BF16_FOUNDATION},
            "adapter_verification": {**VERIFIED, "ok": False, "missing": ["x.lora_A.weight"]},
        }
    )
    assert any("do not match the adapter file" in b for b in run(failed)["reportable_blockers"])
    fp32_foundation = AdapterBackend(
        {"foundation_parameter_dtypes": {"bfloat16": BF16_FOUNDATION, "float32": 4096}, "adapter_verification": VERIFIED}
    )
    blockers = run(fp32_foundation)["reportable_blockers"]
    assert any("unexpected floating parameter dtypes ['float32']" in b for b in blockers)


def test_real_peft_adapter_verifies(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    text = {
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16, "vocab_size": 256,
        "max_position_embeddings": 4096, "mtp_num_hidden_layers": 0,
    }  # fmt: skip

    def base():
        torch.manual_seed(0)
        return transformers.AutoModelForCausalLM.from_config(transformers.Qwen3_5Config(text_config=text)).to(torch.bfloat16)

    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=r".*\.(q_proj|out_proj|gate_proj)$", task_type="CAUSAL_LM")
    peft.get_peft_model(base(), config).save_pretrained(tmp_path)
    model = peft.PeftModel.from_pretrained(base(), str(tmp_path), is_trainable=False)
    file = adapter_file_tensors(tmp_path / "adapter_model.safetensors")
    result = verify_parameters(model.named_parameters(), file)
    assert result["adapter_verification"]["ok"], result["adapter_verification"]
    assert set(result["foundation_parameter_dtypes"]) == {"bfloat16"}
    assert result["adapter_parameter_dtypes"] == result["adapter_verification"]["file_dtypes"]
