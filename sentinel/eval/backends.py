"""Model backends for QuickEval.

A backend only ever receives the rendered chat messages, never ground truth. Heavy ML
dependencies are imported lazily so validation and scoring stay lightweight.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sentinel.eval.baseline import detect

QUANTIZATION_MODES = ("none", "bnb-8bit", "bnb-4bit")
DTYPES = ("bfloat16", "float16", "float32")
MIN_TRANSFORMERS = "5.12"  # keep in sync with the [model] extra in pyproject.toml


@dataclass(frozen=True)
class GenerationSettings:
    """Decoding settings. Must be identical across bake-off candidates."""

    max_new_tokens: int = 384
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    enable_thinking: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not self.do_sample and any(v is not None for v in (self.temperature, self.top_p, self.top_k)):
            raise ValueError("temperature/top_p/top_k require do_sample=true")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSettings:
    """Hardware-facing settings. May differ between candidates only with a recorded reason."""

    device: str = "cuda"
    dtype: str = "bfloat16"
    quantization: str = "none"

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if self.quantization not in QUANTIZATION_MODES:
            raise ValueError(f"quantization must be one of {QUANTIZATION_MODES}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int | None = None
    prompt_tokens: int | None = None


class ModelBackend(Protocol):
    name: str

    def load(self) -> None: ...

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult: ...

    def describe(self) -> dict[str, Any]:
        """Runtime facts for the run artifact (library versions, GPU, resolved revision)."""
        ...

    def peak_vram_gib(self) -> float | None: ...


class BaselineBackend:
    """Label-blind regex baseline speaking the model output contract.

    Exercises prompt -> output -> parser -> scorer without a GPU. Not a security model.
    """

    name = "baseline"
    _NUMBERED = re.compile(r"^\s*\d+ \| ?(.*)$")

    def load(self) -> None:
        return None

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        parts = messages[-1]["content"].split("\nCode:\n", 1)
        if len(parts) < 2:  # not a QuickEval review prompt (e.g. coding regression): no answer
            return GenerationResult(text="")
        code_lines = [m.group(1) for line in parts[1].splitlines() if (m := self._NUMBERED.match(line))]
        hit = detect("\n".join(code_lines))
        if hit is None:
            payload: dict[str, Any] = {"decision": "no_finding", "evidence": "No baseline rule matched."}
        else:
            cwe, line = hit
            payload = {
                "decision": "finding",
                "cwe": cwe,
                "start_line": line,
                "end_line": line,
                "confidence": 0.55,
                "evidence": f"Baseline rule for {cwe} matched line {line}.",
            }
        return GenerationResult(text=json.dumps(payload))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}

    def peak_vram_gib(self) -> float | None:
        return None


class HFBackend:
    """Hugging Face transformers backend for Qwen3.5 text generation (and compatible causal LMs)."""

    name = "hf"

    def __init__(self, model_id: str, revision: str | None, runtime: RuntimeSettings, generation: GenerationSettings):
        self.model_id = model_id
        self.revision = revision
        self.runtime = runtime
        self.generation = generation
        self._model: Any = None
        self._tokenizer: Any = None
        self._generation_config: Any = None
        self._loaded_dtype: str | None = None
        self._parameter_dtypes: dict[str, int] = {}

    def _cuda(self) -> bool:
        return self.runtime.device.startswith("cuda") or self.runtime.device == "auto"

    def load(self) -> None:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if Version(transformers.__version__) < Version(MIN_TRANSFORMERS):
            raise RuntimeError(f"transformers>={MIN_TRANSFORMERS} is required, found {transformers.__version__}")
        torch.manual_seed(self.generation.seed)
        dtype = getattr(torch, self.runtime.dtype)
        kwargs: dict[str, Any] = {"revision": self.revision, "dtype": dtype, "device_map": self.runtime.device}
        if self.runtime.quantization != "none":
            from transformers import BitsAndBytesConfig

            if self.runtime.quantization == "bnb-4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if self._cuda() and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model, info = AutoModelForCausalLM.from_pretrained(self.model_id, output_loading_info=True, **kwargs)
        # A silently re-initialized layer would make every score meaningless.
        missing = list(info.get("missing_keys") or [])
        if missing:
            raise RuntimeError(f"checkpoint is missing weights ({len(missing)}), e.g. {missing[:5]}")
        model.eval()
        self._model = model
        # Record what was actually loaded; the runner blocks reporting when it differs from the request.
        self._loaded_dtype = str(model.dtype).removeprefix("torch.")
        for parameter in model.parameters():
            name = str(parameter.dtype).removeprefix("torch.")
            self._parameter_dtypes[name] = self._parameter_dtypes.get(name, 0) + parameter.numel()

        config = copy.deepcopy(model.generation_config)
        config.max_new_tokens = self.generation.max_new_tokens
        config.do_sample = self.generation.do_sample
        config.temperature = self.generation.temperature
        config.top_p = self.generation.top_p
        config.top_k = self.generation.top_k
        if config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        self._generation_config = config

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        import torch

        template_kwargs = {"add_generation_prompt": True, "enable_thinking": self.generation.enable_thinking}
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        inputs = self._tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output = self._model.generate(**inputs, generation_config=self._generation_config)
        new_tokens = output[0, prompt_tokens:]
        if self._cuda() and torch.cuda.is_available():
            torch.cuda.synchronize()
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        if rendered.rstrip().endswith("<think>"):
            # The template opened a reasoning block; restore the tag so the parser can strip it.
            text = "<think>" + text
        return GenerationResult(text=text, generated_tokens=int(new_tokens.shape[0]), prompt_tokens=prompt_tokens)

    def describe(self) -> dict[str, Any]:
        import torch
        import transformers

        info: dict[str, Any] = {
            "backend": self.name,
            "resolved_revision": getattr(self._model.config, "_commit_hash", None) if self._model else None,
            "loaded_dtype": self._loaded_dtype,
            "parameter_dtypes": dict(sorted(self._parameter_dtypes.items())),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            # Optional Gated-DeltaNet kernels change throughput, so both candidates must match.
            "fla_kernels": importlib.util.find_spec("fla") is not None,
            "causal_conv1d_kernels": importlib.util.find_spec("causal_conv1d") is not None,
        }
        if self._generation_config is not None:
            info["effective_generation_config"] = {
                key: getattr(self._generation_config, key, None)
                for key in ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty")
            }
        if self.runtime.quantization != "none":
            import bitsandbytes

            info["bitsandbytes_version"] = bitsandbytes.__version__
        return info

    def peak_vram_gib(self) -> float | None:
        import torch

        if not (self._cuda() and torch.cuda.is_available()):
            return None
        # Summed over visible GPUs, in case device_map spreads the model.
        return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
