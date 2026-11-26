"""Model backends for QuickEval.

A backend only ever receives the rendered chat messages, never ground truth. Heavy ML
dependencies are imported lazily so validation and scoring stay lightweight.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from sentinel.eval.baseline import detect
from sentinel.hf_revision import RevisionUnverified, reset_cuda_peak_memory, verify_loaded_revision

QUANTIZATION_MODES = ("none", "bnb-8bit", "bnb-4bit")
DTYPES = ("bfloat16", "float16", "float32")
MIN_TRANSFORMERS = "5.12"  # keep in sync with the [model] extra in pyproject.toml
# The chat template ends an assistant turn with this token. The pinned foundations ship no
# generation_config.json, so the config-derived EOS is <|endoftext|>: without an explicit stop the
# model would continue into a new turn after its answer.
END_OF_TURN = "<|im_end|>"


def stop_token_ids(tokenizer: Any, configured: Any) -> list[int]:
    """End-of-turn first, then whatever EOS the model config defines (deduplicated, in order)."""
    eot = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if not isinstance(eot, int) or eot < 0 or eot == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"tokenizer has no single {END_OF_TURN} token; the chat template contract does not hold")
    extra = configured if isinstance(configured, list) else ([] if configured is None else [configured])
    ids: list[int] = []
    for token_id in [eot, *extra]:
        if isinstance(token_id, int) and token_id not in ids:
            ids.append(token_id)
    return ids


def truncate_at_stop(token_ids: list[int], stop_ids: list[int]) -> list[int]:
    """Generated tokens up to (not including) the first stop token: nothing after the turn is kept."""
    stops = set(stop_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in stops:
            return token_ids[:index]
    return token_ids


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
    # Training Optimization v1: added to the log-probability of the `finding` decision token before the
    # decision is taken (the rest of the reply is generated as usual). 0.0 = plain greedy decoding. A non-zero
    # bias is part of the evaluated model and must be chosen on the training validation split, never on the
    # development set it is selected on or on a final test (sentinel.eval.decision_scores calibrate).
    finding_logit_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not self.do_sample and any(v is not None for v in (self.temperature, self.top_p, self.top_k)):
            raise ValueError("temperature/top_p/top_k require do_sample=true")
        if not math.isfinite(self.finding_logit_bias) or abs(self.finding_logit_bias) > 20:
            raise ValueError("finding_logit_bias must be finite and within [-20, 20]")
        if self.finding_logit_bias != 0.0 and self.do_sample:
            raise ValueError("finding_logit_bias is defined for greedy decoding only")

    def as_dict(self) -> dict[str, Any]:
        settings = asdict(self)
        if settings["finding_logit_bias"] == 0.0:
            del settings["finding_logit_bias"]  # plain decoding keeps the generation record of every earlier run
        return settings


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


# PEFT names a loaded LoRA tensor `<module>.lora_A.<adapter name>.weight`; the adapter file stores it
# as `<module>.lora_A.weight`.
_ADAPTER_PARAMETER = re.compile(r"\.(lora_(?:A|B|embedding_A|embedding_B|magnitude_vector))\.")
_SAFETENSORS_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16"}


def adapter_file_tensors(weights: "Path") -> dict[str, dict[str, Any]]:
    """Tensor names, dtypes and shapes of a safetensors file, read from its header (no tensor data)."""
    import struct

    with weights.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return {
        name: {"dtype": _SAFETENSORS_DTYPES.get(meta["dtype"], meta["dtype"]), "shape": list(meta["shape"])}
        for name, meta in header.items()
        if name != "__metadata__"
    }


def verify_parameters(
    named_parameters: Any, adapter_tensors: dict[str, dict[str, Any]] | None, adapter_name: str = "default"
) -> dict[str, Any]:
    """Split the loaded parameters into foundation and adapter parameters by identity, not by count.

    Without an adapter every parameter is a foundation parameter. With one, a parameter is an adapter
    parameter only if its name is a LoRA tensor; each must be exactly one tensor of the adapter file
    (same name, same shape, the file's dtype, or float32 when PEFT upcast a half-precision adapter), and
    every tensor of the file must be loaded. Everything else is a foundation parameter.
    """
    foundation: dict[str, int] = {}
    adapter: dict[str, int] = {}
    loaded: dict[str, tuple[str, list[int], int]] = {}
    for name, parameter in named_parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        count = int(parameter.numel())
        if adapter_tensors is not None and _ADAPTER_PARAMETER.search(name):
            key = _ADAPTER_PARAMETER.sub(lambda m: f".{m.group(1)}.", name.replace(f".{adapter_name}.", ".", 1), count=1)
            loaded[key] = (dtype, list(parameter.shape), count)
            adapter[dtype] = adapter.get(dtype, 0) + count
        else:
            foundation[dtype] = foundation.get(dtype, 0) + count
    result: dict[str, Any] = {"foundation_parameter_dtypes": dict(sorted(foundation.items()))}
    if adapter_tensors is None:
        return result
    missing = sorted(set(adapter_tensors) - set(loaded))
    unexpected = sorted(set(loaded) - set(adapter_tensors))
    shape_mismatches, dtype_mismatches = [], []
    for name in sorted(set(loaded) & set(adapter_tensors)):
        dtype, shape, _ = loaded[name]
        expected = adapter_tensors[name]
        if shape != expected["shape"]:
            shape_mismatches.append(name)
        if dtype != expected["dtype"] and not (dtype == "float32" and expected["dtype"] in {"bfloat16", "float16"}):
            dtype_mismatches.append(name)
    file_dtypes: dict[str, int] = {}
    for meta in adapter_tensors.values():
        size = 1
        for dimension in meta["shape"]:
            size *= int(dimension)
        file_dtypes[meta["dtype"]] = file_dtypes.get(meta["dtype"], 0) + size
    expected_parameters = sum(file_dtypes.values())
    loaded_parameters = sum(count for _, _, count in loaded.values())
    result["adapter_parameter_dtypes"] = dict(sorted(adapter.items()))
    result["adapter_verification"] = {
        "ok": not (missing or unexpected or shape_mismatches or dtype_mismatches) and loaded_parameters == expected_parameters,
        "expected_tensors": len(adapter_tensors),
        "loaded_tensors": len(loaded),
        "expected_parameters": expected_parameters,
        "loaded_parameters": loaded_parameters,
        "file_dtypes": dict(sorted(file_dtypes.items())),
        "missing": missing[:10],
        "unexpected": unexpected[:10],
        "shape_mismatches": shape_mismatches[:10],
        "dtype_mismatches": dtype_mismatches[:10],
    }
    return result


def describe_adapter(path: "Path", model_id: str, revision: str | None) -> dict[str, Any]:
    """Identity of a LoRA adapter; refuses an adapter trained for another base model or revision."""
    import hashlib

    config_path = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise ValueError(f"{path} is not a PEFT adapter directory (adapter_config.json + adapter_model.safetensors)")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != model_id:
        raise ValueError(f"adapter was trained for {config.get('base_model_name_or_path')!r}, not {model_id!r}")
    if config.get("revision") != revision:
        raise ValueError(f"adapter was trained on revision {config.get('revision')!r}, not {revision!r}")
    # A training checkpoint (`checkpoints/step-NNNNNN/adapter`) carries checksums.json one level up: the
    # adapter files must still be exactly what the trainer wrote.
    checksums = path.parent / "checksums.json"
    if path.name == "adapter" and checksums.is_file():
        recorded = json.loads(checksums.read_text(encoding="utf-8"))
        for name in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json"):
            actual = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
            if recorded.get(name) != actual:
                raise ValueError(f"checkpoint file {name} does not match {checksums}")
    run_file = path / "training_run.json"
    return {
        "path": str(path),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_revision": config.get("revision"),
        "training_run": json.loads(run_file.read_text(encoding="utf-8")) if run_file.is_file() else None,
    }


class HFBackend:
    """Hugging Face transformers backend for the Qwen3.5-architecture foundation (Ornith-1.5-9B) and compatible causal LMs."""

    name = "hf"

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        runtime: RuntimeSettings,
        generation: GenerationSettings,
        adapter: "Path | None" = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.runtime = runtime
        self.generation = generation
        self.adapter = adapter
        # Set before loading so a mismatched adapter is refused without downloading the base model.
        self.adapter_info: dict[str, Any] | None = describe_adapter(adapter, model_id, revision) if adapter else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._generation_config: Any = None
        self._stop_ids: list[int] = []
        self._revision_verification: dict[str, Any] | None = None
        self._parameter_verification: dict[str, Any] = {}
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
        if self._cuda():
            reset_cuda_peak_memory(torch)  # initializes CUDA and each device's allocator before resetting

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model, info = AutoModelForCausalLM.from_pretrained(self.model_id, output_loading_info=True, **kwargs)
        # A silently re-initialized layer would make every score meaningless.
        missing = list(info.get("missing_keys") or [])
        if missing:
            raise RuntimeError(f"checkpoint is missing weights ({len(missing)}), e.g. {missing[:5]}")
        # Prove which revision was loaded from the files themselves (config._commit_hash is None for
        # the pinned Ornith checkpoint). Unverified stays None, which keeps the run non-reportable.
        try:
            self._revision_verification = verify_loaded_revision(self.model_id, self.revision) if self.revision else None
        except RevisionUnverified as exc:
            self._revision_verification = {"error": str(exc)}
        if self.adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter), is_trainable=False)
        model.eval()
        self._model = model
        # Record what was actually loaded; the runner blocks reporting when it differs from the request.
        self._loaded_dtype = str(model.dtype).removeprefix("torch.")
        for parameter in model.parameters():
            name = str(parameter.dtype).removeprefix("torch.")
            self._parameter_dtypes[name] = self._parameter_dtypes.get(name, 0) + parameter.numel()
        # Foundation and adapter parameters are verified separately, by parameter identity.
        adapter_tensors = adapter_file_tensors(self.adapter / "adapter_model.safetensors") if self.adapter is not None else None
        self._parameter_verification = verify_parameters(model.named_parameters(), adapter_tensors)

        config = copy.deepcopy(model.generation_config)
        config.max_new_tokens = self.generation.max_new_tokens
        config.do_sample = self.generation.do_sample
        config.temperature = self.generation.temperature
        config.top_p = self.generation.top_p
        config.top_k = self.generation.top_k
        self._stop_ids = stop_token_ids(self._tokenizer, config.eos_token_id)
        config.eos_token_id = list(self._stop_ids)
        if config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._stop_ids[0]
        self._generation_config = config

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        import torch

        template_kwargs = {"add_generation_prompt": True, "enable_thinking": self.generation.enable_thinking}
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        inputs = self._tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        forced: list[int] = []
        if self.generation.finding_logit_bias != 0.0:
            forced = self._biased_decision_prefix(inputs["input_ids"])
            if forced:
                prefix = torch.tensor([forced], device=self._model.device, dtype=inputs["input_ids"].dtype)
                inputs = {"input_ids": torch.cat([inputs["input_ids"], prefix], dim=1)}
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.inference_mode():
            output = self._model.generate(**inputs, generation_config=self._generation_config)
        new_tokens = output[0, prompt_tokens:]
        if self._cuda() and torch.cuda.is_available():
            torch.cuda.synchronize()
        # <think>/</think> are ordinary tokens and stay in the text for the parser to strip;
        # special tokens are dropped, and nothing after the end of the turn is decoded.
        answer = truncate_at_stop([int(t) for t in new_tokens.tolist()], self._stop_ids)
        text = self._tokenizer.decode(answer, skip_special_tokens=True)
        if rendered.rstrip().endswith("<think>"):
            # The template opened a reasoning block; restore the tag so the parser can strip it.
            text = "<think>" + text
        return GenerationResult(text=text, generated_tokens=int(new_tokens.shape[0]), prompt_tokens=prompt_tokens)

    def _biased_decision_prefix(self, prompt_ids: Any) -> list[int]:
        """`{"decision": "` plus the decision token chosen with the finding bias, when greedy decoding would
        start the reply with that prefix and a decision token; otherwise [] (the reply is generated unchanged)."""
        import torch

        from sentinel.eval.decision_scores import decision_token_ids
        from sentinel.train.data import DECISION_PREFIX

        prefix = list(self._tokenizer(DECISION_PREFIX, add_special_tokens=False)["input_ids"])
        ids = decision_token_ids(self._tokenizer)
        tail = torch.tensor([prefix], device=self._model.device, dtype=prompt_ids.dtype)
        with torch.inference_mode():
            logits = self._model(input_ids=torch.cat([prompt_ids, tail], dim=1)).logits[0].float()
        start = int(prompt_ids.shape[1]) - 1
        if logits[start : start + len(prefix)].argmax(dim=-1).tolist() != prefix or int(logits[-1].argmax()) not in ids.values():
            return []
        logprobs = torch.log_softmax(logits[-1], dim=-1)
        scores = {d: float(logprobs[i]) + (self.generation.finding_logit_bias if d == "finding" else 0.0) for d, i in ids.items()}
        return prefix + [ids[max(scores, key=lambda d: scores[d])]]

    def describe(self) -> dict[str, Any]:
        import torch
        import transformers

        info: dict[str, Any] = {
            "backend": self.name,
            "resolved_revision": self.verified_revision(),
            "revision_verification": self._revision_verification,
            "loaded_dtype": self._loaded_dtype,
            "parameter_dtypes": dict(sorted(self._parameter_dtypes.items())),
            **self._parameter_verification,
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
                for key in ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id")
            }
        if self.runtime.quantization != "none":
            import bitsandbytes

            info["bitsandbytes_version"] = bitsandbytes.__version__
        return info

    def verified_revision(self) -> str | None:
        """The revision proven by file digests; never the requested revision taken on trust."""
        verification = self._revision_verification or {}
        return None if "error" in verification else verification.get("revision")

    def peak_vram_gib(self) -> float | None:
        import torch

        if not (self._cuda() and torch.cuda.is_available()):
            return None
        # Summed over visible GPUs, in case device_map spreads the model.
        return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
