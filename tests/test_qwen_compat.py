"""Compatibility with the real Qwen/Qwen3.5-9B components, without downloading weights.

Skipped unless SENTINEL_QWEN_DIR points to a directory holding the pinned revision's small files
(config.json, model.safetensors.index.json, tokenizer.json, tokenizer_config.json,
chat_template.jinja, vocab.json, merges.txt; about 23 MB, no weights) and transformers is installed:

    rev=c202236235762e1c871ad0ccb60c8ee5ba337b9a
    for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \\
             chat_template.jinja vocab.json merges.txt; do
      curl -sLo "$DIR/$f" "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/$rev/$f"; done
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

QWEN_DIR = os.environ.get("SENTINEL_QWEN_DIR")
pytestmark = pytest.mark.skipif(not QWEN_DIR, reason="SENTINEL_QWEN_DIR not set")


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(QWEN_DIR)


def _examples():
    from sentinel.garden import render, supervision
    from tests.train_fixtures import _records

    records = _records("example/webapp", "GHSA-2222-3333-4444", "train")
    return [render.to_chat_example(r) for r in records] + supervision.patch_examples(records)


def test_training_prompt_equals_the_evaluation_prompt_and_only_the_reply_is_supervised(tokenizer) -> None:
    from sentinel.train.data import IGNORE_INDEX, encode_all

    examples = _examples()
    encoded, dropped = encode_all(tokenizer, examples, 4096, "<|im_end|>")
    assert not dropped and len(encoded) == len(examples) >= 4
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for example, item in zip(examples, encoded):
        *prompt, reply = example["messages"]
        # exactly what HFBackend.generate feeds the model at evaluation time
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        joint = tokenizer(prompt_text + reply["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        assert item.input_ids == joint  # no token merges across the prompt/reply boundary
        supervised = [t for t in item.labels if t != IGNORE_INDEX]
        assert supervised[-1] == eot
        assert tokenizer.decode(supervised) == reply["content"] + "<|im_end|>"
        full = tokenizer.apply_chat_template(example["messages"], tokenize=False, enable_thinking=False)
        assert full.startswith(prompt_text + reply["content"] + "<|im_end|>")


def test_checkpoint_weight_names_load_into_the_text_model() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.conversion_mapping import get_model_conversion_mapping

    config = transformers.AutoConfig.from_pretrained(QWEN_DIR)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    renames = [m for m in get_model_conversion_mapping(model) if type(m).__name__ == "PrefixChange"]
    assert any(p == r"^model\.language_model\.(.+)" for m in renames for p in m.source_patterns)
    stored = json.loads((Path(QWEN_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    mapped = {re.sub(r"^model\.language_model\.", "model.", k) for k in stored}
    expected = set(model.state_dict())
    assert expected <= mapped, sorted(expected - mapped)[:5]  # every text weight is in the checkpoint
    ignored = {k.split(".")[0] + "." + k.split(".")[1] for k in mapped - expected}
    assert ignored <= {"model.visual", "mtp.fc", "mtp.layers", "mtp.norm", "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"}
