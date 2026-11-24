"""Compatibility with the real pinned foundation files (ornith-ai/Ornith-1.5-9B), without weights.

Skipped unless SENTINEL_FOUNDATION_DIR points to a directory holding the pinned revision's small
files (about 23 MB, no weights) and transformers is installed:

    rev=489cb97981b8654bcfcf30ce1f94ed1b62e07b53
    for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \\
             chat_template.jinja vocab.json merges.txt; do
      curl -sfLo "$DIR/$f" "https://huggingface.co/ornith-ai/Ornith-1.5-9B/resolve/$rev/$f"; done
    SENTINEL_FOUNDATION_DIR=$DIR pytest tests/test_foundation_compat.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
import yaml

FOUNDATION_DIR = os.environ.get("SENTINEL_FOUNDATION_DIR")
pytestmark = pytest.mark.skipif(not FOUNDATION_DIR, reason="SENTINEL_FOUNDATION_DIR not set")

NON_THINKING_PROMPT_END = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(FOUNDATION_DIR)


def _examples():
    from sentinel.garden import render, supervision
    from tests.train_fixtures import _records

    records = _records("example/webapp", "GHSA-2222-3333-4444", "train")
    return [render.to_chat_example(r) for r in records] + supervision.patch_examples(records)


def test_files_are_the_pinned_ones() -> None:
    from sentinel.foundation_model import FOUNDATION_CONFIG

    pinned = yaml.safe_load(FOUNDATION_CONFIG.read_text(encoding="utf-8"))["files_sha256"]
    for name, digest in pinned.items():
        assert hashlib.sha256((Path(FOUNDATION_DIR) / name).read_bytes()).hexdigest() == digest, name


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
        assert prompt_text.endswith(NON_THINKING_PROMPT_END)  # official non-thinking path: empty, closed reasoning block
        joint = tokenizer(prompt_text + reply["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        assert item.input_ids == joint  # no token merges across the prompt/reply boundary
        supervised = [t for t in item.labels if t != IGNORE_INDEX]
        assert supervised[-1] == eot
        decoded = tokenizer.decode(supervised)
        assert decoded == reply["content"] + "<|im_end|>"
        assert "<think>" not in decoded and "</think>" not in decoded  # no reasoning tokens are ever trained
        # The template renders a finished assistant turn the same way (empty reasoning block), so a
        # trained example is exactly a conversation the model can be served with.
        full = tokenizer.apply_chat_template(example["messages"], tokenize=False, enable_thinking=False)
        assert full.startswith(prompt_text + reply["content"] + "<|im_end|>")


def test_reasoning_tags_survive_decoding_and_the_turn_ends_with_im_end(tokenizer) -> None:
    from sentinel.eval.backends import stop_token_ids, truncate_at_stop
    from sentinel.eval.parse import parse_model_output

    # <think>/</think> are ordinary tokens: decoding keeps them, so the parser can strip reasoning.
    for tag in ("<think>", "</think>"):
        assert tokenizer.convert_tokens_to_ids(tag) not in tokenizer.all_special_ids
    stops = stop_token_ids(tokenizer, 248044)  # config-derived EOS of the pinned config is <|endoftext|>
    assert stops == [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    answer = '{"decision": "no_finding", "evidence": "The query is parameterized."}'
    generated = "<think>\nchecking {the} query\n</think>\n\n" + answer + "<|im_end|>\n<|im_start|>user\n{}"
    ids = tokenizer(generated, add_special_tokens=False)["input_ids"]
    text = tokenizer.decode(truncate_at_stop(ids, stops), skip_special_tokens=True)
    assert "user" not in text  # nothing after the end of the turn
    parsed = parse_model_output(text, {"case_id": "probe", "code": "query(sql, params)\n"})
    assert parsed.error is None and parsed.prediction["decision"] == "no_finding"
    assert "<think>" not in parsed.visible_output


def test_checkpoint_weight_names_load_into_the_text_model() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.conversion_mapping import get_model_conversion_mapping

    config = transformers.AutoConfig.from_pretrained(FOUNDATION_DIR)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    renames = [m for m in get_model_conversion_mapping(model) if type(m).__name__ == "PrefixChange"]
    assert any(p == r"^model\.language_model\.(.+)" for m in renames for p in m.source_patterns)
    stored = json.loads((Path(FOUNDATION_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    mapped = {re.sub(r"^model\.language_model\.", "model.", k) for k in stored}
    expected = set(model.state_dict())
    assert expected <= mapped, sorted(expected - mapped)[:5]  # every text weight is in the checkpoint
    ignored = {k.split(".")[0] + "." + k.split(".")[1] for k in mapped - expected}
    assert ignored <= {"model.visual", "mtp.fc", "mtp.layers", "mtp.norm", "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"}


def test_committed_module_list_matches_the_model_built_from_the_pinned_config() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.inspect_foundation import main as inspect

    assert inspect(["--files-dir", str(FOUNDATION_DIR), "--check"]) == 0
