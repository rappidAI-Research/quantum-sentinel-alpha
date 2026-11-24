"""End-of-turn handling of the HF backend, without torch: the turn ends at <|im_end|>."""

import pytest

from sentinel.eval.backends import END_OF_TURN, stop_token_ids, truncate_at_stop


class FakeTokenizer:
    unk_token_id = None

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token: str):
        return self.vocab.get(token)


def test_end_of_turn_comes_first_and_config_eos_is_kept() -> None:
    tokenizer = FakeTokenizer({END_OF_TURN: 7, "<|endoftext|>": 9})
    assert stop_token_ids(tokenizer, 9) == [7, 9]
    assert stop_token_ids(tokenizer, [9, 7, 9]) == [7, 9]
    assert stop_token_ids(tokenizer, None) == [7]


def test_missing_end_of_turn_token_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="im_end"):
        stop_token_ids(FakeTokenizer({}), 9)


def test_nothing_after_the_first_stop_token_is_kept() -> None:
    assert truncate_at_stop([1, 2, 7, 3, 9], [7, 9]) == [1, 2]
    assert truncate_at_stop([1, 2, 3], [7]) == [1, 2, 3]
    assert truncate_at_stop([9, 1], [7, 9]) == []
