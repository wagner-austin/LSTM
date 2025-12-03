"""Tests for char_lstm.data module."""

from __future__ import annotations

from pathlib import Path

import pytest

from char_lstm.data import (
    CharDataset,
    _decode_vocab_json,
    build_vocab_with_unk,
    encode,
    load_vocab_json,
    save_vocab_json,
)


def test_build_vocab_with_unk_includes_all_chars() -> None:
    """Test vocabulary building includes all chars plus UNK."""
    text = "hello"
    stoi, _itos, vocab_size = build_vocab_with_unk(text)

    # Should have unique chars + <unk>
    unique_chars = set(text)
    assert vocab_size == len(unique_chars) + 1
    assert "<unk>" in stoi
    assert stoi["<unk>"] == vocab_size - 1  # UNK is last


def test_build_vocab_with_unk_sorted_order() -> None:
    """Test that vocab chars are sorted alphabetically."""
    text = "cba"
    stoi, itos, _vocab_size = build_vocab_with_unk(text)

    # Should be sorted: a, b, c, then <unk>
    assert stoi["a"] == 0
    assert stoi["b"] == 1
    assert stoi["c"] == 2
    assert itos[0] == "a"
    assert itos[1] == "b"
    assert itos[2] == "c"


def test_build_vocab_with_custom_unk_token() -> None:
    """Test vocabulary building with custom UNK token."""
    text = "abc"
    unk = "[UNK]"
    stoi, itos, vocab_size = build_vocab_with_unk(text, unk_token=unk)

    assert unk in stoi
    assert itos[vocab_size - 1] == unk


def test_build_vocab_when_text_contains_unk() -> None:
    """Test vocabulary building when text already contains UNK token."""
    # Include the UNK token in the text itself
    text = "abc<unk>def"
    stoi, itos, vocab_size = build_vocab_with_unk(text)

    # UNK should still be at the end, only appearing once
    assert stoi["<unk>"] == vocab_size - 1
    assert itos[vocab_size - 1] == "<unk>"
    # UNK should not be duplicated
    unk_count = sum(1 for ch in itos.values() if ch == "<unk>")
    assert unk_count == 1


def test_build_vocab_with_single_char_unk_in_text() -> None:
    """Test vocab building when single-char UNK token is in text.

    This tests the branch where UNK is removed from chars before being
    appended at the end (line 58 in data.py).
    """
    # Text contains "*" which we'll use as UNK token
    text = "abc*def"
    unk = "*"
    stoi, itos, vocab_size = build_vocab_with_unk(text, unk_token=unk)

    # UNK should be at the end, only appearing once
    assert stoi["*"] == vocab_size - 1
    assert itos[vocab_size - 1] == "*"
    # Vocab should have: a, b, c, d, e, f, * (7 chars)
    assert vocab_size == 7
    # UNK should not be duplicated (it was removed from middle and added at end)
    unk_count = sum(1 for ch in itos.values() if ch == "*")
    assert unk_count == 1


def test_encode_known_chars() -> None:
    """Test encoding with known characters."""
    text = "abc"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)

    encoded = encode("abc", stoi)
    assert len(encoded) == 3
    assert encoded == [stoi["a"], stoi["b"], stoi["c"]]


def test_encode_unknown_chars() -> None:
    """Test encoding maps unknown chars to UNK."""
    text = "abc"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    unk_idx = stoi["<unk>"]

    encoded = encode("xyz", stoi)
    assert encoded == [unk_idx, unk_idx, unk_idx]


def test_encode_mixed_known_unknown() -> None:
    """Test encoding with mix of known and unknown chars."""
    text = "abc"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    unk_idx = stoi["<unk>"]

    encoded = encode("axb", stoi)
    assert encoded == [stoi["a"], unk_idx, stoi["b"]]


def test_save_and_load_vocab(tmp_path: Path) -> None:
    """Test vocab round-trip through JSON."""
    text = "hello world"
    stoi, itos, vocab_size = build_vocab_with_unk(text)

    vocab_path = tmp_path / "vocab.json"
    save_vocab_json(itos, vocab_path)

    loaded_stoi, loaded_itos, loaded_size, unk = load_vocab_json(vocab_path)

    assert loaded_size == vocab_size
    assert loaded_stoi == stoi
    assert loaded_itos == itos
    assert unk == "<unk>"


def test_save_vocab_creates_parent_dirs(tmp_path: Path) -> None:
    """Test that save_vocab_json creates parent directories."""
    text = "abc"
    _stoi, itos, _vocab_size = build_vocab_with_unk(text)

    vocab_path = tmp_path / "nested" / "dir" / "vocab.json"
    save_vocab_json(itos, vocab_path)

    assert vocab_path.exists()


def test_char_dataset_length() -> None:
    """Test dataset length calculation."""
    text = "hello world"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    seq_len = 5

    dataset = CharDataset(text, stoi, seq_len)

    # Length should be len(text) - seq_len
    assert len(dataset) == len(text) - seq_len


def test_char_dataset_length_empty() -> None:
    """Test dataset length when text shorter than seq_len."""
    text = "hi"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    seq_len = 10

    dataset = CharDataset(text, stoi, seq_len)

    assert len(dataset) == 0


def test_char_dataset_item_shape() -> None:
    """Test dataset returns correct shapes."""
    text = "hello world"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    seq_len = 5

    dataset = CharDataset(text, stoi, seq_len)
    x, y = dataset[0]

    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)


def test_char_dataset_xy_offset() -> None:
    """Test that y is offset by 1 from x (next char prediction)."""
    text = "abcdefghij"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    seq_len = 5

    dataset = CharDataset(text, stoi, seq_len)
    x, y = dataset[0]

    # y should be x shifted by 1
    encoded = encode(text, stoi)
    assert x.tolist() == encoded[0:5]
    assert y.tolist() == encoded[1:6]


def test_char_dataset_different_indices() -> None:
    """Test that different indices return different sequences."""
    text = "abcdefghij"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text)
    seq_len = 3

    dataset = CharDataset(text, stoi, seq_len)
    x0, _y0 = dataset[0]
    x1, _y1 = dataset[1]

    # Different starting positions should give different sequences
    assert x0.tolist() != x1.tolist()


def test_char_dataset_with_custom_unk() -> None:
    """Test CharDataset with custom UNK token."""
    text = "abc"
    unk = "[UNK]"
    stoi, _itos, _vocab_size = build_vocab_with_unk(text, unk_token=unk)
    seq_len = 2

    dataset = CharDataset(text, stoi, seq_len, unk_token=unk)

    assert len(dataset) == len(text) - seq_len


def test_decode_vocab_json_valid() -> None:
    """Test _decode_vocab_json with valid data."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = {"itos": ["a", "b", "c"], "unk": "<unk>"}
    result = _decode_vocab_json(raw)
    assert result["itos"] == ["a", "b", "c"]
    assert result["unk"] == "<unk>"


def test_decode_vocab_json_not_dict() -> None:
    """Test _decode_vocab_json with non-dict raises TypeError."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = "not a dict"
    with pytest.raises(TypeError, match="Expected dict"):
        _decode_vocab_json(raw)


def test_decode_vocab_json_missing_itos() -> None:
    """Test _decode_vocab_json with missing itos raises KeyError."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = {"unk": "<unk>"}
    with pytest.raises(KeyError, match="Missing required key 'itos'"):
        _decode_vocab_json(raw)


def test_decode_vocab_json_itos_not_list() -> None:
    """Test _decode_vocab_json with non-list itos raises TypeError."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = {"itos": "not a list", "unk": "<unk>"}
    with pytest.raises(TypeError, match="Expected list for 'itos'"):
        _decode_vocab_json(raw)


def test_decode_vocab_json_itos_non_string_element() -> None:
    """Test _decode_vocab_json with non-string itos element raises TypeError."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = {"itos": ["a", 123, "c"], "unk": "<unk>"}
    with pytest.raises(TypeError, match="Expected str at itos"):
        _decode_vocab_json(raw)


def test_decode_vocab_json_unk_not_string() -> None:
    """Test _decode_vocab_json with non-string unk raises TypeError."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = {"itos": ["a", "b"], "unk": 123}
    with pytest.raises(TypeError, match="Expected str for 'unk'"):
        _decode_vocab_json(raw)


def test_decode_vocab_json_default_unk() -> None:
    """Test _decode_vocab_json uses default unk when not provided."""
    from char_lstm._types import UnknownJson

    raw: UnknownJson = {"itos": ["a", "b", "c"]}
    result = _decode_vocab_json(raw)
    assert result["unk"] == "<unk>"
