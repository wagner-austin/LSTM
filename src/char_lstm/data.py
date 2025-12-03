"""Character-level dataset and vocabulary utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from char_lstm._types import UnknownJson

UNK = "<unk>"


class VocabJson(TypedDict):
    """Schema for vocabulary JSON files."""

    itos: list[str]
    unk: str


class VocabResult(TypedDict):
    """Result of building or loading vocabulary."""

    stoi: dict[str, int]
    itos: dict[int, str]
    vocab_size: int


class VocabLoadResult(TypedDict):
    """Result of loading vocabulary from JSON."""

    stoi: dict[str, int]
    itos: dict[int, str]
    vocab_size: int
    unk_token: str


def build_vocab_with_unk(
    text: str,
    unk_token: str = UNK,
) -> tuple[dict[str, int], dict[int, str], int]:
    """Build vocabulary from text with UNK token appended at the end.

    Args:
        text: Source text to extract characters from.
        unk_token: Token to use for unknown characters.

    Returns:
        Tuple of (stoi, itos, vocab_size) where:
        - stoi: Character to index mapping
        - itos: Index to character mapping
        - vocab_size: Total vocabulary size including UNK
    """
    chars = sorted(set(text))
    if unk_token in chars:
        chars.remove(unk_token)
    chars.append(unk_token)
    stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
    itos: dict[int, str] = dict(enumerate(chars))
    return stoi, itos, len(chars)


def save_vocab_json(
    itos: dict[int, str],
    path: str | Path,
    unk_token: str = UNK,
) -> None:
    """Save vocabulary to JSON file.

    Args:
        itos: Index to character mapping.
        path: Output file path.
        unk_token: UNK token used in vocabulary.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    itos_list: list[str] = [itos[i] for i in range(len(itos))]
    vocab_data: VocabJson = {"itos": itos_list, "unk": unk_token}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)


def _decode_vocab_json(raw: UnknownJson) -> VocabJson:
    """Decode raw JSON data as VocabJson.

    Internal function for JSON validation - uses UnknownJson type.

    Args:
        raw: Parsed JSON data to validate.

    Returns:
        Validated VocabJson data.

    Raises:
        TypeError: If data structure is incorrect.
        KeyError: If required keys are missing.
    """
    if not isinstance(raw, dict):
        msg = f"Expected dict, got {type(raw).__name__}"
        raise TypeError(msg)

    if "itos" not in raw:
        msg = "Missing required key 'itos'"
        raise KeyError(msg)

    itos_raw = raw["itos"]
    if not isinstance(itos_raw, list):
        msg = f"Expected list for 'itos', got {type(itos_raw).__name__}"
        raise TypeError(msg)

    itos_list: list[str] = []
    for i, item in enumerate(itos_raw):
        if not isinstance(item, str):
            msg = f"Expected str at itos[{i}], got {type(item).__name__}"
            raise TypeError(msg)
        itos_list.append(item)

    unk_value = raw.get("unk", UNK)
    if not isinstance(unk_value, str):
        msg = f"Expected str for 'unk', got {type(unk_value).__name__}"
        raise TypeError(msg)
    unk_token: str = unk_value

    result: VocabJson = {"itos": itos_list, "unk": unk_token}
    return result


def _load_vocab_json_data(path: Path) -> VocabJson:
    """Load and validate vocab JSON data from file.

    Internal _load* function - entry point for JSON parsing.

    Args:
        path: Path to vocab JSON file.

    Returns:
        Validated VocabJson data.

    Raises:
        KeyError: If required keys are missing.
        TypeError: If data types are incorrect.
    """
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    raw: UnknownJson = json.loads(content)
    return _decode_vocab_json(raw)


def load_vocab_json(
    path: str | Path,
) -> tuple[dict[str, int], dict[int, str], int, str]:
    """Load vocabulary from JSON file.

    Args:
        path: Path to vocab JSON file.

    Returns:
        Tuple of (stoi, itos, vocab_size, unk_token).
    """
    p = Path(path)
    vocab_data = _load_vocab_json_data(p)
    itos_list = vocab_data["itos"]
    unk_token = vocab_data["unk"]
    stoi: dict[str, int] = {ch: i for i, ch in enumerate(itos_list)}
    itos: dict[int, str] = dict(enumerate(itos_list))
    return stoi, itos, len(itos_list), unk_token


def encode(text: str, stoi: dict[str, int], unk_token: str = UNK) -> list[int]:
    """Encode text to list of token indices.

    Args:
        text: Text to encode.
        stoi: Character to index mapping.
        unk_token: Token for unknown characters.

    Returns:
        List of integer indices.
    """
    unk_idx = stoi[unk_token]
    return [stoi.get(c, unk_idx) for c in text]


class CharDataset(Dataset[tuple[Tensor, Tensor]]):
    """Character-level dataset for next-character prediction.

    Each sample is a (x, y) pair where y is x shifted by one position.
    """

    def __init__(
        self,
        text: str,
        stoi: dict[str, int],
        seq_len: int,
        unk_token: str = UNK,
    ) -> None:
        """Initialize character dataset.

        Args:
            text: Source text to create samples from.
            stoi: Character to index mapping.
            seq_len: Sequence length for each sample.
            unk_token: Token for unknown characters.
        """
        self.data: list[int] = encode(text, stoi, unk_token)
        self.seq_len: int = seq_len

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x, y) tensors where y is x shifted by 1.
        """
        x = torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y
