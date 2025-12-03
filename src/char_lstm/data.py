import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset

UNK = "<unk>"


def build_vocab_with_unk(text: str, unk_token: str = UNK) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Builds a vocab from text and appends an <unk> token."""
    chars = sorted(list(set(text)))
    if unk_token in chars:
        chars.remove(unk_token)
    chars.append(unk_token)
    stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
    itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def save_vocab_json(itos: Dict[int, str], path: Union[str, Path], unk_token: str = UNK) -> None:
    """Saves vocab as JSON with ordered itos list so indices are stable."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    itos_list: List[str] = [itos[i] for i in range(len(itos))]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"itos": itos_list, "unk": unk_token}, f, ensure_ascii=False, indent=2)


def load_vocab_json(path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str], int, str]:
    """Loads vocab JSON saved by save_vocab_json."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    itos_list: List[str] = obj["itos"]
    unk_token: str = obj.get("unk", UNK)
    stoi: Dict[str, int] = {ch: i for i, ch in enumerate(itos_list)}
    itos: Dict[int, str] = {i: ch for i, ch in enumerate(itos_list)}
    return stoi, itos, len(itos_list), unk_token


def encode(text: str, stoi: Dict[str, int], unk_token: str = UNK) -> List[int]:
    """Encodes text to indices; unknown chars map to <unk>."""
    unk_idx = stoi[unk_token]
    return [stoi.get(c, unk_idx) for c in text]


class CharDataset(Dataset):
    """Character-level dataset producing (x, y) pairs of length seq_len."""
    def __init__(self, text: str, stoi: Dict[str, int], seq_len: int, unk_token: str = UNK) -> None:
        self.data: List[int] = encode(text, stoi, unk_token)
        self.seq_len: int = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

