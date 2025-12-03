"""Tests for char_lstm.train checkpoint save/load functionality."""

from __future__ import annotations

from pathlib import Path

import torch

from char_lstm.model import CharLSTM
from char_lstm.train import (
    _load_checkpoint_state_dict,
    next_checkpoint_path,
    save_checkpoint,
)


def test_next_checkpoint_path_new_file(tmp_path: Path) -> None:
    """Test next_checkpoint_path when file doesn't exist."""
    base = tmp_path / "model.pt"
    result = next_checkpoint_path(base)
    assert result == base


def test_next_checkpoint_path_existing_file(tmp_path: Path) -> None:
    """Test next_checkpoint_path increments when file exists."""
    base = tmp_path / "model.pt"
    base.touch()

    result = next_checkpoint_path(base)
    assert result == tmp_path / "model.ft1.pt"


def test_next_checkpoint_path_multiple_existing(tmp_path: Path) -> None:
    """Test next_checkpoint_path finds next available number."""
    base = tmp_path / "model.pt"
    base.touch()
    (tmp_path / "model.ft1.pt").touch()
    (tmp_path / "model.ft2.pt").touch()

    result = next_checkpoint_path(base)
    assert result == tmp_path / "model.ft3.pt"


def test_next_checkpoint_path_loop(tmp_path: Path) -> None:
    """Test next_checkpoint_path continues incrementing until free slot found."""
    base = tmp_path / "model.pt"
    base.touch()
    (tmp_path / "model.ft1.pt").touch()
    (tmp_path / "model.ft2.pt").touch()
    (tmp_path / "model.ft3.pt").touch()
    (tmp_path / "model.ft4.pt").touch()

    result = next_checkpoint_path(base)
    assert result == tmp_path / "model.ft5.pt"


def test_save_and_load_checkpoint(tmp_path: Path, device: torch.device) -> None:
    """Test save_checkpoint and _load_checkpoint_state_dict round-trip."""
    vocab_size = 10
    model = CharLSTM(vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0).to(device)

    _optim_module = __import__("torch.optim", fromlist=["Adam"])
    _adam_cls = _optim_module.Adam
    optimizer = _adam_cls(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "test_checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        val_loss=0.5,
        epoch=2,
        global_step=100,
        vocab_size=vocab_size,
    )

    assert checkpoint_path.exists()

    # Load and verify
    loaded_state_dict = _load_checkpoint_state_dict(checkpoint_path, device)
    assert "embedding.weight" in loaded_state_dict
    assert "lstm.weight_ih_l0" in loaded_state_dict
    assert "linear.weight" in loaded_state_dict


def test_load_checkpoint_raw_state_dict(tmp_path: Path, device: torch.device) -> None:
    """Test _load_checkpoint_state_dict with raw state dict."""
    vocab_size = 10
    model = CharLSTM(vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0).to(device)

    checkpoint_path = tmp_path / "raw_state.pt"

    # Save raw state dict directly
    torch.save(model.state_dict(), checkpoint_path)

    # Load should handle raw state dict
    loaded = _load_checkpoint_state_dict(checkpoint_path, device)
    assert "embedding.weight" in loaded
