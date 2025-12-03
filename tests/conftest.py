"""Pytest fixtures for char_lstm tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def device() -> torch.device:
    """Return available device (prefer CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
