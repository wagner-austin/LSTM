"""Pytest fixtures for char_lstm tests."""

from __future__ import annotations

import gc
from collections.abc import Generator
from pathlib import Path
from typing import Protocol

import pytest
import torch


class _CudaModule(Protocol):
    """Protocol for torch.cuda module interface."""

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        ...

    def is_initialized(self) -> bool:
        """Check if a CUDA context has been created in this process."""
        ...

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        ...

    def empty_cache(self) -> None:
        """Empty CUDA memory cache."""
        ...


def _get_cuda_module() -> _CudaModule:
    """Get torch.cuda module with strict typing."""
    torch_mod = __import__("torch")
    cuda_mod: _CudaModule = torch_mod.cuda
    return cuda_mod


def _cleanup_cuda_memory() -> None:
    """Clean up CUDA memory after test execution.

    This function performs a thorough cleanup of GPU memory:
    1. Runs Python garbage collector to release tensor references
    2. Synchronizes CUDA to ensure all operations complete
    3. Clears the CUDA memory cache

    Only performs cleanup if this process actually created a CUDA context.
    ``is_initialized()`` (unlike ``is_available()``) never initializes CUDA
    itself, so CPU-only tests in parallel xdist workers skip GPU work
    entirely instead of opening 20+ simultaneous device contexts.
    """
    cuda = _get_cuda_module()
    if not cuda.is_initialized():
        return

    # Force garbage collection to release tensor references
    gc.collect()

    # Synchronize to ensure all CUDA operations complete
    cuda.synchronize()

    # Clear the CUDA memory cache
    cuda.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_cuda_after_test() -> Generator[None, None, None]:
    """Autouse fixture that cleans up CUDA memory after each test.

    This fixture ensures GPU memory is properly released after each test,
    preventing memory exhaustion when running tests in parallel with
    pytest-xdist on machines with limited GPU memory.

    Yields:
        None - fixture runs cleanup after test completes
    """
    # Run the test
    yield

    # Cleanup after test
    _cleanup_cuda_memory()


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
