"""Internal type aliases and protocols for strict typing.

These types enable strict typing without Any, object, or cast.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

# Recursive type for JSON data - only for internal _load*/_decode* functions
UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


class _TorchLoad(Protocol):
    """Protocol for torch.load function with strict return type."""

    def __call__(
        self,
        f: str,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, Tensor]: ...


def _get_torch_load() -> _TorchLoad:
    """Get typed torch.load function via dynamic import."""
    torch_mod = __import__("torch")
    load_fn: _TorchLoad = torch_mod.load
    return load_fn


__all__ = ["UnknownJson", "_get_torch_load"]
