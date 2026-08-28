"""Internal type aliases and protocols for strict typing.

These types enable strict typing without Any, object, or cast.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

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


class ResumePayload(TypedDict):
    """Everything ``save_resume_state`` writes, named rather than inferred.

    A resume state is NOT a state dict. It carries weights, optimizer moments,
    three RNG streams, a vocabulary size and four scalars, so the file is
    heterogeneous -- and it was being loaded through :class:`_TorchLoad`,
    which promises ``dict[str, Tensor]``. Every field that is not a tensor
    then needed an ``object`` annotation and a ``type: ignore`` to get out
    again: six of them, plus two ``object``-in-annotation violations, all
    flagged by this repository's own guard and all committed anyway.

    Naming the shape removes the cause rather than the symptoms.

    Attributes:
        model_state: The model's weights.
        optimizer_state: Adam's moment estimates. Without these a resumed run
            restarts from zero momentum and diverges from the one it claims
            to continue.
        epoch: Index of the last COMPLETED epoch, 0-based.
        global_step: Optimiser steps taken so far.
        best_val_loss: Best validation loss seen.
        epochs_no_improve: Patience counter.
        vocab_size: Checked on restore; a mismatch means the corpus changed
            under the checkpoint.
        rng_torch: Torch's CPU generator state.
        rng_cuda: One generator state per visible CUDA device, empty on a
            CPU run.
        rng_python_json: ``random.getstate()`` as JSON, because the raw tuple
            nests ints and the payload must stay loadable under
            ``weights_only=True``.
    """

    model_state: dict[str, Tensor]
    optimizer_state: dict[str, Tensor]
    epoch: int
    global_step: int
    best_val_loss: float
    epochs_no_improve: int
    vocab_size: int
    rng_torch: Tensor
    rng_cuda: list[Tensor]
    rng_python_json: str


class _ResumeLoad(Protocol):
    """Protocol for torch.load reading a resume payload."""

    def __call__(
        self,
        f: str,
        *,
        map_location: str,
        weights_only: bool,
    ) -> ResumePayload: ...


def _get_resume_load() -> _ResumeLoad:
    """Get torch.load typed to return a resume payload.

    Returns:
        The load function, promising the shape this project actually writes.
    """
    torch_mod = __import__("torch")
    load_fn: _ResumeLoad = torch_mod.load
    return load_fn


__all__ = ["ResumePayload", "UnknownJson", "_get_resume_load", "_get_torch_load"]
