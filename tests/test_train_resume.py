"""Resume-state round trip for interrupted training runs.

Saving weights is easy to get right and proves nothing. What decides whether a
preempted run continues *as the same run* is narrower:

  1. Adam's moment estimates survive -- otherwise the restored run gets a loss
     spike and follows a different trajectory than the one that was killed.
  2. The RNG streams continue rather than restart -- otherwise the run is not
     reproducible across the interruption.
  3. A configuration mismatch is refused rather than silently accepted.

These tests assert those three directly, because a test that only compared
model weights would pass against the previous weights-only implementation,
which could not resume at all.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from char_lstm._types import ResumePayload
from char_lstm.train import (
    TrainState,
    load_resume_state,
    resume_state_path,
    save_resume_state,
)


class _TinyModel(nn.Module):
    """Minimal module standing in for CharLSTM; only state_dict matters here."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer."""
        result: torch.Tensor = self.fc(x)
        return result


def _step_a_few_times(model: nn.Module, optimizer: torch.optim.Adam) -> None:
    """Take real optimisation steps so Adam accumulates non-zero moments."""
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(3, 8)).pow(2).mean()
        loss.backward()
        optimizer.step()


def _fresh_state() -> TrainState:
    """Build the state a run starts from.

    Returns:
        The state, typed as the contract rather than as a bare mapping --
        which is what five ``type: ignore[arg-type]`` comments were
        suppressing.
    """
    return TrainState(
        global_step=0,
        window_sum=0.0,
        window_n=0,
        best_val_loss=float("inf"),
        epochs_no_improve=0,
    )


def test_resume_state_path_is_a_sibling_of_the_checkpoint() -> None:
    """The resume file must not collide with the best-model checkpoint.

    ``scripts/zero_shot_eval.py`` loads the checkpoint as a bare state dict with
    ``strict=True``; writing the richer payload there would break it.
    """
    assert resume_state_path(Path("ckpt/tr_best.pt")).name == "tr_best.pt.resume"


def test_resume_restores_optimizer_moments_and_rng(tmp_path: Path) -> None:
    """A restored run continues the interrupted run, not merely its weights."""
    resume_path = resume_state_path(tmp_path / "best.pt")

    torch.manual_seed(1234)
    random.seed(1234)
    model_a = _TinyModel()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    _step_a_few_times(model_a, optimizer_a)

    moments_before = optimizer_a.state_dict()["state"][0]["exp_avg"].clone()
    if float(moments_before.abs().sum()) == 0.0:
        raise AssertionError("optimizer accumulated no moments; test is vacuous")

    state = TrainState(
        global_step=137,
        window_sum=1.5,
        window_n=3,
        best_val_loss=0.4242,
        epochs_no_improve=2,
    )
    save_resume_state(
        path=resume_path,
        model=model_a,
        optimizer=optimizer_a,
        epoch=7,
        state=state,
        vocab_size=99,
    )

    # The atomic write must leave no partial file behind.
    assert not resume_path.with_name(resume_path.name + ".tmp").exists()

    # What the interrupted process would have drawn next.
    expected_torch_next = torch.randn(4)
    expected_python_next = random.random()

    # A fresh process: different seed, nothing carried over in memory.
    torch.manual_seed(9999)
    random.seed(9999)
    model_b = _TinyModel()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    restored = _fresh_state()

    start_epoch = load_resume_state(
        path=resume_path,
        model=model_b,
        optimizer=optimizer_b,
        device=torch.device("cpu"),
        state=restored,
        vocab_size=99,
    )

    assert start_epoch == 8
    assert restored["global_step"] == 137
    assert restored["epochs_no_improve"] == 2
    assert abs(float(restored["best_val_loss"]) - 0.4242) < 1e-9
    assert torch.equal(model_a.fc.weight, model_b.fc.weight)
    assert torch.equal(optimizer_b.state_dict()["state"][0]["exp_avg"], moments_before)
    assert torch.equal(torch.randn(4), expected_torch_next)
    assert random.random() == expected_python_next


def test_resume_refuses_a_different_vocab_size(tmp_path: Path) -> None:
    """Resuming under a changed config would be neither the old run nor a new one."""
    resume_path = resume_state_path(tmp_path / "best.pt")
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_resume_state(
        path=resume_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        state=_fresh_state(),
        vocab_size=99,
    )

    with pytest.raises(ValueError, match="vocab_size"):
        load_resume_state(
            path=resume_path,
            model=_TinyModel(),
            optimizer=torch.optim.Adam(_TinyModel().parameters(), lr=1e-3),
            device=torch.device("cpu"),
            state=_fresh_state(),
            vocab_size=100,
        )


def test_missing_resume_file_is_a_fresh_run(tmp_path: Path) -> None:
    """Callers invoke this unconditionally, so absence must not be an error."""
    model = _TinyModel()
    start_epoch = load_resume_state(
        path=tmp_path / "absent.resume",
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        state=_fresh_state(),
        vocab_size=99,
    )
    assert start_epoch == 0


def test_a_state_saved_where_there_was_no_gpu_restores_where_there_is_one() -> None:
    """The fleet is mixed, so this crossing happens for real.

    Training runs on HPC3's ``free-gpu`` partition, which records one CUDA
    generator state per visible device; the workstation and the free CPU
    partition record none. A resume file written on either must load on the
    other, and with no CUDA states recorded there is nothing to restore, so
    the restore has to skip that step rather than fail on it.

    Written as a payload rather than by saving on a CPU-only machine because
    this one has a card: the branch is unreachable from ``save_resume_state``
    here, and stating the file a card-less run writes is how it gets covered.
    """
    with torch.no_grad():
        model = _TinyModel()
    payload: ResumePayload = {
        "model_state": model.state_dict(),
        "optimizer_state": torch.optim.Adam(model.parameters()).state_dict(),
        "epoch": 3,
        "global_step": 41,
        "best_val_loss": 0.5,
        "epochs_no_improve": 1,
        "vocab_size": 99,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": [],
        "rng_python_json": json.dumps(random.getstate()),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.pt"
        torch.save(payload, str(path))
        restored = _fresh_state()
        start_epoch = load_resume_state(
            path=path,
            model=_TinyModel(),
            optimizer=torch.optim.Adam(_TinyModel().parameters()),
            device=torch.device("cpu"),
            state=restored,
            vocab_size=99,
        )

    assert start_epoch == 4
    assert restored["global_step"] == 41


def test_a_resume_onto_the_device_it_trains_on_restores_the_rng(
    device: torch.device,
) -> None:
    """The failure that killed a real resume, and the one shape these tests
    could not previously reach.

    `bases-r1-ky` survived a preemption with 1189 seconds checkpointed, was
    resubmitted, and died 25 seconds in with

        TypeError: RNG state must be a torch.ByteTensor

    `load_resume_state` maps the payload onto the training device, which is
    right for weights and wrong for RNG states: `torch.set_rng_state` and
    `torch.cuda.set_rng_state_all` both document a `torch.ByteTensor`, meaning
    a CPU one, and map_location had moved them onto the card.

    Every other test in this file passes `torch.device("cpu")` explicitly, so
    map_location was always "cpu", the states never moved, and a suite at 100%
    coverage could not see it. This one resumes onto the device the fixture
    offers -- the card when there is one, which is where it reproduces.
    """
    with torch.no_grad():
        model = _TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    payload: ResumePayload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": 3,
        "global_step": 41,
        "best_val_loss": 0.5,
        "epochs_no_improve": 1,
        "vocab_size": 99,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": ([torch.cuda.get_rng_state()] if device.type == "cuda" else []),
        "rng_python_json": json.dumps(random.getstate()),
    }
    expected = payload["rng_torch"].clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.pt"
        torch.save(payload, str(path))
        restored = _fresh_state()
        start_epoch = load_resume_state(
            path=path,
            model=_TinyModel().to(device),
            optimizer=torch.optim.Adam(_TinyModel().to(device).parameters()),
            device=device,
            state=restored,
            vocab_size=99,
        )

    assert start_epoch == 4
    # The stream continues rather than restarts, which is the whole point of
    # restoring it -- and it is back on the CPU where torch keeps it.
    assert torch.equal(torch.get_rng_state(), expected)
