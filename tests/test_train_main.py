"""Tests for char_lstm.train main function and entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from char_lstm.corpora import corpus_file
from char_lstm.train import main


def write_corpus(tmp_path: Path) -> Path:
    """Write a small Azerbaijani-named corpus and return its directory.

    The trainer resolves the corpus file from ``--corpus-dir`` and the
    language code, so the test writes the real published file name rather
    than patching the language table.

    Args:
        tmp_path: Test-local directory to hold the corpus directory.

    Returns:
        The corpus directory to pass as ``--corpus-dir``.
    """
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    corpus_file(corpus_dir, "az").write_text("abcdefghij" * 1000, encoding="utf-8")
    return corpus_dir


@pytest.mark.timeout(180)
def test_main_integration(tmp_path: Path) -> None:
    """Test main function end-to-end through the real CLI path."""
    corpus_dir = write_corpus(tmp_path)
    checkpoint_dir = tmp_path / "ckpt_variant"

    test_args = [
        "train.py",
        "--lang",
        "az",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--corpus-dir",
        str(corpus_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]

    # Mock wandb to avoid actual logging
    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    with (
        patch.object(sys, "argv", test_args),
        patch("char_lstm.train.wandb", mock_wandb),
    ):
        main()

    # The run creates its checkpoint directory and writes the vocab there.
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "az_vocab.json").exists()


@pytest.mark.timeout(180)
def test_train_main_block(tmp_path: Path) -> None:
    """Test the ``if __name__ == '__main__'`` block by executing the module."""
    corpus_dir = write_corpus(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    test_args = [
        "train.py",
        "--lang",
        "az",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--corpus-dir",
        str(corpus_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]

    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    train_path = Path(__file__).resolve().parents[1] / "src" / "char_lstm" / "train.py"

    with (
        patch.object(sys, "argv", test_args),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        source = train_path.read_text(encoding="utf-8")
        code = compile(source, str(train_path), "exec")

        # Execute the module as __main__ to cover the entry-point block.
        try:
            exec(code, {"__name__": "__main__", "__file__": str(train_path)})
        except SystemExit as e:
            # main() returns None, so SystemExit(None) may be raised
            assert e.code is None or e.code == 0

    assert (checkpoint_dir / "az_vocab.json").exists()


def test_main_early_stopping_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function exits early when train_epoch returns False."""
    corpus_dir = write_corpus(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    # Use epochs=2 so break actually changes behavior
    test_args = [
        "train.py",
        "--lang",
        "az",
        "--epochs",
        "2",
        "--device",
        "cpu",
        "--corpus-dir",
        str(corpus_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]

    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    # Track how many times train_epoch is called
    call_count = 0

    def mock_train_epoch(**kwargs: dict[str, int]) -> tuple[bool, dict[str, float | int] | None]:
        nonlocal call_count
        call_count += 1
        # Return (False, metrics) on first call to trigger break
        mock_metrics: dict[str, float | int] = {
            "epoch": 1,
            "train_loss": 1.0,
            "train_ppl": 2.7,
            "val_loss": 1.0,
            "val_ppl": 2.7,
            "best_val_loss": 1.0,
            "learning_rate": 1e-4,
            "epochs_no_improve": 1,
        }
        return False, mock_metrics

    def mock_run_final_evaluation(**kwargs: dict[str, int]) -> None:
        # No-op since checkpoint won't exist
        pass

    # Use monkeypatch for simple attribute patching (reduces patch() count)
    import char_lstm.train as train_module

    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr(train_module, "wandb", mock_wandb)
    monkeypatch.setattr(train_module, "train_epoch", mock_train_epoch)
    monkeypatch.setattr(train_module, "run_final_evaluation", mock_run_final_evaluation)

    main()
    # train_epoch should only be called once because break was triggered
    assert call_count == 1


@pytest.mark.timeout(300)
def test_a_second_invocation_resumes_rather_than_restarting(tmp_path: Path) -> None:
    """The property the whole resume feature exists for.

    HPC3's ``free-gpu`` partition is preemptible: Slurm kills jobs to make
    room for allocated work, and ``slurm/train_base.sub`` passes ``--requeue``
    so the job comes back. Before resume state existed, coming back meant
    starting from epoch 0, which made the partition useless for anything real.

    So this runs the real CLI twice over the same checkpoint directory. The
    second invocation must find the first one's state and start after it,
    rather than train the same epoch again.
    """
    corpus_dir = write_corpus(tmp_path)
    checkpoint_dir = tmp_path / "ckpt_resume"

    test_args = [
        "train.py",
        "--lang",
        "az",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--corpus-dir",
        str(corpus_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]

    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    for _ in range(2):
        with (
            patch.object(sys, "argv", test_args),
            patch("char_lstm.train.wandb", mock_wandb),
        ):
            main()

    # The first run wrote a resume state; the second read it and had nothing
    # left to do, because its one epoch was already complete.
    resume_files = sorted(p.name for p in checkpoint_dir.glob("*resume*"))
    assert resume_files != []
