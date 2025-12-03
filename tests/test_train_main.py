"""Tests for char_lstm.train main function and entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from char_lstm.train import main


def test_main_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function end-to-end with mocked dependencies."""
    # Create a small test corpus
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefghij" * 1000)

    # Temporarily override LANGUAGES to use our test corpus
    test_languages = {
        "test": ("Test Language", str(corpus_path)),
    }

    # Prepare command-line args
    test_args = ["train.py", "--lang", "test", "--epochs", "1"]

    # Mock wandb to avoid actual logging
    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    # Change to tmp_path so checkpoints directory is created there
    original_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)

    try:
        with (
            patch.object(sys, "argv", test_args),
            patch("char_lstm.train.LANGUAGES", test_languages),
            patch("char_lstm.train.wandb", mock_wandb),
        ):
            main()

        # Verify checkpoint was created
        checkpoint_dir = tmp_path / "checkpoints"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "vocab.json").exists()

    finally:
        monkeypatch.chdir(original_cwd)


def test_train_main_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if __name__ == '__main__' block at train.py:890."""
    # Create a small test corpus
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefghij" * 1000)

    test_args = ["train.py", "--lang", "test", "--epochs", "1"]

    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    monkeypatch.chdir(tmp_path)

    train_path = Path(__file__).resolve().parents[1] / "src" / "char_lstm" / "train.py"

    with (
        patch.object(sys, "argv", test_args),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        # Read and compile the file
        source = train_path.read_text(encoding="utf-8")
        # Replace LANGUAGES definition with test version (escape backslashes for Windows)
        corpus_str = str(corpus_path).replace("\\", "\\\\")
        lang_val = f'{{"test": ("Test Language", "{corpus_str}")}}'
        source = source.replace(
            "LANGUAGES: dict[str, tuple[str, str]] = {",
            f"LANGUAGES: dict[str, tuple[str, str]] = {lang_val} or {{",
        )
        code = compile(source, str(train_path), "exec")

        # Execute the module as __main__ - this covers line 890
        # The exec catches SystemExit internally
        try:
            exec(code, {"__name__": "__main__", "__file__": str(train_path)})
        except SystemExit as e:
            # main() returns None, so SystemExit(None) may be raised
            assert e.code is None or e.code == 0


def test_main_early_stopping_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function exits early when train_epoch returns False.

    This covers the break statement at train.py:875.
    """
    # Create a small test corpus
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefghij" * 1000)

    test_languages = {
        "test": ("Test Language", str(corpus_path)),
    }

    # Use epochs=2 so break actually changes behavior
    test_args = ["train.py", "--lang", "test", "--epochs", "2"]

    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    original_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)

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
    monkeypatch.setattr(train_module, "LANGUAGES", test_languages)
    monkeypatch.setattr(train_module, "wandb", mock_wandb)
    monkeypatch.setattr(train_module, "train_epoch", mock_train_epoch)
    monkeypatch.setattr(train_module, "run_final_evaluation", mock_run_final_evaluation)

    try:
        main()
        # train_epoch should only be called once because break was triggered
        assert call_count == 1
    finally:
        monkeypatch.chdir(original_cwd)
