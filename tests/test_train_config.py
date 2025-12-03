"""Tests for char_lstm.train config and argument parsing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from char_lstm.train import (
    ParsedArgs,
    TrainConfig,
    _extract_args,
    build_run_paths,
    build_train_config,
    parse_args,
    print_config,
)


def test_build_train_config_cuda() -> None:
    """Test build_train_config with CUDA enabled."""
    args: ParsedArgs = {
        "lang": "tr",
        "from_checkpoint": None,
        "freeze_embed": False,
        "epochs": 5,
        "lr": 1e-3,
    }
    config = build_train_config(args, use_cuda=True)

    assert config["batch_size"] == 256
    assert config["num_workers"] == 4
    assert config["pin_memory"] is True
    assert config["num_epochs"] == 5
    assert config["lr"] == 1e-3


def test_build_train_config_cpu() -> None:
    """Test build_train_config with CPU."""
    args: ParsedArgs = {
        "lang": "tr",
        "from_checkpoint": None,
        "freeze_embed": False,
        "epochs": 3,
        "lr": 1e-4,
    }
    config = build_train_config(args, use_cuda=False)

    assert config["batch_size"] == 64
    assert config["num_workers"] == 0
    assert config["pin_memory"] is False
    assert config["num_epochs"] == 3


def test_extract_args_valid() -> None:
    """Test _extract_args with valid namespace."""
    ns = argparse.Namespace(
        lang="tr",
        from_checkpoint=None,
        freeze_embed=False,
        epochs=5,
        lr=1e-3,
    )
    result = _extract_args(ns)
    assert result["lang"] == "tr"
    assert result["from_checkpoint"] is None
    assert result["freeze_embed"] is False
    assert result["epochs"] == 5
    assert result["lr"] == 1e-3


def test_extract_args_with_checkpoint() -> None:
    """Test _extract_args with from_checkpoint set."""
    ns = argparse.Namespace(
        lang="az",
        from_checkpoint="checkpoints/tr_best.pt",
        freeze_embed=True,
        epochs=3,
        lr=5e-5,
    )
    result = _extract_args(ns)
    assert result["lang"] == "az"
    assert result["from_checkpoint"] == "checkpoints/tr_best.pt"
    assert result["freeze_embed"] is True


def test_extract_args_invalid_lang() -> None:
    """Test _extract_args with invalid lang type raises TypeError."""
    ns = argparse.Namespace(
        lang=123,  # Invalid: should be str
        from_checkpoint=None,
        freeze_embed=False,
        epochs=5,
        lr=1e-3,
    )
    with pytest.raises(TypeError, match="Expected str for lang"):
        _extract_args(ns)


def test_extract_args_invalid_from_checkpoint() -> None:
    """Test _extract_args with invalid from_checkpoint type raises TypeError."""
    ns = argparse.Namespace(
        lang="tr",
        from_checkpoint=123,  # Invalid: should be str or None
        freeze_embed=False,
        epochs=5,
        lr=1e-3,
    )
    with pytest.raises(TypeError, match="Expected str or None for from_checkpoint"):
        _extract_args(ns)


def test_extract_args_invalid_freeze_embed() -> None:
    """Test _extract_args with invalid freeze_embed type raises TypeError."""
    ns = argparse.Namespace(
        lang="tr",
        from_checkpoint=None,
        freeze_embed="yes",  # Invalid: should be bool
        epochs=5,
        lr=1e-3,
    )
    with pytest.raises(TypeError, match="Expected bool for freeze_embed"):
        _extract_args(ns)


def test_extract_args_invalid_epochs() -> None:
    """Test _extract_args with invalid epochs type raises TypeError."""
    ns = argparse.Namespace(
        lang="tr",
        from_checkpoint=None,
        freeze_embed=False,
        epochs="five",  # Invalid: should be int
        lr=1e-3,
    )
    with pytest.raises(TypeError, match="Expected int for epochs"):
        _extract_args(ns)


def test_extract_args_invalid_lr() -> None:
    """Test _extract_args with invalid lr type raises TypeError."""
    ns = argparse.Namespace(
        lang="tr",
        from_checkpoint=None,
        freeze_embed=False,
        epochs=5,
        lr="0.001",  # Invalid: should be float
    )
    with pytest.raises(TypeError, match="Expected float for lr"):
        _extract_args(ns)


def test_build_run_paths_new_training() -> None:
    """Test build_run_paths for new training (no checkpoint)."""
    paths = build_run_paths("tr", None)

    assert paths["run_name"] == "tr-train"
    assert paths["checkpoint_dir"] == Path("checkpoints")
    assert paths["vocab_json_path"] == Path("checkpoints/vocab.json")
    assert paths["checkpoint_best"] == Path("checkpoints/tr_best.pt")
    assert paths["source_checkpoint_path"] == Path("checkpoints/tr_best.pt")


def test_build_run_paths_finetune() -> None:
    """Test build_run_paths for fine-tuning."""
    paths = build_run_paths("az", "checkpoints/tr_best.pt")

    assert paths["run_name"] == "tr->az"
    assert paths["checkpoint_dir"] == Path("checkpoints")
    assert paths["vocab_json_path"] == Path("checkpoints/vocab.json")
    assert paths["checkpoint_best"] == Path("checkpoints/tr_to_az.pt")
    assert paths["source_checkpoint_path"] == Path("checkpoints/tr_best.pt")


def test_parse_args_basic() -> None:
    """Test parse_args parses command-line arguments."""
    test_args = ["train.py", "--lang", "tr"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()

    assert args.lang == "tr"
    assert args.from_checkpoint is None
    assert args.freeze_embed is False
    assert args.epochs == 3  # default
    assert args.lr == 1e-4  # default


def test_parse_args_all_options() -> None:
    """Test parse_args with all optional arguments."""
    test_args = [
        "train.py",
        "--lang",
        "az",
        "--from-checkpoint",
        "checkpoints/tr_best.pt",
        "--freeze-embed",
        "--epochs",
        "10",
        "--lr",
        "5e-5",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()

    assert args.lang == "az"
    assert args.from_checkpoint == "checkpoints/tr_best.pt"
    assert args.freeze_embed is True
    assert args.epochs == 10
    assert args.lr == 5e-5


def test_print_config_does_not_raise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test print_config outputs expected information."""
    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "num_workers": 0,
        "pin_memory": False,
    }

    print_config(
        lang_name="Turkish",
        lang_code="tr",
        is_finetune=False,
        checkpoint_path=tmp_path / "checkpoint.pt",
        device="cpu",
        config=config,
        freeze_embed=False,
        output_path=tmp_path / "output.pt",
    )

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    # Verify expected config values appear in structured output
    assert any("Turkish" in line and "Language" in line for line in output_lines)
    assert any("scratch" in line.lower() for line in output_lines)
    assert any("cpu" in line.lower() for line in output_lines)


def test_print_config_finetune_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test print_config shows fine-tune mode."""
    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "num_workers": 0,
        "pin_memory": False,
    }

    print_config(
        lang_name="Azerbaijani",
        lang_code="az",
        is_finetune=True,
        checkpoint_path=tmp_path / "tr_best.pt",
        device="cuda",
        config=config,
        freeze_embed=True,
        output_path=tmp_path / "tr_to_az.pt",
    )

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    # Verify fine-tune mode info appears in structured output
    assert any("Fine-tune" in line or "finetune" in line.lower() for line in output_lines)
    assert any("freeze" in line.lower() and "True" in line for line in output_lines)
