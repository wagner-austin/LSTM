"""Tests for char_lstm.train wandb logging functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from char_lstm.train import (
    EpochMetrics,
    WandbConfig,
    wb_config,
    wb_log,
    wb_log_epoch_table,
)


def test_wb_log_no_run() -> None:
    """Test wb_log does nothing when wandb.run is None."""
    # This should not raise - just a no-op
    wb_log({"loss": 0.5})


def test_wb_log_with_active_run() -> None:
    """Test wb_log calls wandb.log when wandb.run is active."""
    mock_run = MagicMock()
    with patch("char_lstm.train.wandb") as mock_wandb:
        mock_wandb.run = mock_run
        mock_wandb.log = MagicMock()

        wb_log({"loss": 0.5, "step": 100})

        mock_wandb.log.assert_called_once_with({"loss": 0.5, "step": 100})


def test_wb_config_no_run() -> None:
    """Test wb_config does nothing when wandb.run is None."""
    config: WandbConfig = {
        "vocab_size": 100,
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "patience": 1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "language": "Turkish",
        "language_code": "tr",
        "is_finetune": False,
        "source_checkpoint": "",
        "freeze_embed": False,
        "device": "cpu",
    }
    # Should not raise - just a no-op when wandb.run is None
    wb_config(config)


def test_wb_config_with_active_run() -> None:
    """Test wb_config sets attributes when wandb.run is active."""
    config: WandbConfig = {
        "vocab_size": 100,
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "patience": 1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "language": "Turkish",
        "language_code": "tr",
        "is_finetune": False,
        "source_checkpoint": "",
        "freeze_embed": False,
        "device": "cpu",
    }

    mock_run = MagicMock()
    # Track setattr calls on config
    set_attrs: dict[str, int | float | str | bool] = {}

    class MockConfig:
        def __setattr__(self, name: str, value: int | float | str | bool) -> None:
            set_attrs[name] = value

    mock_wandb_config = MockConfig()
    with patch("char_lstm.train.wandb") as mock_wandb:
        mock_wandb.run = mock_run
        mock_wandb.config = mock_wandb_config

        wb_config(config)

        # Should have set all config attributes
        assert len(set_attrs) == len(config)
        assert set_attrs["vocab_size"] == 100
        assert set_attrs["language"] == "Turkish"


def test_wb_log_epoch_table_no_run() -> None:
    """Test wb_log_epoch_table does nothing when wandb.run is None."""
    epoch_history: list[EpochMetrics] = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train_ppl": 2.7,
            "val_loss": 0.9,
            "val_ppl": 2.5,
            "best_val_loss": 0.9,
            "learning_rate": 1e-4,
            "epochs_no_improve": 0,
        }
    ]
    # Should not raise - just a no-op when wandb.run is None
    wb_log_epoch_table(epoch_history)


def test_wb_log_epoch_table_empty_history() -> None:
    """Test wb_log_epoch_table does nothing with empty history."""
    epoch_history: list[EpochMetrics] = []

    mock_run = MagicMock()
    with patch("char_lstm.train.wandb") as mock_wandb:
        mock_wandb.run = mock_run
        mock_wandb.Table = MagicMock()
        mock_wandb.log = MagicMock()

        wb_log_epoch_table(epoch_history)

        # Should not create table for empty history
        mock_wandb.Table.assert_not_called()


def test_wb_log_epoch_table_with_data() -> None:
    """Test wb_log_epoch_table creates and logs table."""
    epoch_history: list[EpochMetrics] = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train_ppl": 2.7,
            "val_loss": 0.9,
            "val_ppl": 2.5,
            "best_val_loss": 0.9,
            "learning_rate": 1e-4,
            "epochs_no_improve": 0,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "train_ppl": 2.2,
            "val_loss": 0.7,
            "val_ppl": 2.0,
            "best_val_loss": 0.7,
            "learning_rate": 1e-4,
            "epochs_no_improve": 0,
        },
    ]

    mock_run = MagicMock()
    mock_table = MagicMock()
    with patch("char_lstm.train.wandb") as mock_wandb:
        mock_wandb.run = mock_run
        mock_wandb.Table = MagicMock(return_value=mock_table)
        mock_wandb.log = MagicMock()

        wb_log_epoch_table(epoch_history)

        # Should create table with columns and data
        mock_wandb.Table.assert_called_once()
        call_kwargs = mock_wandb.Table.call_args
        assert "columns" in call_kwargs.kwargs
        assert "data" in call_kwargs.kwargs
        assert len(call_kwargs.kwargs["data"]) == 2

        # Should log the table
        mock_wandb.log.assert_called_once_with({"epoch_summary": mock_table})
