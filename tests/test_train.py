"""Tests for char_lstm.train module utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from char_lstm.data import CharDataset, build_vocab_with_unk
from char_lstm.model import CharLSTM
from char_lstm.train import (
    ParsedArgs,
    TrainConfig,
    _create_optimizer,
    _extract_args,
    _load_checkpoint_state_dict,
    build_run_paths,
    build_train_config,
    evaluate,
    next_checkpoint_path,
    print_config,
    save_checkpoint,
    wb_log,
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


def test_wb_log_no_run() -> None:
    """Test wb_log does nothing when wandb.run is None."""
    # This should not raise - just a no-op
    wb_log({"loss": 0.5})


def test_wb_log_with_active_run() -> None:
    """Test wb_log calls wandb.log when wandb.run is active."""
    from unittest.mock import MagicMock, patch

    mock_run = MagicMock()
    with patch("char_lstm.train.wandb") as mock_wandb:
        mock_wandb.run = mock_run
        mock_wandb.log = MagicMock()

        wb_log({"loss": 0.5, "step": 100})

        mock_wandb.log.assert_called_once_with({"loss": 0.5, "step": 100})


def test_evaluate(device: torch.device) -> None:
    """Test evaluate function computes loss and perplexity."""
    import torch.nn as nn

    # Create test data using CharDataset for proper typing
    text = "abcdefghij" * 10  # Repeat to have enough data
    stoi, _itos, actual_vocab_size = build_vocab_with_unk(text)
    # Create model with actual vocab size, dropout=0.0 since num_layers=1
    model = CharLSTM(actual_vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0).to(
        device
    )

    dataset = CharDataset(text, stoi, seq_len=5)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=2)

    criterion = nn.CrossEntropyLoss()

    avg_loss, avg_ppl = evaluate(model, loader, criterion, device, actual_vocab_size)

    assert avg_loss > 0
    assert avg_ppl > 1.0  # perplexity is exp(loss), always > 1 for loss > 0


def test_print_config_does_not_raise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test print_config outputs expected information."""
    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
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
    assert "Turkish" in captured.out
    assert "Train from scratch" in captured.out
    assert "cpu" in captured.out


def test_print_config_finetune_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test print_config shows fine-tune mode."""
    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
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
    assert "Fine-tune from" in captured.out
    assert "True" in captured.out  # freeze_embed


def test_create_optimizer() -> None:
    """Test _create_optimizer creates valid optimizer."""
    model = CharLSTM(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    optimizer = _create_optimizer(model, lr=1e-3)

    # Verify it's a valid optimizer (implements OptimizerProtocol)
    assert hasattr(optimizer, "zero_grad")
    assert hasattr(optimizer, "step")
    assert hasattr(optimizer, "state_dict")
    assert hasattr(optimizer, "param_groups")


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


def test_load_and_split_corpus(tmp_path: Path) -> None:
    """Test load_and_split_corpus loads and splits text correctly."""
    from char_lstm.train import load_and_split_corpus

    # Create test corpus file
    corpus_text = "a" * 1000  # 1000 characters
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(corpus_text)

    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.20,
        "num_workers": 0,
        "pin_memory": False,
    }

    corpus = load_and_split_corpus(str(corpus_path), config)

    assert len(corpus["train_text"]) == 750  # 75% of 1000
    assert len(corpus["val_text"]) == 200  # 20% of 1000
    assert len(corpus["test_text"]) == 50  # remaining 5%


def test_setup_vocab_new_build(tmp_path: Path) -> None:
    """Test setup_vocab builds new vocab when not fine-tuning."""
    from char_lstm.train import CorpusSplit, setup_vocab

    corpus: CorpusSplit = {
        "train_text": "abcdef",
        "val_text": "ghij",
        "test_text": "klmn",
    }

    vocab_path = tmp_path / "vocab.json"
    vocab = setup_vocab(corpus, is_finetune=False, vocab_json_path=vocab_path)

    assert vocab["vocab_size"] > 0
    assert "a" in vocab["stoi"]
    assert vocab_path.exists()


def test_setup_vocab_load_existing(tmp_path: Path) -> None:
    """Test setup_vocab loads existing vocab when fine-tuning."""
    from char_lstm.data import save_vocab_json
    from char_lstm.train import CorpusSplit, setup_vocab

    # Create and save a vocab first
    itos = {0: "x", 1: "y", 2: "z", 3: "<unk>"}
    vocab_path = tmp_path / "vocab.json"
    save_vocab_json(itos, vocab_path)

    corpus: CorpusSplit = {
        "train_text": "abcdef",
        "val_text": "ghij",
        "test_text": "klmn",
    }

    vocab = setup_vocab(corpus, is_finetune=True, vocab_json_path=vocab_path)

    # Should load existing vocab, not build new
    assert vocab["vocab_size"] == 4
    assert vocab["stoi"]["x"] == 0


def test_create_dataloaders() -> None:
    """Test create_dataloaders creates all three loaders."""
    from char_lstm.train import CorpusSplit, create_dataloaders

    # Need enough text for the sequence length
    corpus: CorpusSplit = {
        "train_text": "a" * 500,
        "val_text": "b" * 200,
        "test_text": "c" * 100,
    }

    stoi = {"a": 0, "b": 1, "c": 2, "<unk>": 3}

    config: TrainConfig = {
        "seq_len": 10,
        "batch_size": 32,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    loaders = create_dataloaders(corpus, stoi, config)

    assert "train_loader" in loaders
    assert "val_loader" in loaders
    assert "test_loader" in loaders
    assert len(loaders["train_loader"]) > 0


def test_setup_model_and_optimizer_new_training(tmp_path: Path) -> None:
    """Test setup_model_and_optimizer for new training."""
    from char_lstm.train import setup_model_and_optimizer

    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    checkpoint_best = tmp_path / "model.pt"
    source_checkpoint = tmp_path / "source.pt"

    setup = setup_model_and_optimizer(
        vocab_size=50,
        config=config,
        is_finetune=False,
        checkpoint_best=checkpoint_best,
        source_checkpoint_path=source_checkpoint,
        freeze_embed=False,
    )

    assert setup["model"] is not None
    assert setup["optimizer"] is not None
    assert setup["criterion"] is not None
    assert setup["device"] is not None
    assert setup["checkpoint_save"] == checkpoint_best


def test_setup_model_and_optimizer_finetune_not_found(tmp_path: Path) -> None:
    """Test setup_model_and_optimizer raises when checkpoint not found."""
    from char_lstm.train import setup_model_and_optimizer

    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    checkpoint_best = tmp_path / "model.pt"
    source_checkpoint = tmp_path / "nonexistent.pt"  # Doesn't exist

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        setup_model_and_optimizer(
            vocab_size=50,
            config=config,
            is_finetune=True,
            checkpoint_best=checkpoint_best,
            source_checkpoint_path=source_checkpoint,
            freeze_embed=False,
        )


def test_setup_model_and_optimizer_finetune_with_checkpoint(
    tmp_path: Path, device: torch.device
) -> None:
    """Test setup_model_and_optimizer loads checkpoint when fine-tuning."""
    from char_lstm.train import setup_model_and_optimizer

    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    # Create a source checkpoint
    vocab_size = 50
    source_model = CharLSTM(vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
    source_checkpoint = tmp_path / "source.pt"
    torch.save(source_model.state_dict(), source_checkpoint)

    checkpoint_best = tmp_path / "finetune.pt"

    setup = setup_model_and_optimizer(
        vocab_size=vocab_size,
        config=config,
        is_finetune=True,
        checkpoint_best=checkpoint_best,
        source_checkpoint_path=source_checkpoint,
        freeze_embed=False,
    )

    assert setup["model"] is not None
    # checkpoint_save should be versioned when fine-tuning
    assert "ft" in str(setup["checkpoint_save"]) or setup["checkpoint_save"] == checkpoint_best


def test_setup_model_and_optimizer_freeze_embed(tmp_path: Path, device: torch.device) -> None:
    """Test setup_model_and_optimizer freezes embeddings when requested."""
    from char_lstm.train import setup_model_and_optimizer

    config: TrainConfig = {
        "seq_len": 100,
        "batch_size": 64,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 1,
        "lr": 1e-4,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    # Create a source checkpoint
    vocab_size = 50
    source_model = CharLSTM(vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
    source_checkpoint = tmp_path / "source.pt"
    torch.save(source_model.state_dict(), source_checkpoint)

    checkpoint_best = tmp_path / "finetune.pt"

    setup = setup_model_and_optimizer(
        vocab_size=vocab_size,
        config=config,
        is_finetune=True,
        checkpoint_best=checkpoint_best,
        source_checkpoint_path=source_checkpoint,
        freeze_embed=True,
    )

    # Check that embedding parameters have requires_grad=False
    for p in setup["model"].embedding.parameters():
        assert not p.requires_grad


def test_train_epoch(tmp_path: Path, device: torch.device) -> None:
    """Test train_epoch runs one epoch of training."""
    import torch.nn as nn

    from char_lstm.train import TrainState, train_epoch

    # Create a simple dataset and model
    text = "abcdefghij" * 100
    stoi, _itos, vocab_size = build_vocab_with_unk(text)

    dataset = CharDataset(text, stoi, seq_len=5)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)

    # dropout=0.0 since num_layers=1 (dropout only applies between layers)
    model = CharLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0).to(device)
    optimizer = _create_optimizer(model, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    config: TrainConfig = {
        "seq_len": 5,
        "batch_size": 4,
        "num_epochs": 1,
        "log_every": 10,
        "patience": 5,
        "lr": 1e-3,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    state: TrainState = {
        "global_step": 0,
        "window_sum": 0.0,
        "window_n": 0,
        "best_val_loss": float("inf"),
        "epochs_no_improve": 0,
    }

    checkpoint_path = tmp_path / "checkpoint.pt"

    # Run one epoch
    should_continue = train_epoch(
        epoch=0,
        num_epochs=1,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        vocab_size=vocab_size,
        config=config,
        state=state,
        checkpoint_save=checkpoint_path,
    )

    assert should_continue is True
    assert state["global_step"] > 0
    assert state["best_val_loss"] < float("inf")


def test_train_epoch_early_stopping(tmp_path: Path, device: torch.device) -> None:
    """Test train_epoch triggers early stopping when no improvement."""
    import torch.nn as nn

    from char_lstm.train import TrainState, train_epoch

    # Create a simple dataset and model
    text = "abcdefghij" * 100
    stoi, _itos, vocab_size = build_vocab_with_unk(text)

    dataset = CharDataset(text, stoi, seq_len=5)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)

    # dropout=0.0 since num_layers=1 (dropout only applies between layers)
    model = CharLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0).to(device)
    optimizer = _create_optimizer(model, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    config: TrainConfig = {
        "seq_len": 5,
        "batch_size": 4,
        "num_epochs": 1,
        "log_every": 100,
        "patience": 1,  # Low patience for early stopping
        "lr": 1e-3,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    # Start with a very low best_val_loss to trigger early stopping
    state: TrainState = {
        "global_step": 0,
        "window_sum": 0.0,
        "window_n": 0,
        "best_val_loss": 0.001,  # Very low, unlikely to improve
        "epochs_no_improve": 0,
    }

    checkpoint_path = tmp_path / "checkpoint.pt"

    # Run one epoch - should trigger early stopping
    should_continue = train_epoch(
        epoch=0,
        num_epochs=1,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        vocab_size=vocab_size,
        config=config,
        state=state,
        checkpoint_save=checkpoint_path,
    )

    assert should_continue is False  # Early stopping triggered
    assert state["epochs_no_improve"] >= config["patience"]


def test_train_epoch_no_improvement_but_continue(tmp_path: Path, device: torch.device) -> None:
    """Test train_epoch continues when no improvement but patience not exhausted.

    This tests the branch at train.py:850->854 where epochs_no_improve is
    incremented but is still less than patience, so training continues.
    """
    import torch.nn as nn

    from char_lstm.train import TrainState, train_epoch

    # Create a simple dataset and model
    text = "abcdefghij" * 100
    stoi, _itos, vocab_size = build_vocab_with_unk(text)

    dataset = CharDataset(text, stoi, seq_len=5)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)

    # dropout=0.0 since num_layers=1 (dropout only applies between layers)
    model = CharLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0).to(device)
    optimizer = _create_optimizer(model, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    config: TrainConfig = {
        "seq_len": 5,
        "batch_size": 4,
        "num_epochs": 1,
        "log_every": 100,
        "patience": 5,  # High patience - should continue
        "lr": 1e-3,
        "train_ratio": 0.75,
        "val_ratio": 0.25,
        "num_workers": 0,
        "pin_memory": False,
    }

    # Start with a very low best_val_loss but high patience
    state: TrainState = {
        "global_step": 0,
        "window_sum": 0.0,
        "window_n": 0,
        "best_val_loss": 0.001,  # Very low, won't improve
        "epochs_no_improve": 0,
    }

    checkpoint_path = tmp_path / "checkpoint.pt"

    # Run one epoch - should NOT trigger early stopping due to high patience
    should_continue = train_epoch(
        epoch=0,
        num_epochs=1,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        vocab_size=vocab_size,
        config=config,
        state=state,
        checkpoint_save=checkpoint_path,
    )

    # Should continue because epochs_no_improve (1) < patience (5)
    assert should_continue is True
    assert state["epochs_no_improve"] == 1
    assert state["epochs_no_improve"] < config["patience"]


def test_run_final_evaluation(tmp_path: Path, device: torch.device) -> None:
    """Test run_final_evaluation loads checkpoint and evaluates."""
    import torch.nn as nn

    from char_lstm.train import run_final_evaluation

    # Create model and test data
    text = "abcdefghij" * 100
    stoi, _itos, vocab_size = build_vocab_with_unk(text)
    # dropout=0.0 since num_layers=1 (dropout only applies between layers)
    model = CharLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0).to(device)

    checkpoint_path = tmp_path / "best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    dataset = CharDataset(text, stoi, seq_len=5)
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)

    criterion = nn.CrossEntropyLoss()

    # This should run without error
    run_final_evaluation(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        vocab_size=vocab_size,
        checkpoint_save=checkpoint_path,
        checkpoint_best=checkpoint_path,
    )


def test_run_final_evaluation_fallback_checkpoint(tmp_path: Path, device: torch.device) -> None:
    """Test run_final_evaluation uses checkpoint_best when checkpoint_save doesn't exist."""
    import torch.nn as nn

    from char_lstm.train import run_final_evaluation

    # Create model and test data
    text = "abcdefghij" * 100
    stoi, _itos, vocab_size = build_vocab_with_unk(text)
    # dropout=0.0 since num_layers=1 (dropout only applies between layers)
    model = CharLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0).to(device)

    # Only checkpoint_best exists
    checkpoint_best = tmp_path / "best.pt"
    torch.save(model.state_dict(), checkpoint_best)
    checkpoint_save = tmp_path / "nonexistent.pt"  # Doesn't exist

    dataset = CharDataset(text, stoi, seq_len=5)
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=4)

    criterion = nn.CrossEntropyLoss()

    # Should fall back to checkpoint_best
    run_final_evaluation(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        vocab_size=vocab_size,
        checkpoint_save=checkpoint_save,
        checkpoint_best=checkpoint_best,
    )


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


def test_parse_args_basic() -> None:
    """Test parse_args parses command-line arguments."""
    import sys
    from unittest.mock import patch

    from char_lstm.train import parse_args

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
    import sys
    from unittest.mock import patch

    from char_lstm.train import parse_args

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


def test_main_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function end-to-end with mocked dependencies."""
    import sys
    from unittest.mock import MagicMock, patch

    from char_lstm.train import main

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
    import sys
    from unittest.mock import MagicMock, patch

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
    import sys
    from unittest.mock import MagicMock, patch

    from char_lstm.train import main

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

    def mock_train_epoch(**kwargs: dict[str, int]) -> bool:
        nonlocal call_count
        call_count += 1
        # Return False on first call to trigger break
        return False

    def mock_run_final_evaluation(**kwargs: dict[str, int]) -> None:
        # No-op since checkpoint won't exist
        pass

    try:
        with (
            patch.object(sys, "argv", test_args),
            patch("char_lstm.train.LANGUAGES", test_languages),
            patch("char_lstm.train.wandb", mock_wandb),
            patch("char_lstm.train.train_epoch", mock_train_epoch),
            patch("char_lstm.train.run_final_evaluation", mock_run_final_evaluation),
        ):
            main()

        # train_epoch should only be called once because break was triggered
        assert call_count == 1

    finally:
        monkeypatch.chdir(original_cwd)
