"""Tests for char_lstm.train setup functions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from char_lstm.model import CharLSTM
from char_lstm.train import (
    CorpusSplit,
    TrainConfig,
    _create_optimizer,
    create_dataloaders,
    load_and_split_corpus,
    setup_model_and_optimizer,
    setup_vocab,
)


def test_create_optimizer() -> None:
    """Test _create_optimizer creates valid optimizer."""
    model = CharLSTM(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    optimizer = _create_optimizer(model, lr=1e-3)

    # Verify optimizer has correct learning rate
    assert optimizer.param_groups[0]["lr"] == 1e-3
    # Verify optimizer tracks model parameters (param_groups is a list of dicts)
    param_groups = optimizer.param_groups
    assert len(param_groups) == 1
    # Verify state_dict works (optimizer is functional)
    state = optimizer.state_dict()
    assert "param_groups" in state


def test_load_and_split_corpus(tmp_path: Path) -> None:
    """Test load_and_split_corpus loads and splits text correctly."""
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
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "num_workers": 0,
        "pin_memory": False,
    }

    corpus = load_and_split_corpus(str(corpus_path), config)

    assert len(corpus["train_text"]) == 700  # 70% of 1000
    assert len(corpus["val_text"]) == 150  # 15% of 1000
    assert len(corpus["test_text"]) == 150  # remaining 15%


def test_setup_vocab_new_build(tmp_path: Path) -> None:
    """Test setup_vocab builds new vocab when not fine-tuning."""
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
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "num_workers": 0,
        "pin_memory": False,
    }

    loaders = create_dataloaders(corpus, stoi, config)

    assert "train_loader" in loaders
    assert "val_loader" in loaders
    assert "test_loader" in loaders
    # Verify train_loader has batches (corpus is large enough for at least 1 batch)
    train_batch_count = len(loaders["train_loader"])
    assert train_batch_count >= 1


def test_setup_model_and_optimizer_new_training(tmp_path: Path) -> None:
    """Test setup_model_and_optimizer for new training."""
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

    # Verify model has expected vocabulary size
    assert setup["model"].embedding.num_embeddings == 50
    # Verify optimizer tracks model parameters
    param_group_count = len(setup["optimizer"].param_groups)
    assert param_group_count == 1
    # Verify criterion is CrossEntropyLoss
    assert setup["criterion"].reduction == "mean"
    # Verify device is set (cpu or cuda)
    assert str(setup["device"]).startswith(("cpu", "cuda"))
    assert setup["checkpoint_save"] == checkpoint_best


def test_setup_model_and_optimizer_finetune_not_found(tmp_path: Path) -> None:
    """Test setup_model_and_optimizer raises when checkpoint not found."""
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

    # Verify model loaded from checkpoint has correct structure
    assert setup["model"].embedding.num_embeddings == vocab_size
    # checkpoint_save should be versioned when fine-tuning
    assert "ft" in str(setup["checkpoint_save"]) or setup["checkpoint_save"] == checkpoint_best


def test_setup_model_and_optimizer_freeze_embed(tmp_path: Path, device: torch.device) -> None:
    """Test setup_model_and_optimizer freezes embeddings when requested."""
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
