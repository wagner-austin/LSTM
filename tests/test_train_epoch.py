"""Tests for char_lstm.train epoch training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from char_lstm.data import CharDataset, build_vocab_with_unk
from char_lstm.model import CharLSTM
from char_lstm.train import (
    TrainConfig,
    TrainState,
    _create_optimizer,
    compute_gradient_norm,
    evaluate,
    get_learning_rate,
    run_final_evaluation,
    train_epoch,
)


def test_compute_gradient_norm(device: torch.device) -> None:
    """Test compute_gradient_norm calculates L2 norm of gradients that enable learning."""
    model = CharLSTM(10, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Before backward, gradients are None
    grad_norm_before = compute_gradient_norm(model)
    assert grad_norm_before == 0.0

    # Create a simple forward/backward pass to generate gradients
    x = torch.randint(0, 10, (2, 5), device=device)
    targets = torch.randint(0, 10, (2, 5), device=device)
    logits, _ = model(x)
    loss_before = criterion(logits.view(-1, 10), targets.view(-1))
    loss_before.backward()

    # After backward, gradients should be non-zero
    grad_norm_after = compute_gradient_norm(model)
    assert grad_norm_after > 0.0

    # Clone weights before optimizer step
    weights_before = next(model.parameters()).clone().detach()

    # Verify gradients enable learning: apply them and check loss decreases
    optimizer.step()
    optimizer.zero_grad()

    # Verify weights changed
    weights_after = next(model.parameters())
    assert not torch.allclose(weights_before, weights_after), "Expected weights to change"

    logits2, _ = model(x)
    loss_after = criterion(logits2.view(-1, 10), targets.view(-1))
    loss_after_value = loss_after.item()
    loss_before_value = loss_before.item()

    # Loss should decrease after applying gradients (model is learning)
    assert loss_after_value < loss_before_value, (
        f"Expected loss to decrease: {loss_after_value} >= {loss_before_value}"
    )


def test_get_learning_rate() -> None:
    """Test get_learning_rate extracts LR from optimizer."""
    model = CharLSTM(10, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lr = get_learning_rate(optimizer)
    assert lr == 1e-3


def test_evaluate(device: torch.device) -> None:
    """Test evaluate function computes loss and perplexity."""
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


def test_train_epoch(tmp_path: Path, device: torch.device) -> None:
    """Test train_epoch runs one epoch of training."""
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
        "train_ratio": 0.70,
        "val_ratio": 0.15,
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
    should_continue, epoch_metrics = train_epoch(
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
    # Verify epoch_metrics contains expected keys with valid values
    assert "train_loss" in epoch_metrics
    assert epoch_metrics["train_loss"] >= 0.0
    assert state["global_step"] > 0
    assert state["best_val_loss"] < float("inf")


def test_train_epoch_early_stopping(tmp_path: Path, device: torch.device) -> None:
    """Test train_epoch triggers early stopping when no improvement."""
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
        "train_ratio": 0.70,
        "val_ratio": 0.15,
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
    should_continue, epoch_metrics = train_epoch(
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
    # Verify epoch_metrics contains expected loss value
    assert "train_loss" in epoch_metrics
    assert epoch_metrics["train_loss"] >= 0.0
    assert state["epochs_no_improve"] >= config["patience"]


def test_train_epoch_no_improvement_but_continue(tmp_path: Path, device: torch.device) -> None:
    """Test train_epoch continues when no improvement but patience not exhausted.

    This tests the branch at train.py:850->854 where epochs_no_improve is
    incremented but is still less than patience, so training continues.
    """
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
        "train_ratio": 0.70,
        "val_ratio": 0.15,
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
    should_continue, epoch_metrics = train_epoch(
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
    # Verify epoch_metrics contains expected loss value
    assert "train_loss" in epoch_metrics
    assert epoch_metrics["train_loss"] >= 0.0
    assert state["epochs_no_improve"] == 1
    assert state["epochs_no_improve"] < config["patience"]


def test_run_final_evaluation(tmp_path: Path, device: torch.device) -> None:
    """Test run_final_evaluation loads checkpoint and evaluates."""
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
