"""ML integration tests for char_lstm.train module.

These tests verify that the model actually learns, following patterns from
api/services/model-trainer/tests.
"""

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
    train_epoch,
)


def test_training_reduces_loss(tmp_path: Path, device: torch.device) -> None:
    """Integration test: verify training actually reduces loss over multiple steps.

    This test runs multiple training steps and verifies:
    1. Loss decreases from first step to last step
    2. Model weights change during training
    """
    # Create learnable pattern - repetitive so model can learn
    pattern = "abababab" * 200 + "cdcdcdcd" * 200
    stoi, _itos, vocab_size = build_vocab_with_unk(pattern)

    dataset = CharDataset(pattern, stoi, seq_len=8)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset, batch_size=16, shuffle=True
    )
    val_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(dataset, batch_size=16)

    model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Capture weights before training
    weights_before: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        weights_before[name] = param.detach().clone()

    config: TrainConfig = {
        "seq_len": 8,
        "batch_size": 16,
        "num_epochs": 3,
        "log_every": 100,
        "patience": 10,
        "lr": 1e-2,
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

    # Track losses across epochs
    epoch_losses: list[float] = []

    for epoch in range(3):
        should_continue, metrics = train_epoch(
            epoch=epoch,
            num_epochs=3,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            vocab_size=vocab_size,
            config=config,
            state=state,
            checkpoint_save=tmp_path / "dummy.pt",
        )
        epoch_losses.append(metrics["train_loss"])
        if not should_continue:
            break

    # Verify loss decreased from first to last epoch
    loss_first = epoch_losses[0]
    loss_last = epoch_losses[-1]
    assert loss_last < loss_first, (
        f"Training should reduce loss: first={loss_first:.4f}, last={loss_last:.4f}"
    )

    # Verify weights changed during training
    any_changed = False
    for name, param in model.named_parameters():
        before = weights_before[name]
        current = param.detach()
        if not torch.equal(current, before):
            any_changed = True
            break
    assert any_changed, "Model weights should change during training"


def test_freeze_embed_preserves_embeddings(device: torch.device) -> None:
    """Integration test: verify freeze_embed=True keeps embedding weights unchanged.

    This test verifies:
    1. Embedding weights remain unchanged after training
    2. Non-embedding weights DO change (proves training occurred)
    3. Loss decreases during training (model learns)
    """
    # Create model with frozen embeddings
    vocab_size = 20
    model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    for p in model.embedding.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Capture embedding weights before training
    embed_before = model.embedding.weight.detach().clone()
    # Get a non-embedding parameter to verify training occurred
    lstm_params = [p for n, p in model.named_parameters() if "lstm" in n]
    lstm_before = lstm_params[0].detach().clone()

    # Create training data
    pattern = "abababab" * 100
    stoi = {chr(ord("a") + i): i for i in range(vocab_size - 1)}
    stoi["<unk>"] = vocab_size - 1
    dataset = CharDataset(pattern, stoi, seq_len=8)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset, batch_size=16, shuffle=True
    )

    # Train for several steps, track first and last loss
    model.train()
    first_loss = 0.0
    last_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 20:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        if batch_idx == 0:
            first_loss = loss.item()
        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    # Verify loss decreased (model is learning)
    loss_before = first_loss
    loss_after = last_loss
    assert loss_after < loss_before, f"Loss should decrease: {loss_before:.4f} -> {loss_after:.4f}"

    # Verify embedding weights unchanged
    embed_after = model.embedding.weight.detach()
    assert torch.equal(embed_before, embed_after), "Embedding weights should be unchanged"

    # Verify LSTM weights changed (proves training occurred)
    lstm_after = lstm_params[0].detach()
    assert not torch.equal(lstm_before, lstm_after), "LSTM weights should change during training"


def test_gradient_flow(device: torch.device) -> None:
    """Integration test: verify all trainable parameters receive gradients.

    This test verifies:
    1. Forward pass produces finite loss
    2. Backward pass populates gradients for ALL trainable parameters
    3. No parameters are accidentally disconnected from the computation graph
    4. Applying gradients reduces loss (model can learn)
    """
    vocab_size = 20
    model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Create input
    batch_size, seq_len = 4, 16
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    model.train()
    logits, _ = model(x)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

    # Verify loss is finite
    assert loss.isfinite(), f"Loss should be finite, got {loss.item()}"
    loss_before = loss.item()

    # Backward pass
    loss.backward()

    # Verify all trainable parameters have gradients
    params_without_grad: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grad.append(name)

    assert len(params_without_grad) == 0, f"Parameters without gradients: {params_without_grad}"

    # Verify gradients are non-zero (model is actually learning)
    any_nonzero = False
    for _name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
            any_nonzero = True
            break
    assert any_nonzero, "At least some gradients should be non-zero"

    # Apply gradients and verify loss decreases
    optimizer.step()
    optimizer.zero_grad()

    logits2, _ = model(x)
    loss2 = criterion(logits2.view(-1, vocab_size), targets.view(-1))
    loss_after = loss2.item()

    assert loss_after < loss_before, (
        f"Applying gradients should reduce loss: {loss_before:.4f} -> {loss_after:.4f}"
    )


def test_save_load_consistency(tmp_path: Path, device: torch.device) -> None:
    """Integration test: verify saved model produces same outputs as original.

    This test verifies:
    1. Model can be saved and loaded
    2. Loaded model produces identical outputs to original
    """
    vocab_size = 20
    model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.0).to(device)

    # Create deterministic input
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (2, 8), device=device)
    targets = torch.randint(0, vocab_size, (2, 8), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    # Get output from original model
    model.eval()
    with torch.no_grad():
        logits_original, _ = model(x)
        loss_original = criterion(logits_original.view(-1, vocab_size), targets.view(-1))
        loss_original_value = loss_original.item()

    # Save model
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Load into new model
    loaded_model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.0)
    loaded_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    loaded_model = loaded_model.to(device)

    # Get output from loaded model
    loaded_model.eval()
    with torch.no_grad():
        logits_loaded, _ = loaded_model(x)
        loss_loaded = criterion(logits_loaded.view(-1, vocab_size), targets.view(-1))
        loss_loaded_value = loss_loaded.item()

    # Verify outputs match exactly
    assert torch.allclose(logits_original, logits_loaded), "Logits should match after load"
    assert abs(loss_original_value - loss_loaded_value) < 1e-6, (
        f"Loss should match: original={loss_original_value}, loaded={loss_loaded_value}"
    )


def test_continued_training_reduces_loss(tmp_path: Path, device: torch.device) -> None:
    """Integration test: verify continued training (fine-tuning) reduces loss.

    This test verifies:
    1. Initial training reduces loss
    2. Saving and loading preserves model state
    3. Continued training from checkpoint further reduces loss
    """
    vocab_size = 20
    pattern = "abababab" * 200 + "cdcdcdcd" * 200
    stoi = {chr(ord("a") + i): i for i in range(vocab_size - 1)}
    stoi["<unk>"] = vocab_size - 1

    dataset = CharDataset(pattern, stoi, seq_len=8)
    # Use non-shuffled loader for consistent evaluation
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset, batch_size=16, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    # Phase 1: Initial training
    model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial_losses: list[float] = []
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 30:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        initial_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Verify initial training worked
    loss_before = initial_losses[0]
    loss_after = initial_losses[-1]
    assert loss_after < loss_before, (
        f"Initial training should reduce loss: {loss_before:.4f} -> {loss_after:.4f}"
    )

    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Phase 2: Load and continue training
    loaded_model = CharLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, dropout=0.1)
    loaded_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    loaded_model = loaded_model.to(device)
    continued_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)  # Lower LR

    continued_losses: list[float] = []
    loaded_model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 30:
            break
        x, y = x.to(device), y.to(device)
        continued_optimizer.zero_grad()
        logits, _ = loaded_model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        continued_losses.append(loss.item())
        loss.backward()
        continued_optimizer.step()

    # Verify continued training reduced loss
    # Compare first third average to last third average for stability
    n = len(continued_losses)
    third = n // 3
    initial_avg = sum(continued_losses[:third]) / third
    final_avg = sum(continued_losses[-third:]) / third
    assert final_avg < initial_avg, (
        f"Continued training should reduce loss: "
        f"initial_avg={initial_avg:.4f}, final_avg={final_avg:.4f}"
    )
