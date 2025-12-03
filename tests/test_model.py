"""Tests for char_lstm.model module."""

from __future__ import annotations

import torch

from char_lstm.model import CharLSTM


def test_model_forward_shape(device: torch.device) -> None:
    """Test model forward pass produces valid logits that can be used for training."""
    vocab_size = 50
    embed_dim = 32
    hidden_dim = 64
    num_layers = 2
    batch_size = 4
    seq_len = 10

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    logits, hidden = model(x)

    # Verify shapes
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert len(hidden) == 2  # (h_n, c_n)
    assert hidden[0].shape == (num_layers, batch_size, hidden_dim)
    assert hidden[1].shape == (num_layers, batch_size, hidden_dim)

    # Verify logits produce valid loss when used with CrossEntropy
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

    # Loss should be finite and positive (indicates model can learn)
    loss_value = loss.item()
    assert loss_value > 0.0, f"Expected positive loss, got {loss_value}"
    assert loss_value < 100.0, f"Expected reasonable loss, got {loss_value}"


def test_model_with_hidden_state(device: torch.device) -> None:
    """Test that model accepts and returns hidden state."""
    vocab_size = 50
    embed_dim = 32
    hidden_dim = 64
    num_layers = 2
    batch_size = 4
    seq_len = 10

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # First forward pass
    logits1, hidden1 = model(x)

    # Second forward pass with hidden state
    logits2, hidden2 = model(x, hidden1)

    assert logits2.shape == logits1.shape
    # Hidden states should be different after processing more input
    assert not torch.allclose(hidden1[0], hidden2[0])


def test_model_no_hidden_returns_fresh_state(device: torch.device) -> None:
    """Test that passing None for hidden returns fresh hidden state that enables learning."""
    vocab_size = 50
    embed_dim = 32
    hidden_dim = 64
    num_layers = 2
    batch_size = 4
    seq_len = 10

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    logits, hidden = model(x, None)

    # Verify shapes
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert len(hidden) == 2
    assert hidden[0].shape == (num_layers, batch_size, hidden_dim)

    # Verify model can compute gradients with fresh hidden state
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss_before = criterion(logits.view(-1, vocab_size), targets.view(-1))

    # Clone weights before training step
    weights_before = model.embedding.weight.clone().detach()

    # Take a gradient step and verify loss changes
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_before.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Verify weights changed after optimizer step
    weights_after = model.embedding.weight
    assert not torch.allclose(weights_before, weights_after), "Expected weights to change"

    # Forward again and check loss decreased (model learned)
    logits2, _ = model(x, None)
    loss_after = criterion(logits2.view(-1, vocab_size), targets.view(-1))
    loss_after_value = loss_after.item()
    loss_before_value = loss_before.item()

    # Verify training step reduced loss (model is learning)
    assert loss_after_value < loss_before_value, (
        f"Expected loss to decrease: {loss_after_value} >= {loss_before_value}"
    )


def test_model_dropout_applied() -> None:
    """Test that dropout parameter is accepted."""
    vocab_size = 50
    embed_dim = 32
    hidden_dim = 64
    num_layers = 2
    dropout = 0.3

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout=dropout)

    # Verify LSTM has correct dropout
    assert model.lstm.dropout == dropout


def test_model_embedding_dimension() -> None:
    """Test that embedding has correct dimensions."""
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 128
    num_layers = 2

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers)

    assert model.embedding.num_embeddings == vocab_size
    assert model.embedding.embedding_dim == embed_dim


def test_model_linear_output_dimension() -> None:
    """Test that linear output layer has correct dimensions."""
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 128
    num_layers = 2

    model = CharLSTM(vocab_size, embed_dim, hidden_dim, num_layers)

    assert model.linear.in_features == hidden_dim
    assert model.linear.out_features == vocab_size
