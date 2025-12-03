"""Character-level LSTM model for language modeling."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class CharLSTM(nn.Module):
    """Character-level LSTM for next-character prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm: nn.LSTM = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear: nn.Linear = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

    def forward(
        self,
        x: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len) containing token indices.
            hidden: Optional tuple of (h_n, c_n) hidden states.

        Returns:
            Tuple of (logits, hidden_state) where:
            - logits: Shape (batch, seq_len, vocab_size)
            - hidden_state: Tuple of (h_n, c_n)
        """
        embedded: Tensor = self.embedding(x)

        # LSTM returns (output, (h_n, c_n)) - explicitly typed
        lstm_out: tuple[Tensor, tuple[Tensor, Tensor]] = self.lstm(embedded, hidden)
        out: Tensor = lstm_out[0]
        hc: tuple[Tensor, Tensor] = lstm_out[1]

        logits: Tensor = self.linear(out)

        return logits, hc
