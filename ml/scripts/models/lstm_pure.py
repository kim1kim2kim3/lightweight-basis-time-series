import torch
import torch.nn as nn

class PureThreeLayerLSTM(nn.Module):
    """Original direct multi-horizon LSTM baseline."""

    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x_hist_full: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x_hist_full)
        return self.head(hidden[-1])
