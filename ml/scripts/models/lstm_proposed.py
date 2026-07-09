import torch
import torch.nn as nn


class ThreeLayerLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        target_idx: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_idx = target_idx
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x_hist_full: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x_hist_full)
        last_value = x_hist_full[:, -1, self.target_idx].unsqueeze(-1)
        residual = self.head(hidden[-1])
        return residual + last_value
