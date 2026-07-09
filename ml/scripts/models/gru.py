import torch
import torch.nn as nn


class GRUForecast(nn.Module):
    """Standard multi-layer GRU baseline for direct multi-horizon forecasting."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
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
        _, hidden = self.gru(x_hist_full)
        return self.head(hidden[-1])
