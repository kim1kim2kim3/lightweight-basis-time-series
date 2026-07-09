from collections.abc import Sequence
import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """A simple moving-average block used for internal fallback decomposition."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """
    Internal fallback decomposition.
    """

    def __init__(self, trend_kernel: int = 25, season_kernel: int = 3):
        super().__init__()
        self.trend_ma = MovingAverage(trend_kernel)
        self.season_ma = MovingAverage(season_kernel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trend = self.trend_ma(x)
        remainder = x - trend
        season = self.season_ma(remainder)
        resid = remainder - season
        return trend, season, resid


class TrendHead(nn.Module):
    def __init__(self, lookback: int, horizon: int, add_last_value: bool = True):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)
        self.add_last_value = add_last_value

    def forward(self, trend: torch.Tensor) -> torch.Tensor:
        x = trend.squeeze(-1)
        last_val = x[:, -1:]
        x = x - last_val
        y = self.linear(x)
        if self.add_last_value:
            y = y + last_val
        return y.unsqueeze(-1)


class SeasonalHead(nn.Module):
    def __init__(self, lookback: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, season: torch.Tensor) -> torch.Tensor:
        x = season.squeeze(-1)
        y = self.linear(x)
        return y.unsqueeze(-1)


class ResidualTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_dim: int = 32,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_in = in_channels
        for dilation in dilations:
            padding = dilation
            layers.append(
                nn.Conv1d(
                    in_channels=current_in,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_in = hidden_dim
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        last_hidden = h[:, :, -1]
        y = self.fc(last_hidden)
        return y.unsqueeze(-1)


class MultivariateSTLTCN(nn.Module):
    """Original STL-TCN baseline before persistence-aware improvements."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        exog_dim: int,
        hidden_dim: int = 32,
        use_pre_decomposed: bool = True,
        trend_kernel: int = 25,
        season_kernel: int = 3,
        tcn_dilations: Sequence[int] = (1, 2),
        tcn_dropout: float = 0.1,
        use_trend_branch: bool = True,
        use_season_branch: bool = True,
        use_resid_branch: bool = True,
    ):
        super().__init__()
        if not (use_trend_branch or use_season_branch or use_resid_branch):
            raise ValueError("At least one STL-TCN branch must be enabled.")
        self.use_pre_decomposed = use_pre_decomposed
        self.use_trend_branch = use_trend_branch
        self.use_season_branch = use_season_branch
        self.use_resid_branch = use_resid_branch
        if not use_pre_decomposed:
            self.decomposer = SeriesDecomp(
                trend_kernel=trend_kernel,
                season_kernel=season_kernel,
            )

        if self.use_trend_branch:
            self.trend_head = TrendHead(lookback, horizon, add_last_value=True)
        if self.use_season_branch:
            self.seasonal_head = SeasonalHead(lookback, horizon)
        if self.use_resid_branch:
            self.resid_tcn = ResidualTCN(
                in_channels=1 + exog_dim,
                horizon=horizon,
                hidden_dim=hidden_dim,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
            )

    def forward(
        self,
        x: torch.Tensor | None = None,
        x_exog: torch.Tensor | None = None,
        pre_trend: torch.Tensor | None = None,
        pre_season: torch.Tensor | None = None,
        pre_resid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_pre_decomposed:
            if pre_trend is None or pre_season is None or pre_resid is None:
                raise ValueError("Pre-decomposed inputs are required when use_pre_decomposed=True.")
            trend = pre_trend
            season = pre_season
            resid = pre_resid
        else:
            if x is None:
                raise ValueError("Raw target history is required for internal decomposition.")
            trend, season, resid = self.decomposer(x)

        outputs: list[torch.Tensor] = []
        if self.use_trend_branch:
            outputs.append(self.trend_head(trend))
        if self.use_season_branch:
            outputs.append(self.seasonal_head(season))
        if self.use_resid_branch:
            if x_exog is not None:
                z = torch.cat([resid, x_exog], dim=-1)
            else:
                z = resid
            outputs.append(self.resid_tcn(z.permute(0, 2, 1)))

        result = outputs[0]
        for branch_output in outputs[1:]:
            result = result + branch_output
        return result
