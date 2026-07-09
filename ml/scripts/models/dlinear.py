from types import SimpleNamespace

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block from OnlineTSF DLinear."""

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition block from OnlineTSF DLinear."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearBackbone(nn.Module):
    """OnlineTSF DLinear backbone inlined for local baseline visibility."""

    def __init__(self, configs: SimpleNamespace) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class DLinearForecast(nn.Module):
    """DLinear baseline adapted to the local multivariate-to-target pipeline."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        target_idx: int,
        moving_avg_kernel: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        if moving_avg_kernel != 25:
            raise ValueError("Official OnlineTSF DLinear uses a fixed moving-average kernel of 25.")
        self.target_idx = target_idx
        self.backbone = DLinearBackbone(
            SimpleNamespace(
                seq_len=lookback,
                pred_len=horizon,
                enc_in=input_dim,
                individual=individual,
            )
        )

    def forward(self, x_hist_full: torch.Tensor) -> torch.Tensor:
        forecast = self.backbone(x_hist_full)
        return forecast[:, :, self.target_idx]
