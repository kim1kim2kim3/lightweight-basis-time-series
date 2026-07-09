from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OFFICIAL_TCN_HIDDEN_DIM = 32
OFFICIAL_TCN_KERNEL_SIZE = 3
OFFICIAL_TCN_DILATIONS = (1, 2, 4)
OFFICIAL_TCN_DROPOUT = 0.1


class SamePadConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_continuous_mask(B: int, T: int, n: int | float = 5, l: int | float = 0.1) -> torch.Tensor:
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t : t + l] = False
    return res


def generate_binomial_mask(B: int, T: int, p: float = 0.5) -> torch.Tensor:
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        depth: int = 10,
        mask_mode: str = "binomial",
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, mask: str | torch.Tensor | None = None) -> torch.Tensor:
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)

        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        return x.transpose(1, 2)


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder: TSEncoder, mask: str) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.encoder(input, mask=self.mask)[:, -1]


class TCNBackbone(nn.Module):
    """OnlineTSF TCN backbone inlined for local baseline visibility."""

    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__()
        encoder = TSEncoder(
            input_dims=args.enc_in + 7,
            output_dims=320,
            hidden_dims=64,
            depth=10,
        )
        self.encoder = TS2VecEncoderWrapper(encoder, mask="all_true")
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len
        self.regressor = nn.Linear(320, self.dim)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        if x_mark is None:
            x_mark = torch.zeros(*x.shape[:2], 7, device=x.device)
        x = torch.cat([x, x_mark], dim=-1)
        rep = self.encoder(x)
        y = self.regressor(rep)
        return y.reshape(len(y), self.pred_len, -1)


class TCNForecast(nn.Module):
    """TCN baseline adapted to the local multivariate-to-target pipeline."""

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        horizon: int,
        target_idx: int,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        unsupported = []
        if hidden_dim != OFFICIAL_TCN_HIDDEN_DIM:
            unsupported.append(f"hidden_dim={hidden_dim}")
        if kernel_size != OFFICIAL_TCN_KERNEL_SIZE:
            unsupported.append(f"kernel_size={kernel_size}")
        if tuple(dilations) != OFFICIAL_TCN_DILATIONS:
            unsupported.append(f"dilations={tuple(dilations)}")
        if dropout != OFFICIAL_TCN_DROPOUT:
            unsupported.append(f"dropout={dropout}")
        if unsupported:
            supported = (
                f"hidden_dim={OFFICIAL_TCN_HIDDEN_DIM}, "
                f"kernel_size={OFFICIAL_TCN_KERNEL_SIZE}, "
                f"dilations={OFFICIAL_TCN_DILATIONS}, "
                f"dropout={OFFICIAL_TCN_DROPOUT}"
            )
            raise ValueError(
                "Official OnlineTSF TCN wrapper only supports the default settings "
                f"({supported}); received {', '.join(unsupported)}."
            )
        self.target_idx = target_idx
        self.backbone = TCNBackbone(
            SimpleNamespace(
                seq_len=lookback,
                pred_len=horizon,
                enc_in=input_dim,
                c_out=input_dim,
            )
        )

    def forward(self, x_hist_full: torch.Tensor) -> torch.Tensor:
        forecast = self.backbone(x_hist_full)
        return forecast[:, :, self.target_idx]
