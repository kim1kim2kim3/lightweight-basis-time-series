import math
from math import sqrt
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


OFFICIAL_PATCHTST_PATCH_LEN = 16
OFFICIAL_PATCHTST_PATCH_STRIDE = 8
OFFICIAL_PATCHTST_D_MODEL = 512
OFFICIAL_PATCHTST_NUM_LAYERS = 2
OFFICIAL_PATCHTST_NUM_HEADS = 8
OFFICIAL_PATCHTST_FF_DIM = 2048
OFFICIAL_PATCHTST_DROPOUT = 0.1


class TriangularCausalMask:
    def __init__(self, batch_size: int, length: int, device: torch.device | str = "cpu") -> None:
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: TriangularCausalMask | None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, length, _, embed_dim = queries.shape
        _, source_length, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(embed_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch_size, length, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -torch.inf)

        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        value = torch.einsum("bhls,bshd->blhd", attention, values)

        if self.output_attention:
            return value.contiguous(), attention
        return value.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: TriangularCausalMask | None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, length, _ = queries.shape
        _, source_length, _ = keys.shape
        heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, length, heads, -1)
        keys = self.key_projection(keys).view(batch_size, source_length, heads, -1)
        values = self.value_projection(values).view(batch_size, source_length, heads, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(batch_size, length, -1)
        return self.out_projection(out), attn


class Transpose(nn.Module):
    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(*self.dims)
        if self.contiguous:
            return x.contiguous()
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: TriangularCausalMask | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers: list[EncoderLayer],
        conv_layers: list[nn.Module] | None = None,
        norm_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: TriangularCausalMask | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class PatchTSTBackbone(nn.Module):
    """PatchTST backbone derived from archive/external/Time-Series-Library for local baseline visibility."""

    def __init__(
        self,
        configs: SimpleNamespace,
        patch_len: int = OFFICIAL_PATCHTST_PATCH_LEN,
        stride: int = OFFICIAL_PATCHTST_PATCH_STRIDE,
    ) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        else:
            raise ValueError(f"PatchTSTBackbone only supports forecasting tasks, got {self.task_name!r}.")

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out, _attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]


class PatchTSTForecast(nn.Module):
    """PatchTST baseline adapted to the local multivariate-to-target pipeline."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        target_idx: int,
        patch_len: int = OFFICIAL_PATCHTST_PATCH_LEN,
        patch_stride: int = OFFICIAL_PATCHTST_PATCH_STRIDE,
        d_model: int = OFFICIAL_PATCHTST_D_MODEL,
        num_layers: int = OFFICIAL_PATCHTST_NUM_LAYERS,
        num_heads: int = OFFICIAL_PATCHTST_NUM_HEADS,
        ff_dim: int = OFFICIAL_PATCHTST_FF_DIM,
        dropout: float = OFFICIAL_PATCHTST_DROPOUT,
        factor: int = 1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.target_idx = target_idx
        self.backbone = PatchTSTBackbone(
            SimpleNamespace(
                task_name="long_term_forecast",
                seq_len=lookback,
                pred_len=horizon,
                d_model=d_model,
                factor=factor,
                n_heads=num_heads,
                d_ff=ff_dim,
                dropout=dropout,
                activation=activation,
                e_layers=num_layers,
                enc_in=input_dim,
            ),
            patch_len=patch_len,
            stride=patch_stride,
        )

    def forward(self, x_hist_full: torch.Tensor) -> torch.Tensor:
        forecast = self.backbone(x_hist_full, None, None, None)
        return forecast[:, :, self.target_idx]
