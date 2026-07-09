from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
    safe = values.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(safe) - torch.log1p(-safe)


def _spread_values(low: float, high: float, rows: int, cols: int) -> torch.Tensor:
    if rows <= 0 or cols <= 0:
        return torch.zeros(rows, cols, dtype=torch.float32)
    total = rows * cols
    values = torch.linspace(low, high, steps=total, dtype=torch.float32)
    return values.view(rows, cols)


def _spread_unit_interval(rows: int, cols: int) -> torch.Tensor:
    if rows <= 0 or cols <= 0:
        return torch.zeros(rows, cols, dtype=torch.float32)
    total = rows * cols
    values = torch.linspace(1.0 / (total + 1.0), total / (total + 1.0), steps=total, dtype=torch.float32)
    return values.view(rows, cols)


class CausalDepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.pad(x, (self.left_pad, 0))
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.activation(y)
        return residual + y


class CausalScaleEncoder(nn.Module):
    def __init__(self, channels: int, depth: int, kernel_size: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if not dilations:
            raise ValueError("dilations must contain at least one value.")
        blocks = []
        for idx in range(depth):
            blocks.append(
                CausalDepthwiseSeparableBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=int(dilations[idx % len(dilations)]),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class OursBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_idx: int,
        latent_groups: int = 16,
        summary_dim: int = 32,
        depth: int = 3,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.target_idx = target_idx
        self.latent_groups = latent_groups
        self.summary_dim = summary_dim

        self.input_projection = nn.Linear(input_dim, latent_groups, bias=False)
        self.encoder = CausalScaleEncoder(
            channels=latent_groups,
            depth=depth,
            kernel_size=kernel_size,
            dilations=dilations,
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(4, summary_dim),
            nn.GELU(),
            nn.Linear(summary_dim, summary_dim),
        )
        self.target_projection = nn.Linear(latent_groups, 1, bias=False)

    def summarize(self, encoded: torch.Tensor) -> torch.Tensor:
        last_state = encoded[:, :, -1]
        mean_state = encoded.mean(dim=-1)
        pooled_state = F.adaptive_avg_pool1d(encoded, 2)
        stats = torch.cat([last_state.unsqueeze(-1), mean_state.unsqueeze(-1), pooled_state], dim=-1)
        return self.summary_mlp(stats)

    def encode(self, x_hist_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        channel_mean = x_hist_full.mean(dim=1, keepdim=True)
        channel_std = x_hist_full.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x_norm = (x_hist_full - channel_mean) / channel_std

        latent = self.input_projection(x_norm)
        encoded = self.encoder(latent.transpose(1, 2))
        summary = self.summarize(encoded)
        return summary, encoded, channel_mean, channel_std

    def project_target(self, latent_forecast: torch.Tensor) -> torch.Tensor:
        return self.target_projection(latent_forecast).squeeze(-1)

    def denormalize_target(
        self,
        target_forecast_norm: torch.Tensor,
        channel_mean: torch.Tensor,
        channel_std: torch.Tensor,
    ) -> torch.Tensor:
        target_mean = channel_mean[:, 0, self.target_idx].unsqueeze(-1)
        target_std = channel_std[:, 0, self.target_idx].unsqueeze(-1)
        return target_forecast_norm * target_std + target_mean


class OursForecast(OursBackbone):
    def __init__(
        self,
        input_dim: int,
        target_idx: int,
        horizon: int,
        latent_groups: int = 16,
        summary_dim: int = 32,
        depth: int = 3,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        trend_basis_count: int = 4,
        seasonal_mode_count: int = 4,
        transient_basis_count: int = 2,
        use_router: bool = True,
        adaptive_bank: bool = True,
        use_trend_branch: bool = True,
        use_seasonal_branch: bool = True,
        use_transient_branch: bool = True,
        use_cluster_bank: bool = False,
        num_clusters: int = 3,
        cluster_bank_fixed: bool = False,
        use_local_correction: bool = False,
        local_correction_hidden_dim: int = 16,
        use_group_frequency_offset: bool = False,
        group_frequency_offset_scale: float = 0.10,
    ) -> None:
        if trend_basis_count < 2:
            raise ValueError("trend_basis_count must be at least 2.")
        if seasonal_mode_count <= 0:
            raise ValueError("seasonal_mode_count must be positive.")
        if transient_basis_count <= 0:
            raise ValueError("transient_basis_count must be positive.")
        if not any((use_trend_branch, use_seasonal_branch, use_transient_branch)):
            raise ValueError("At least one structural branch must be enabled.")
        if use_cluster_bank and num_clusters < 2:
            raise ValueError("num_clusters must be at least 2 when use_cluster_bank is enabled.")
        if use_local_correction and local_correction_hidden_dim <= 0:
            raise ValueError("local_correction_hidden_dim must be positive when use_local_correction is enabled.")
        if use_group_frequency_offset and group_frequency_offset_scale <= 0.0:
            raise ValueError("group_frequency_offset_scale must be positive when use_group_frequency_offset is enabled.")

        super().__init__(
            input_dim=input_dim,
            target_idx=target_idx,
            latent_groups=latent_groups,
            summary_dim=summary_dim,
            depth=depth,
            kernel_size=kernel_size,
            dilations=dilations,
        )
        self.horizon = horizon
        self.trend_basis_count = trend_basis_count
        self.seasonal_mode_count = seasonal_mode_count
        self.transient_basis_count = transient_basis_count
        self.trend_decay_count = trend_basis_count - 2
        self.total_basis_count = trend_basis_count + 2 * seasonal_mode_count + transient_basis_count
        self.use_router = use_router
        self.adaptive_bank = adaptive_bank
        self.use_trend_branch = use_trend_branch
        self.use_seasonal_branch = use_seasonal_branch
        self.use_transient_branch = use_transient_branch
        self.use_cluster_bank = use_cluster_bank
        self.num_clusters = num_clusters
        self.cluster_bank_fixed = cluster_bank_fixed
        self.use_local_correction = use_local_correction
        self.use_group_frequency_offset = use_group_frequency_offset
        self.group_frequency_offset_scale = group_frequency_offset_scale
        self.local_correction_scale = 0.1

        if not self.use_cluster_bank:
            self.gamma_head = nn.Linear(summary_dim, self.trend_decay_count) if self.adaptive_bank and self.trend_decay_count > 0 else None
            self.rho_head = nn.Linear(summary_dim, seasonal_mode_count) if self.adaptive_bank else None
            self.omega_head = nn.Linear(summary_dim, seasonal_mode_count) if self.adaptive_bank else None
            self.beta_head = nn.Linear(summary_dim, transient_basis_count) if self.adaptive_bank else None
            self.fixed_gamma = self._make_decay_parameter(1, self.trend_decay_count) if not self.adaptive_bank else None
            self.fixed_rho = self._make_decay_parameter(1, seasonal_mode_count) if not self.adaptive_bank else None
            self.fixed_omega = self._make_frequency_parameter(1, seasonal_mode_count) if not self.adaptive_bank else None
            self.fixed_beta = self._make_decay_parameter(1, transient_basis_count) if not self.adaptive_bank else None
            self.cluster_router = None
            self.cluster_gamma_heads = None
            self.cluster_rho_heads = None
            self.cluster_omega_heads = None
            self.cluster_beta_heads = None
            self.fixed_cluster_gamma = None
            self.fixed_cluster_rho = None
            self.fixed_cluster_omega = None
            self.fixed_cluster_beta = None
        else:
            self.gamma_head = None
            self.rho_head = None
            self.omega_head = None
            self.beta_head = None
            self.fixed_gamma = None
            self.fixed_rho = None
            self.fixed_omega = None
            self.fixed_beta = None
            self.cluster_router = nn.Sequential(
                nn.Linear(summary_dim, summary_dim),
                nn.GELU(),
                nn.Linear(summary_dim, num_clusters),
            )
            if cluster_bank_fixed:
                self.cluster_gamma_heads = None
                self.cluster_rho_heads = None
                self.cluster_omega_heads = None
                self.cluster_beta_heads = None
                self.fixed_cluster_gamma = self._make_decay_parameter(num_clusters, self.trend_decay_count)
                self.fixed_cluster_rho = self._make_decay_parameter(num_clusters, seasonal_mode_count)
                self.fixed_cluster_omega = self._make_frequency_parameter(num_clusters, seasonal_mode_count)
                self.fixed_cluster_beta = self._make_decay_parameter(num_clusters, transient_basis_count)
            else:
                self.cluster_gamma_heads = (
                    nn.ModuleList(nn.Linear(summary_dim, self.trend_decay_count) for _ in range(num_clusters))
                    if self.trend_decay_count > 0
                    else None
                )
                self.cluster_rho_heads = nn.ModuleList(nn.Linear(summary_dim, seasonal_mode_count) for _ in range(num_clusters))
                self.cluster_omega_heads = nn.ModuleList(nn.Linear(summary_dim, seasonal_mode_count) for _ in range(num_clusters))
                self.cluster_beta_heads = nn.ModuleList(nn.Linear(summary_dim, transient_basis_count) for _ in range(num_clusters))
                self.fixed_cluster_gamma = None
                self.fixed_cluster_rho = None
                self.fixed_cluster_omega = None
                self.fixed_cluster_beta = None

        self.trend_head = nn.Sequential(
            nn.Linear(summary_dim, 2 * summary_dim),
            nn.GELU(),
            nn.Linear(2 * summary_dim, trend_basis_count),
        )
        self.seasonal_head = nn.Sequential(
            nn.Linear(summary_dim, 2 * summary_dim),
            nn.GELU(),
            nn.Linear(2 * summary_dim, 2 * seasonal_mode_count),
        )
        self.transient_head = nn.Sequential(
            nn.Linear(summary_dim, 2 * summary_dim),
            nn.GELU(),
            nn.Linear(2 * summary_dim, transient_basis_count),
        )
        self.router_head = nn.Sequential(
            nn.Linear(summary_dim, summary_dim),
            nn.GELU(),
            nn.Linear(summary_dim, 3),
        )
        self.frequency_offset_head = (
            nn.Linear(summary_dim, seasonal_mode_count)
            if self.use_group_frequency_offset and self.use_seasonal_branch
            else None
        )
        self.local_correction_head = (
            nn.Sequential(
                nn.Linear(summary_dim, local_correction_hidden_dim),
                nn.GELU(),
                nn.Linear(local_correction_hidden_dim, horizon),
            )
            if self.use_local_correction
            else None
        )
        if self.local_correction_head is not None:
            last = self.local_correction_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _make_decay_parameter(self, rows: int, cols: int) -> nn.Parameter | None:
        if cols <= 0:
            return None
        desired = _spread_values(0.7, 0.98, rows, cols)
        if rows == 1:
            desired = desired.squeeze(0)
        return nn.Parameter(_inverse_sigmoid(desired))

    def _make_frequency_parameter(self, rows: int, cols: int) -> nn.Parameter | None:
        if cols <= 0:
            return None
        desired = _spread_unit_interval(rows, cols)
        if rows == 1:
            desired = desired.squeeze(0)
        return nn.Parameter(_inverse_sigmoid(desired))

    def _branch_mask(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(
            [float(self.use_trend_branch), float(self.use_seasonal_branch), float(self.use_transient_branch)]
        )

    def _stack_cluster_head_outputs(
        self,
        heads: nn.ModuleList | None,
        global_summary: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        batch_size = global_summary.shape[0]
        if feature_count <= 0:
            return global_summary.new_zeros((batch_size, self.num_clusters, 0))
        if heads is None:
            raise RuntimeError("Cluster heads are required for adaptive cluster-bank mode.")
        outputs = [head(global_summary) for head in heads]
        return torch.stack(outputs, dim=1)

    def resolve_router_weights(self, summary: torch.Tensor) -> torch.Tensor | None:
        branch_mask = self._branch_mask(summary)
        if not self.use_router:
            return None
        logits = self.router_head(summary)
        inactive = branch_mask <= 0
        if bool(inactive.any()):
            logits = logits.masked_fill(inactive.view(1, 1, 3), -1e9)
        return torch.softmax(logits, dim=-1)

    def build_basis(
        self,
        gamma: torch.Tensor,
        rho: torch.Tensor,
        omega: torch.Tensor,
        beta: torch.Tensor,
        effective_omega: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = rho.shape[0]
        group_count = effective_omega.shape[1]
        device = rho.device
        dtype = rho.dtype
        eps = torch.tensor(1e-5, device=device, dtype=dtype)
        h = torch.arange(1, self.horizon + 1, device=device, dtype=dtype).view(1, self.horizon, 1)

        trend_pieces = [
            torch.ones(batch_size, self.horizon, 1, device=device, dtype=dtype),
            (h / float(self.horizon)).expand(batch_size, -1, -1),
        ]
        if self.trend_decay_count > 0:
            gamma_pow = torch.pow(gamma.unsqueeze(1), h)
            damped = (1.0 - gamma_pow) / (1.0 - gamma.unsqueeze(1) + eps)
            trend_pieces.append(damped)
        trend_basis = torch.cat(trend_pieces, dim=-1)

        seasonal_h = h.view(1, self.horizon, 1, 1)
        rho_pow = torch.pow(rho.unsqueeze(1).unsqueeze(1), seasonal_h)
        cos_part = rho_pow * torch.cos(seasonal_h * effective_omega.unsqueeze(1))
        sin_part = rho_pow * torch.sin(seasonal_h * effective_omega.unsqueeze(1))
        seasonal_basis = torch.cat([cos_part, sin_part], dim=-1)

        transient_basis = torch.pow(beta.unsqueeze(1), h)
        trend_basis_group = trend_basis.unsqueeze(2).expand(-1, -1, group_count, -1)
        transient_basis_group = transient_basis.unsqueeze(2).expand(-1, -1, group_count, -1)
        full_basis = torch.cat([trend_basis_group, seasonal_basis, transient_basis_group], dim=-1)
        return {
            "trend_basis": trend_basis,
            "seasonal_basis": seasonal_basis,
            "transient_basis": transient_basis,
            "full_basis": full_basis,
        }

    def _mix_cluster_bank(self, global_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cluster_router is None:
            raise RuntimeError("Cluster router is required when use_cluster_bank is enabled.")
        cluster_weights = torch.softmax(self.cluster_router(global_summary), dim=-1)
        batch_size = global_summary.shape[0]

        if self.cluster_bank_fixed:
            gamma_clusters = (
                torch.sigmoid(self.fixed_cluster_gamma).unsqueeze(0).expand(batch_size, -1, -1)
                if self.fixed_cluster_gamma is not None
                else global_summary.new_zeros((batch_size, self.num_clusters, 0))
            )
            rho_clusters = torch.sigmoid(self.fixed_cluster_rho).unsqueeze(0).expand(batch_size, -1, -1)
            omega_clusters = math.pi * torch.sigmoid(self.fixed_cluster_omega).unsqueeze(0).expand(batch_size, -1, -1)
            beta_clusters = torch.sigmoid(self.fixed_cluster_beta).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            gamma_clusters = torch.sigmoid(
                self._stack_cluster_head_outputs(self.cluster_gamma_heads, global_summary, self.trend_decay_count)
            )
            rho_clusters = torch.sigmoid(
                self._stack_cluster_head_outputs(self.cluster_rho_heads, global_summary, self.seasonal_mode_count)
            )
            omega_clusters = math.pi * torch.sigmoid(
                self._stack_cluster_head_outputs(self.cluster_omega_heads, global_summary, self.seasonal_mode_count)
            )
            beta_clusters = torch.sigmoid(
                self._stack_cluster_head_outputs(self.cluster_beta_heads, global_summary, self.transient_basis_count)
            )

        gamma = torch.einsum("bq,bqk->bk", cluster_weights, gamma_clusters) if self.trend_decay_count > 0 else global_summary.new_zeros((batch_size, 0))
        rho = torch.einsum("bq,bqk->bk", cluster_weights, rho_clusters)
        omega = torch.einsum("bq,bqk->bk", cluster_weights, omega_clusters)
        beta = torch.einsum("bq,bqk->bk", cluster_weights, beta_clusters)
        return gamma, rho, omega, beta, cluster_weights

    def bank_parameters(
        self,
        global_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = global_summary.shape[0]
        if self.use_cluster_bank:
            gamma, rho, omega, beta, cluster_weights = self._mix_cluster_bank(global_summary)
            return gamma, rho, omega, beta, cluster_weights

        if self.adaptive_bank:
            gamma = torch.sigmoid(self.gamma_head(global_summary)) if self.gamma_head is not None else global_summary.new_zeros((batch_size, 0))
            rho = torch.sigmoid(self.rho_head(global_summary)) if self.rho_head is not None else global_summary.new_zeros((batch_size, self.seasonal_mode_count))
            omega = math.pi * torch.sigmoid(self.omega_head(global_summary)) if self.omega_head is not None else global_summary.new_zeros((batch_size, self.seasonal_mode_count))
            beta = torch.sigmoid(self.beta_head(global_summary)) if self.beta_head is not None else global_summary.new_zeros((batch_size, self.transient_basis_count))
            return gamma, rho, omega, beta, None

        gamma = (
            torch.sigmoid(self.fixed_gamma).unsqueeze(0).expand(batch_size, -1)
            if self.fixed_gamma is not None
            else global_summary.new_zeros((batch_size, 0))
        )
        rho = torch.sigmoid(self.fixed_rho).unsqueeze(0).expand(batch_size, -1)
        omega = math.pi * torch.sigmoid(self.fixed_omega).unsqueeze(0).expand(batch_size, -1)
        beta = torch.sigmoid(self.fixed_beta).unsqueeze(0).expand(batch_size, -1)
        return gamma, rho, omega, beta, None

    def resolve_effective_omega(
        self,
        summary: torch.Tensor,
        omega: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        base_omega = omega.unsqueeze(1).expand(-1, summary.shape[1], -1)
        if self.frequency_offset_head is None:
            return None, base_omega

        delta_omega = self.group_frequency_offset_scale * torch.tanh(self.frequency_offset_head(summary))
        effective_omega = (base_omega + delta_omega).clamp(min=1e-4, max=(math.pi - 1e-4))
        return delta_omega, effective_omega

    def compute_local_correction(self, summary: torch.Tensor) -> torch.Tensor | None:
        if self.local_correction_head is None:
            return None
        group_correction = self.local_correction_head(summary)
        target_weights = self.target_projection.weight.squeeze(0)
        return self.local_correction_scale * torch.einsum("bgh,g->bh", group_correction, target_weights)

    def forward(self, x_hist_full: torch.Tensor) -> dict[str, torch.Tensor | bool | None]:
        summary, _encoded, channel_mean, channel_std = self.encode(x_hist_full)
        global_summary = summary.mean(dim=1)
        gamma, rho, omega, beta, cluster_weights = self.bank_parameters(global_summary)
        delta_omega, effective_omega = self.resolve_effective_omega(summary, omega)
        basis_parts = self.build_basis(
            gamma=gamma,
            rho=rho,
            omega=omega,
            beta=beta,
            effective_omega=effective_omega,
        )

        trend_coeff = self.trend_head(summary)
        seasonal_coeff = self.seasonal_head(summary)
        transient_coeff = self.transient_head(summary)

        if not self.use_trend_branch:
            trend_coeff = torch.zeros_like(trend_coeff)
        if not self.use_seasonal_branch:
            seasonal_coeff = torch.zeros_like(seasonal_coeff)
        if not self.use_transient_branch:
            transient_coeff = torch.zeros_like(transient_coeff)

        router_weights = self.resolve_router_weights(summary)
        if router_weights is not None:
            trend_coeff = trend_coeff * router_weights[..., 0:1]
            seasonal_coeff = seasonal_coeff * router_weights[..., 1:2]
            transient_coeff = transient_coeff * router_weights[..., 2:3]

        coefficients = torch.cat([trend_coeff, seasonal_coeff, transient_coeff], dim=-1)
        latent_forecast = torch.einsum("bhgk,bgk->bhg", basis_parts["full_basis"], coefficients)
        target_forecast_norm = self.project_target(latent_forecast)
        local_correction = self.compute_local_correction(summary)
        if local_correction is not None:
            target_forecast_norm = target_forecast_norm + local_correction
        target_pred = self.denormalize_target(target_forecast_norm, channel_mean, channel_std)
        return {
            "target_pred": target_pred,
            "router_weights": router_weights,
            "router_enabled": self.use_router,
            "cluster_weights": cluster_weights,
            "trend_coeff": trend_coeff,
            "seasonal_coeff": seasonal_coeff,
            "transient_coeff": transient_coeff,
            "seasonal_basis": basis_parts["seasonal_basis"],
            "gamma": gamma,
            "rho": rho,
            "omega": omega,
            "beta": beta,
            "delta_omega": delta_omega,
            "effective_omega": effective_omega,
            "local_correction": local_correction,
        }


class OursDirectHeadForecast(OursBackbone):
    def __init__(
        self,
        input_dim: int,
        target_idx: int,
        horizon: int,
        latent_groups: int = 16,
        summary_dim: int = 32,
        depth: int = 3,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            target_idx=target_idx,
            latent_groups=latent_groups,
            summary_dim=summary_dim,
            depth=depth,
            kernel_size=kernel_size,
            dilations=dilations,
        )
        self.horizon = horizon
        self.direct_head = nn.Sequential(
            nn.Linear(summary_dim, 2 * summary_dim),
            nn.GELU(),
            nn.Linear(2 * summary_dim, horizon),
        )

    def forward(self, x_hist_full: torch.Tensor) -> dict[str, torch.Tensor | bool | None]:
        summary, _encoded, channel_mean, channel_std = self.encode(x_hist_full)
        latent_forecast = self.direct_head(summary).transpose(1, 2)
        target_forecast_norm = self.project_target(latent_forecast)
        target_pred = self.denormalize_target(target_forecast_norm, channel_mean, channel_std)
        return {
            "target_pred": target_pred,
            "router_weights": None,
            "router_enabled": False,
            "cluster_weights": None,
            "trend_coeff": None,
            "seasonal_coeff": None,
            "transient_coeff": None,
            "seasonal_basis": None,
            "gamma": None,
            "rho": None,
            "omega": None,
            "beta": None,
            "delta_omega": None,
            "effective_omega": None,
            "local_correction": None,
        }
