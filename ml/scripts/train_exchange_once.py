from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficiency_metrics import (
    DEFAULT_ACCURACY_TOLERANCE,
    EFFICIENCY_ERROR_METRICS,
    add_parameter_efficiency_columns,
    add_epes_cpls_columns,
    add_legacy_efficiency_columns,
    build_pairwise_cpl_dataframe,
    validate_efficiency_error_metric,
)
from statsmodels.tsa.seasonal import MSTL, STL
from torch.utils.data import DataLoader, Dataset

from models.dlinear import DLinearForecast
from models.gru import GRUForecast
from models.ours import OursDirectHeadForecast, OursForecast
from models.lstm_proposed import ThreeLayerLSTM
from models.lstm_pure import PureThreeLayerLSTM
from models.patchtst import (
    OFFICIAL_PATCHTST_D_MODEL,
    OFFICIAL_PATCHTST_DROPOUT,
    OFFICIAL_PATCHTST_FF_DIM,
    OFFICIAL_PATCHTST_NUM_HEADS,
    OFFICIAL_PATCHTST_NUM_LAYERS,
    OFFICIAL_PATCHTST_PATCH_LEN,
    OFFICIAL_PATCHTST_PATCH_STRIDE,
    PatchTSTForecast,
)
from results_layout import build_run_dir
from models.stl_tcn import MultivariateSTLTCN
from models.tcn import TCNForecast

MODEL_ORDER = [
    "ours",
    "ours_no_router",
    "ours_fixed_bank",
    "ours_cluster_bank",
    "ours_cluster_bank_fixed",
    "ours_extended",
    "ours_no_transient",
    "ours_no_seasonal",
    "ours_trend_only",
    "ours_direct_head",
    "stl_tcn",
    "dlinear",
    "tcn",
    "gru",
    "lstm_pure",
    "patchtst",
    "lstm_proposed",
]
DEFAULT_BENCHMARK_MODELS = (
    "dlinear",
    "gru",
    "lstm_pure",
    "patchtst",
)
OURS_MODEL_NAMES = frozenset(
    {
        "ours",
        "ours_no_router",
        "ours_fixed_bank",
        "ours_cluster_bank",
        "ours_cluster_bank_fixed",
        "ours_extended",
        "ours_no_transient",
        "ours_no_seasonal",
        "ours_trend_only",
        "ours_direct_head",
    }
)
TIME_ORDER_COLUMN_CANDIDATES = ("date", "datetime", "timestamp", "ds")
CROSS_SPLIT_LABEL = "cross_split"
OPTIMIZER_NAMES = ("adamw", "adam")
LOSS_NAMES = ("auto", "mse", "mae", "l1")
LR_SCHEDULES = ("none", "tslib_type1")
EARLY_STOPPING_METRICS = ("selection_metric", "val_loss")


@dataclass
class ExperimentConfig:
    data_path: str
    results_dir: str
    target_col: str
    dataset_name: str | None = None
    lookback: int = 56
    horizon: int = 14
    stl_period: int = 5
    batch_size: int = 64
    epochs: int = 12
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    loss_name: str = "auto"
    lr_schedule: str = "none"
    early_stopping_metric: str = "selection_metric"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    efficiency_error_metric: str = "mae"
    lstm_proposed_hidden_dim: int = 64
    lstm_pure_hidden_dim: int = 64
    gru_hidden_dim: int = 64
    gru_num_layers: int = 2
    gru_dropout: float = 0.1
    stl_hidden_dim: int = 32
    tcn_hidden_dim: int = 32
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    patchtst_patch_len: int = OFFICIAL_PATCHTST_PATCH_LEN
    patchtst_patch_stride: int = OFFICIAL_PATCHTST_PATCH_STRIDE
    patchtst_d_model: int = OFFICIAL_PATCHTST_D_MODEL
    patchtst_num_layers: int = OFFICIAL_PATCHTST_NUM_LAYERS
    patchtst_num_heads: int = OFFICIAL_PATCHTST_NUM_HEADS
    patchtst_ff_dim: int = OFFICIAL_PATCHTST_FF_DIM
    patchtst_dropout: float = OFFICIAL_PATCHTST_DROPOUT
    dlinear_moving_avg_kernel: int = 25
    dlinear_individual: bool = False
    decomposition_mode: str = "stl"
    mstl_periods: tuple[int, ...] = ()
    stl_use_trend_branch: bool = True
    stl_use_season_branch: bool = True
    stl_use_resid_branch: bool = True
    ours_latent_groups: int = 16
    ours_summary_dim: int = 32
    ours_depth: int = 3
    ours_kernel_size: int = 3
    ours_dilations: tuple[int, ...] = (1, 2, 4)
    ours_trend_basis_count: int = 4
    ours_seasonal_mode_count: int = 4
    ours_transient_basis_count: int = 2
    ours_use_router: bool = True
    ours_adaptive_bank: bool = True
    ours_use_trend_branch: bool = True
    ours_use_seasonal_branch: bool = True
    ours_use_transient_branch: bool = True
    ours_coeff_sparsity_weight: float = 0.0
    ours_seasonal_diversity_weight: float = 0.0
    ours_seasonal_diversity_tau: float = 0.25
    ours_router_entropy_weight: float = 0.0
    ours_num_clusters: int = 3
    ours_use_cluster_bank: bool = False
    ours_cluster_bank_fixed: bool = False
    ours_use_local_correction: bool = False
    ours_local_correction_hidden_dim: int = 16
    ours_use_group_frequency_offset: bool = False
    ours_group_frequency_offset_scale: float = 0.10
    deterministic: bool = False
    export_all_window_predictions: bool = False
    export_eval_predictions: bool = True
    export_ours_diagnostics: bool = True
    models: tuple[str, ...] = DEFAULT_BENCHMARK_MODELS


class FeatureScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        self.mean_ = values.mean(axis=0, dtype=np.float64)
        self.std_ = values.std(axis=0, dtype=np.float64)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform().")
        return (values - self.mean_) / self.std_

    def inverse_target(self, values: np.ndarray, target_idx: int) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fit before inverse_target().")
        return values * self.std_[target_idx] + self.mean_[target_idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_determinism(enabled: bool) -> None:
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(not enabled)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(not enabled)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(enabled, warn_only=enabled)


def seed_for_model(base_seed: int, model_name: str) -> int:
    return base_seed + MODEL_ORDER.index(model_name)


def infer_dataset_name(config: ExperimentConfig) -> str:
    if config.dataset_name is not None and config.dataset_name.strip():
        return config.dataset_name.strip()
    return Path(config.data_path).stem


def normalize_periods(raw: tuple[int, ...] | list[int] | int | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, int):
        return (int(raw),) if int(raw) > 0 else ()
    periods = tuple(int(value) for value in raw if int(value) > 0)
    return periods


def resolve_decomposition_mode(config: ExperimentConfig) -> str:
    mode = str(config.decomposition_mode).strip().lower()
    if mode not in {"stl", "mstl"}:
        raise ValueError(f"Unsupported decomposition_mode: {config.decomposition_mode}")
    return mode


def resolve_mstl_periods(config: ExperimentConfig) -> tuple[int, ...]:
    periods = normalize_periods(config.mstl_periods)
    if periods:
        return periods
    return (int(config.stl_period),) if int(config.stl_period) > 0 else ()


def resolve_season_periods(config: ExperimentConfig) -> tuple[int, ...]:
    mode = resolve_decomposition_mode(config)
    if mode == "mstl":
        periods = resolve_mstl_periods(config)
        if not periods:
            raise ValueError("mstl requires at least one positive period.")
        return periods
    period = int(config.stl_period)
    if period <= 0:
        raise ValueError(f"stl_period must be positive, got {config.stl_period}")
    return (period,)


def load_time_ordered_frame(data_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(data_path)
    metadata: dict[str, object] = {
        "time_order_source": "row_order",
        "time_order_column": None,
        "time_order_was_sorted": False,
        "time_order_warning": None,
    }

    lower_to_original = {column.lower(): column for column in df.columns}
    matched_column = next(
        (lower_to_original[candidate] for candidate in TIME_ORDER_COLUMN_CANDIDATES if candidate in lower_to_original),
        None,
    )
    if matched_column is None:
        warning_message = "No explicit time column found; assuming CSV row order is temporal."
        metadata["time_order_warning"] = warning_message
        warnings.warn(warning_message)
        return df, metadata

    parsed = pd.to_datetime(df[matched_column], errors="coerce")
    metadata["time_order_column"] = matched_column
    if parsed.isna().any():
        warning_message = (
            f"Could not fully parse time column '{matched_column}'; "
            "falling back to CSV row order."
        )
        metadata["time_order_warning"] = warning_message
        warnings.warn(warning_message)
        return df.drop(columns=[matched_column]), metadata

    metadata["time_order_source"] = "explicit_column"
    if parsed.is_monotonic_increasing:
        return df.drop(columns=[matched_column]), metadata

    ordered = (
        df.assign(__time_order__=parsed)
        .sort_values("__time_order__", kind="mergesort")
        .drop(columns="__time_order__")
        .reset_index(drop=True)
    )
    warning_message = f"Sorted rows by non-monotonic time column '{matched_column}'."
    metadata["time_order_was_sorted"] = True
    metadata["time_order_warning"] = warning_message
    warnings.warn(warning_message)
    return ordered.drop(columns=[matched_column]), metadata


def build_split_indices(
    total_len: int,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    train_end_indices = np.arange(lookback, train_end - horizon + 1)
    val_end_indices = np.arange(train_end, val_end - horizon + 1)
    test_end_indices = np.arange(val_end, total_len - horizon + 1)
    return train_end_indices, val_end_indices, test_end_indices, train_end, val_end


def resolve_split_boundaries(
    dataset_name: str,
    total_len: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    normalized_name = dataset_name.strip().lower()
    ett_split_boundaries = {
        "etth1": (12 * 30 * 24, (12 + 4) * 30 * 24),
        "etth2": (12 * 30 * 24, (12 + 4) * 30 * 24),
        "ettm1": (12 * 30 * 24 * 4, (12 + 4) * 30 * 24 * 4),
        "ettm2": (12 * 30 * 24 * 4, (12 + 4) * 30 * 24 * 4),
    }
    if normalized_name in ett_split_boundaries:
        train_end, val_end = ett_split_boundaries[normalized_name]
        if total_len <= val_end:
            raise ValueError(
                f"{dataset_name} requires at least {val_end + 1} rows for the official ETT split, got {total_len}."
            )
        return train_end, val_end

    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    return train_end, val_end


def build_split_indices_from_boundaries(
    total_len: int,
    lookback: int,
    horizon: int,
    train_end: int,
    val_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    train_end_indices = np.arange(lookback, train_end - horizon + 1)
    val_end_indices = np.arange(train_end, val_end - horizon + 1)
    test_end_indices = np.arange(val_end, total_len - horizon + 1)
    return train_end_indices, val_end_indices, test_end_indices, train_end, val_end


def decompose_history(
    y_hist: np.ndarray,
    period: int,
    *,
    decomposition_mode: str = "stl",
    mstl_periods: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    mode = decomposition_mode.strip().lower()
    periods = normalize_periods(mstl_periods)
    min_period = max(periods) if mode == "mstl" and periods else period
    min_required = max(min_period * 2, 8)

    component_count = len(periods) if mode == "mstl" and periods else 1

    def fallback_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        trend = np.full_like(y_hist, y_hist.mean())
        season = np.zeros_like(y_hist)
        resid = y_hist - trend
        season_components = np.zeros((len(y_hist), component_count), dtype=np.float32)
        if component_count == 1:
            season_components[:, 0] = season
        return trend, season, resid, season_components, True

    if len(y_hist) < min_required:
        return fallback_outputs()

    try:
        if mode == "mstl" and periods:
            mstl = MSTL(y_hist, periods=list(periods), stl_kwargs={"robust": True}).fit()
            trend = np.asarray(mstl.trend, dtype=np.float32)
            seasonal = np.asarray(mstl.seasonal, dtype=np.float32)
            season_components = seasonal if seasonal.ndim == 2 else seasonal[:, None]
            season = season_components.sum(axis=1, dtype=np.float32)
            resid = np.asarray(mstl.resid, dtype=np.float32)
        else:
            stl = STL(y_hist, period=period, robust=True).fit()
            trend = np.asarray(stl.trend, dtype=np.float32)
            season = np.asarray(stl.seasonal, dtype=np.float32)
            resid = np.asarray(stl.resid, dtype=np.float32)
            season_components = season[:, None]
        return trend, season, resid, season_components, False
    except ValueError:
        return fallback_outputs()


def decompose_multichannel_history(
    hist: np.ndarray,
    period: int,
    *,
    decomposition_mode: str = "stl",
    mstl_periods: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    trend_full = np.zeros_like(hist, dtype=np.float32)
    season_full = np.zeros_like(hist, dtype=np.float32)
    resid_full = np.zeros_like(hist, dtype=np.float32)
    fallback_count = 0

    for col_idx in range(hist.shape[1]):
        trend, season, resid, _season_components, did_fallback = decompose_history(
            hist[:, col_idx],
            period,
            decomposition_mode=decomposition_mode,
            mstl_periods=mstl_periods,
        )
        trend_full[:, col_idx] = trend
        season_full[:, col_idx] = season
        resid_full[:, col_idx] = resid
        fallback_count += int(did_fallback)

    return trend_full, season_full, resid_full, fallback_count


class ExchangeTargetDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        end_indices: np.ndarray,
        target_idx: int,
        lookback: int,
        horizon: int,
        stl_period: int,
        decomposition_mode: str = "stl",
        mstl_periods: tuple[int, ...] = (),
        compute_decomposition: bool = True,
    ) -> None:
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.target_idx = target_idx
        self.lookback = lookback
        self.horizon = horizon
        self.stl_period = stl_period
        self.decomposition_mode = decomposition_mode
        self.mstl_periods = normalize_periods(mstl_periods)
        self.compute_decomposition = compute_decomposition
        self.season_periods = self.mstl_periods if self.decomposition_mode == "mstl" and self.mstl_periods else (self.stl_period,)

        num_windows = len(self.end_indices)
        num_features = values.shape[1]
        exog_dim = num_features - 1
        num_season_components = len(self.season_periods)

        self.full_hist = np.zeros((num_windows, lookback, num_features), dtype=np.float32)
        self.exog_hist = np.zeros((num_windows, lookback, exog_dim), dtype=np.float32)
        self.exog_resid = np.zeros((num_windows, lookback, exog_dim), dtype=np.float32)
        self.trend_full = np.zeros((num_windows, lookback, num_features), dtype=np.float32)
        self.season_full = np.zeros((num_windows, lookback, num_features), dtype=np.float32)
        self.resid_full = np.zeros((num_windows, lookback, num_features), dtype=np.float32)
        self.trend = np.zeros((num_windows, lookback, 1), dtype=np.float32)
        self.season = np.zeros((num_windows, lookback, 1), dtype=np.float32)
        self.resid = np.zeros((num_windows, lookback, 1), dtype=np.float32)
        self.season_components = np.zeros((num_windows, lookback, num_season_components), dtype=np.float32)
        self.last_trend = np.zeros((num_windows, 1), dtype=np.float32)
        self.target = np.zeros((num_windows, horizon), dtype=np.float32)
        self.decomposition_fallback_events = 0
        self.decomposition_fallback_windows = 0

        self._precompute(values)

    def _precompute(self, values: np.ndarray) -> None:
        exog_idx = [i for i in range(values.shape[1]) if i != self.target_idx]
        for row, end_idx in enumerate(self.end_indices):
            start_idx = end_idx - self.lookback
            hist = values[start_idx:end_idx]
            self.full_hist[row] = hist
            self.exog_hist[row] = hist[:, exog_idx]
            self.target[row] = values[end_idx : end_idx + self.horizon, self.target_idx]

            if not self.compute_decomposition:
                self.last_trend[row, 0] = hist[-1, self.target_idx]
                continue

            trend_full, season_full, resid_full, fallback_count = decompose_multichannel_history(
                hist,
                self.stl_period,
                decomposition_mode=self.decomposition_mode,
                mstl_periods=self.mstl_periods,
            )
            target_trend, target_season, target_resid, target_components, target_fallback = decompose_history(
                hist[:, self.target_idx],
                self.stl_period,
                decomposition_mode=self.decomposition_mode,
                mstl_periods=self.mstl_periods,
            )

            self.exog_resid[row] = resid_full[:, exog_idx]
            self.trend_full[row] = trend_full
            self.season_full[row] = season_full
            self.resid_full[row] = resid_full
            self.trend[row, :, 0] = target_trend
            self.season[row, :, 0] = target_season
            self.resid[row, :, 0] = target_resid
            self.season_components[row] = target_components
            self.last_trend[row, 0] = target_trend[-1]
            self.decomposition_fallback_events += fallback_count + int(target_fallback)
            if fallback_count > 0 or target_fallback:
                self.decomposition_fallback_windows += 1

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_hist_full": torch.from_numpy(self.full_hist[idx]),
            "x_exog": torch.from_numpy(self.exog_hist[idx]),
            "x_exog_resid": torch.from_numpy(self.exog_resid[idx]),
            "trend_full": torch.from_numpy(self.trend_full[idx]),
            "season_full": torch.from_numpy(self.season_full[idx]),
            "resid_full": torch.from_numpy(self.resid_full[idx]),
            "trend": torch.from_numpy(self.trend[idx]),
            "season": torch.from_numpy(self.season[idx]),
            "season_components": torch.from_numpy(self.season_components[idx]),
            "resid": torch.from_numpy(self.resid[idx]),
            "last_trend": torch.from_numpy(self.last_trend[idx]),
            "y": torch.from_numpy(self.target[idx]),
        }


def build_train_loader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)


def build_eval_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def models_require_decomposition(models: tuple[str, ...]) -> bool:
    return "stl_tcn" in models


def make_datasets(
    values: np.ndarray,
    target_idx: int,
    config: ExperimentConfig,
    include_all_windows: bool = False,
) -> tuple[Dataset, Dataset, Dataset, Dataset | None, FeatureScaler, dict[str, int], np.ndarray]:
    dataset_name = infer_dataset_name(config)
    train_end, val_end = resolve_split_boundaries(
        dataset_name=dataset_name,
        total_len=len(values),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    train_idx, val_idx, test_idx, train_end, val_end = build_split_indices_from_boundaries(
        total_len=len(values),
        lookback=config.lookback,
        horizon=config.horizon,
        train_end=train_end,
        val_end=val_end,
    )

    scaler = FeatureScaler()
    scaler.fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)
    compute_decomposition = models_require_decomposition(config.models)

    train_ds = ExchangeTargetDataset(
        scaled,
        train_idx,
        target_idx=target_idx,
        lookback=config.lookback,
        horizon=config.horizon,
        stl_period=config.stl_period,
        decomposition_mode=resolve_decomposition_mode(config),
        mstl_periods=resolve_mstl_periods(config),
        compute_decomposition=compute_decomposition,
    )
    val_ds = ExchangeTargetDataset(
        scaled,
        val_idx,
        target_idx=target_idx,
        lookback=config.lookback,
        horizon=config.horizon,
        stl_period=config.stl_period,
        decomposition_mode=resolve_decomposition_mode(config),
        mstl_periods=resolve_mstl_periods(config),
        compute_decomposition=compute_decomposition,
    )
    test_ds = ExchangeTargetDataset(
        scaled,
        test_idx,
        target_idx=target_idx,
        lookback=config.lookback,
        horizon=config.horizon,
        stl_period=config.stl_period,
        decomposition_mode=resolve_decomposition_mode(config),
        mstl_periods=resolve_mstl_periods(config),
        compute_decomposition=compute_decomposition,
    )
    all_end_indices = np.arange(config.lookback, len(values) - config.horizon + 1)
    all_ds: Dataset | None = None
    if include_all_windows:
        all_ds = ExchangeTargetDataset(
            scaled,
            all_end_indices,
            target_idx=target_idx,
            lookback=config.lookback,
            horizon=config.horizon,
            stl_period=config.stl_period,
            decomposition_mode=resolve_decomposition_mode(config),
            mstl_periods=resolve_mstl_periods(config),
            compute_decomposition=compute_decomposition,
        )

    stats = {
        "train_rows": train_end,
        "val_rows": val_end - train_end,
        "test_rows": len(values) - val_end,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "all_windows": len(all_end_indices),
        "train_end": train_end,
        "val_end": val_end,
        "train_decomposition_fallback_events": train_ds.decomposition_fallback_events,
        "val_decomposition_fallback_events": val_ds.decomposition_fallback_events,
        "test_decomposition_fallback_events": test_ds.decomposition_fallback_events,
        "train_decomposition_fallback_windows": train_ds.decomposition_fallback_windows,
        "val_decomposition_fallback_windows": val_ds.decomposition_fallback_windows,
        "test_decomposition_fallback_windows": test_ds.decomposition_fallback_windows,
    }
    if all_ds is not None:
        stats["all_decomposition_fallback_events"] = all_ds.decomposition_fallback_events
        stats["all_decomposition_fallback_windows"] = all_ds.decomposition_fallback_windows

    return train_ds, val_ds, test_ds, all_ds, scaler, stats, all_end_indices


def make_dataloaders(
    values: np.ndarray,
    target_idx: int,
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader | None, FeatureScaler, dict[str, int], np.ndarray]:
    train_ds, val_ds, test_ds, all_ds, scaler, stats, all_end_indices = make_datasets(
        values,
        target_idx,
        config,
        include_all_windows=config.export_all_window_predictions,
    )
    train_loader = build_train_loader(train_ds, batch_size=config.batch_size, seed=config.seed)
    val_loader = build_eval_loader(val_ds, batch_size=config.batch_size)
    test_loader = build_eval_loader(test_ds, batch_size=config.batch_size)
    all_loader = None if all_ds is None else build_eval_loader(all_ds, batch_size=config.batch_size)
    return train_loader, val_loader, test_loader, all_loader, scaler, stats, all_end_indices


def build_model(
    model_name: str,
    input_dim: int,
    target_idx: int,
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    ours_common_kwargs = dict(
        input_dim=input_dim,
        target_idx=target_idx,
        horizon=config.horizon,
        latent_groups=config.ours_latent_groups,
        summary_dim=config.ours_summary_dim,
        depth=config.ours_depth,
        kernel_size=config.ours_kernel_size,
        dilations=config.ours_dilations,
        trend_basis_count=config.ours_trend_basis_count,
        seasonal_mode_count=config.ours_seasonal_mode_count,
        transient_basis_count=config.ours_transient_basis_count,
        use_router=config.ours_use_router,
        adaptive_bank=config.ours_adaptive_bank,
        use_trend_branch=config.ours_use_trend_branch,
        use_seasonal_branch=config.ours_use_seasonal_branch,
        use_transient_branch=config.ours_use_transient_branch,
        use_cluster_bank=config.ours_use_cluster_bank,
        num_clusters=config.ours_num_clusters,
        cluster_bank_fixed=config.ours_cluster_bank_fixed,
        use_local_correction=config.ours_use_local_correction,
        local_correction_hidden_dim=config.ours_local_correction_hidden_dim,
        use_group_frequency_offset=config.ours_use_group_frequency_offset,
        group_frequency_offset_scale=config.ours_group_frequency_offset_scale,
    )
    ours_variant_overrides: dict[str, dict[str, object]] = {
        "ours": {},
        "ours_no_router": {"use_router": False, "use_cluster_bank": False, "cluster_bank_fixed": False},
        "ours_fixed_bank": {"adaptive_bank": False, "use_cluster_bank": False, "cluster_bank_fixed": False},
        "ours_cluster_bank": {"use_cluster_bank": True, "cluster_bank_fixed": False},
        "ours_cluster_bank_fixed": {"use_cluster_bank": True, "cluster_bank_fixed": True},
        "ours_extended": {
            "use_cluster_bank": True,
            "cluster_bank_fixed": False,
            "use_local_correction": True,
            "use_group_frequency_offset": True,
        },
        "ours_no_transient": {"use_transient_branch": False, "use_cluster_bank": False, "cluster_bank_fixed": False},
        "ours_no_seasonal": {"use_seasonal_branch": False, "use_cluster_bank": False, "cluster_bank_fixed": False},
        "ours_trend_only": {
            "use_trend_branch": True,
            "use_seasonal_branch": False,
            "use_transient_branch": False,
            "use_cluster_bank": False,
            "cluster_bank_fixed": False,
        },
    }
    resolved_ours_kwargs = ours_common_kwargs | ours_variant_overrides.get(model_name, {})

    if model_name == "stl_tcn":
        return MultivariateSTLTCN(
            lookback=config.lookback,
            horizon=config.horizon,
            exog_dim=input_dim - 1,
            hidden_dim=config.stl_hidden_dim,
            use_pre_decomposed=True,
            tcn_dilations=(1, 2),
            use_trend_branch=config.stl_use_trend_branch,
            use_season_branch=config.stl_use_season_branch,
            use_resid_branch=config.stl_use_resid_branch,
        ).to(device)
    if model_name == "dlinear":
        return DLinearForecast(
            lookback=config.lookback,
            horizon=config.horizon,
            input_dim=input_dim,
            target_idx=target_idx,
            moving_avg_kernel=config.dlinear_moving_avg_kernel,
            individual=config.dlinear_individual,
        ).to(device)
    if model_name == "tcn":
        return TCNForecast(
            lookback=config.lookback,
            input_dim=input_dim,
            horizon=config.horizon,
            target_idx=target_idx,
            hidden_dim=config.tcn_hidden_dim,
            kernel_size=config.tcn_kernel_size,
            dropout=config.tcn_dropout,
        ).to(device)
    if model_name == "gru":
        return GRUForecast(
            input_dim=input_dim,
            horizon=config.horizon,
            hidden_dim=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            dropout=config.gru_dropout,
        ).to(device)
    if model_name == "lstm_proposed":
        return ThreeLayerLSTM(
            input_dim=input_dim,
            horizon=config.horizon,
            target_idx=target_idx,
            hidden_dim=config.lstm_proposed_hidden_dim,
        ).to(device)
    if model_name == "lstm_pure":
        return PureThreeLayerLSTM(
            input_dim=input_dim,
            horizon=config.horizon,
            hidden_dim=config.lstm_pure_hidden_dim,
        ).to(device)
    if model_name == "patchtst":
        return PatchTSTForecast(
            lookback=config.lookback,
            horizon=config.horizon,
            input_dim=input_dim,
            target_idx=target_idx,
            patch_len=config.patchtst_patch_len,
            patch_stride=config.patchtst_patch_stride,
            d_model=config.patchtst_d_model,
            num_layers=config.patchtst_num_layers,
            num_heads=config.patchtst_num_heads,
            ff_dim=config.patchtst_ff_dim,
            dropout=config.patchtst_dropout,
        ).to(device)
    if model_name in OURS_MODEL_NAMES - {"ours_direct_head"}:
        return OursForecast(**resolved_ours_kwargs).to(device)
    if model_name == "ours_direct_head":
        return OursDirectHeadForecast(
            input_dim=input_dim,
            target_idx=target_idx,
            horizon=config.horizon,
            latent_groups=config.ours_latent_groups,
            summary_dim=config.ours_summary_dim,
            depth=config.ours_depth,
            kernel_size=config.ours_kernel_size,
            dilations=config.ours_dilations,
        ).to(device)
    raise ValueError(f"Unknown model_name: {model_name}")


def build_models(
    input_dim: int,
    target_idx: int,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, nn.Module]:
    return {
        name: build_model(name, input_dim=input_dim, target_idx=target_idx, config=config, device=device)
        for name in config.models
    }


def model_forward(model_name: str, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, object]:
    if model_name == "stl_tcn":
        out = model(
            x_exog=batch["x_exog"],
            pre_trend=batch["trend"],
            pre_season=batch["season"],
            pre_resid=batch["resid"],
        )
        return out.squeeze(-1)
    if model_name in {"dlinear", "tcn", "gru", "lstm_proposed", "lstm_pure", "patchtst"}:
        return model(batch["x_hist_full"])
    if model_name in OURS_MODEL_NAMES:
        return model(batch["x_hist_full"])
    raise ValueError(f"Unknown model_name: {model_name}")


def extract_target_prediction(model_output: torch.Tensor | dict[str, object]) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    target_pred = model_output.get("target_pred")
    if not isinstance(target_pred, torch.Tensor):
        raise TypeError("Structured model output must include a tensor target_pred.")
    return target_pred


def compute_model_loss(
    model_name: str,
    model_output: torch.Tensor | dict[str, object],
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    config: ExperimentConfig,
) -> torch.Tensor:
    target_pred = extract_target_prediction(model_output)
    if model_name not in OURS_MODEL_NAMES:
        return loss_fn(target_pred, batch["y"])

    if not isinstance(model_output, dict):
        raise TypeError("Ours models must return structured outputs.")
    target_loss = loss_fn(target_pred, batch["y"])
    total_loss = target_loss

    coeff_sparsity = target_loss.new_zeros(())
    if config.ours_coeff_sparsity_weight > 0.0:
        seasonal_coeff = model_output.get("seasonal_coeff")
        transient_coeff = model_output.get("transient_coeff")
        sparse_terms = []
        if isinstance(seasonal_coeff, torch.Tensor):
            sparse_terms.append(seasonal_coeff.abs().mean())
        if isinstance(transient_coeff, torch.Tensor):
            sparse_terms.append(transient_coeff.abs().mean())
        if sparse_terms:
            coeff_sparsity = torch.stack(sparse_terms).sum()
            total_loss = total_loss + (config.ours_coeff_sparsity_weight * coeff_sparsity)

    seasonal_diversity = target_loss.new_zeros(())
    if config.ours_seasonal_diversity_weight > 0.0:
        if config.ours_seasonal_diversity_tau <= 0.0:
            raise ValueError("ours_seasonal_diversity_tau must be positive when ours_seasonal_diversity_weight > 0.")
        omega = model_output.get("omega")
        if isinstance(omega, torch.Tensor) and omega.ndim == 2 and omega.shape[-1] > 1:
            pairwise_distance = torch.abs(omega.unsqueeze(-1) - omega.unsqueeze(-2))
            pair_idx = torch.triu_indices(
                omega.shape[-1],
                omega.shape[-1],
                offset=1,
                device=omega.device,
            )
            upper_pairs = pairwise_distance[:, pair_idx[0], pair_idx[1]]
            if upper_pairs.numel() > 0:
                seasonal_diversity = torch.exp(-upper_pairs / config.ours_seasonal_diversity_tau).mean()
        if seasonal_diversity.numel() > 0:
            total_loss = total_loss + (config.ours_seasonal_diversity_weight * seasonal_diversity)

    router_entropy = target_loss.new_zeros(())
    router_weights = model_output.get("router_weights")
    router_enabled = bool(model_output.get("router_enabled", False))
    if config.ours_router_entropy_weight > 0.0 and router_enabled and isinstance(router_weights, torch.Tensor):
        safe_weights = router_weights.clamp_min(1e-8)
        router_entropy = -(safe_weights * safe_weights.log()).sum(dim=-1).mean()
        total_loss = total_loss + (config.ours_router_entropy_weight * router_entropy)

    return total_loss


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def collect_predictions(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred = extract_target_prediction(model_forward(model_name, model, batch))
            preds.append(pred.detach().cpu().numpy())
            trues.append(batch["y"].detach().cpu().numpy())
    if not preds:
        raise RuntimeError("No predictions were collected. Check dataset split sizes.")
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def collect_ours_diagnostics(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    window_end_indices: np.ndarray,
    split_label: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if model_name not in OURS_MODEL_NAMES:
        return None, None

    model.eval()
    branch_frames: list[pd.DataFrame] = []
    frequency_frames: list[pd.DataFrame] = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            batch_size = int(batch["y"].shape[0])
            batch_end_indices = window_end_indices[offset : offset + batch_size]
            offset += batch_size

            model_output = model_forward(model_name, model, batch)
            if not isinstance(model_output, dict):
                continue

            trend_coeff = model_output.get("trend_coeff")
            seasonal_coeff = model_output.get("seasonal_coeff")
            transient_coeff = model_output.get("transient_coeff")
            router_weights = model_output.get("router_weights")

            if (
                isinstance(trend_coeff, torch.Tensor)
                and isinstance(seasonal_coeff, torch.Tensor)
                and isinstance(transient_coeff, torch.Tensor)
            ):
                trend_abs = trend_coeff.abs().mean(dim=-1)
                seasonal_abs = seasonal_coeff.abs().mean(dim=-1)
                transient_abs = transient_coeff.abs().mean(dim=-1)
                if isinstance(router_weights, torch.Tensor):
                    branch_weights = router_weights
                else:
                    branch_mass = torch.stack([trend_abs, seasonal_abs, transient_abs], dim=-1)
                    branch_weights = branch_mass / branch_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                group_count = int(trend_abs.shape[1])
                branch_frames.append(
                    pd.DataFrame(
                        {
                            "model": np.full(batch_size * group_count, model_name, dtype=object),
                            "split": np.full(batch_size * group_count, split_label, dtype=object),
                            "window_end_index": np.repeat(batch_end_indices, group_count),
                            "group_index": np.tile(np.arange(group_count, dtype=np.int64), batch_size),
                            "trend_weight": branch_weights[..., 0].reshape(-1).detach().cpu().numpy(),
                            "seasonal_weight": branch_weights[..., 1].reshape(-1).detach().cpu().numpy(),
                            "transient_weight": branch_weights[..., 2].reshape(-1).detach().cpu().numpy(),
                            "trend_abs_coeff": trend_abs.reshape(-1).detach().cpu().numpy(),
                            "seasonal_abs_coeff": seasonal_abs.reshape(-1).detach().cpu().numpy(),
                            "transient_abs_coeff": transient_abs.reshape(-1).detach().cpu().numpy(),
                        }
                    )
                )

            omega = model_output.get("omega")
            effective_omega = model_output.get("effective_omega")
            delta_omega = model_output.get("delta_omega")
            if isinstance(omega, torch.Tensor) and isinstance(effective_omega, torch.Tensor):
                expanded_omega = omega.unsqueeze(1).expand_as(effective_omega)
                if not isinstance(delta_omega, torch.Tensor):
                    delta_omega = torch.zeros_like(effective_omega)

                group_count = int(effective_omega.shape[1])
                mode_count = int(effective_omega.shape[2])
                frequency_frames.append(
                    pd.DataFrame(
                        {
                            "model": np.full(batch_size * group_count * mode_count, model_name, dtype=object),
                            "split": np.full(batch_size * group_count * mode_count, split_label, dtype=object),
                            "window_end_index": np.repeat(np.repeat(batch_end_indices, group_count), mode_count),
                            "group_index": np.tile(np.repeat(np.arange(group_count, dtype=np.int64), mode_count), batch_size),
                            "mode_index": np.tile(np.arange(mode_count, dtype=np.int64), batch_size * group_count),
                            "omega": expanded_omega.reshape(-1).detach().cpu().numpy(),
                            "effective_omega": effective_omega.reshape(-1).detach().cpu().numpy(),
                            "delta_omega": delta_omega.reshape(-1).detach().cpu().numpy(),
                        }
                    )
                )

    if offset != len(window_end_indices):
        raise RuntimeError("Ours diagnostics collection did not consume all expected windows.")

    branch_frame = pd.concat(branch_frames, ignore_index=True) if branch_frames else None
    frequency_frame = pd.concat(frequency_frames, ignore_index=True) if frequency_frames else None
    return branch_frame, frequency_frame


def compute_average_loss(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    example_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            model_output = model_forward(model_name, model, batch)
            loss = compute_model_loss(model_name, model_output, batch, loss_fn, config)
            batch_size = batch["y"].shape[0]
            running_loss += loss.item() * batch_size
            example_count += batch_size
    if example_count == 0:
        raise RuntimeError("Loader is empty. Adjust lookback/horizon or split ratios.")
    return running_loss / example_count


def compute_metrics(
    pred_scaled: np.ndarray,
    target_scaled: np.ndarray,
    scaler: FeatureScaler,
    target_idx: int,
) -> dict[str, float]:
    mae_scaled = float(np.mean(np.abs(pred_scaled - target_scaled)))
    rmse_scaled = float(math.sqrt(np.mean((pred_scaled - target_scaled) ** 2)))
    pred = scaler.inverse_target(pred_scaled, target_idx)
    target = scaler.inverse_target(target_scaled, target_idx)
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(math.sqrt(np.mean((pred - target) ** 2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_scaled": mae_scaled,
        "rmse_scaled": rmse_scaled,
    }


def run_epoch(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    example_count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        model_output = model_forward(model_name, model, batch)
        loss = compute_model_loss(model_name, model_output, batch, loss_fn, config)
        loss.backward()
        optimizer.step()

        batch_size = batch["y"].shape[0]
        running_loss += loss.item() * batch_size
        example_count += batch_size
    if example_count == 0:
        raise RuntimeError("Training loader is empty. Adjust lookback/horizon or split ratios.")
    return running_loss / example_count


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def measure_inference_profile(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 1,
    measure_batches: int = 5,
) -> dict[str, float | str]:
    batches = list(loader)
    if not batches:
        return {
            "test_inference_ms_per_batch": float("nan"),
            "test_inference_ms_per_sample": float("nan"),
            "test_peak_memory_mb": float("nan"),
            "device_type": device.type,
        }

    warmup = min(warmup_batches, len(batches))
    measured = batches[: min(len(batches), max(measure_batches, 1))]
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    peak_rss = 0
    process = None
    if device.type != "cuda":
        try:
            import psutil

            process = psutil.Process(os.getpid())
            peak_rss = process.memory_info().rss
        except Exception:
            process = None

    with torch.no_grad():
        for batch in measured[:warmup]:
            batch = batch_to_device(batch, device)
            _ = model_forward(model_name, model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = 0.0
        sample_count = 0
        for batch in measured:
            batch = batch_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model_forward(model_name, model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            sample_count += int(batch["y"].shape[0])
            if process is not None:
                peak_rss = max(peak_rss, process.memory_info().rss)

    peak_memory_mb = float("nan")
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    elif process is not None:
        peak_memory_mb = float(peak_rss / (1024**2))

    batch_count = len(measured)
    return {
        "test_inference_ms_per_batch": float((elapsed / batch_count) * 1000.0),
        "test_inference_ms_per_sample": float((elapsed / max(sample_count, 1)) * 1000.0),
        "test_peak_memory_mb": peak_memory_mb,
        "device_type": device.type,
    }


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_mae = [row["val_mae"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="train_loss", color="#1f77b4")
    axes[0].plot(epochs, val_loss, label="val_loss", color="#d62728")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / Val Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_mae, label="val_mae", color="#2a9d8f")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_plot(
    pred_scaled: np.ndarray,
    target_scaled: np.ndarray,
    scaler: FeatureScaler,
    target_idx: int,
    output_path: Path,
    title: str,
) -> None:
    pred = scaler.inverse_target(pred_scaled[:1], target_idx)[0]
    target = scaler.inverse_target(target_scaled[:1], target_idx)[0]
    steps = np.arange(1, len(pred) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, target, marker="o", label="actual")
    ax.plot(steps, pred, marker="o", label="prediction")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Target value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_summary_plot(
    histories: dict[str, list[dict[str, float]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for model_name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label=model_name)
        axes[1].plot(epochs, [row["val_loss"] for row in history], marker="o", label=model_name)

    axes[0].set_title("Train Loss by Model")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Val Loss by Model")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_color_list(count: int, cmap_name: str) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap(cmap_name)
    if count <= 1:
        return [cmap(0.5)]
    return [cmap(idx / (count - 1)) for idx in range(count)]


def save_mae_mse_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels = metrics_df["model"].tolist()
    colors = build_color_list(len(labels), "magma")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, metrics_df["test_mae"], color=colors)
    axes[0].set_title("Test MAE")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, metrics_df["test_mse"], color=colors)
    axes[1].set_title("Test MSE")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_parameter_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels = metrics_df["model"].tolist()
    colors = build_color_list(len(labels), "viridis")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, metrics_df["params"], color=colors)
    ax.set_title("Parameter Count")
    ax.set_ylabel("Trainable params")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_epes_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels = metrics_df["model"].tolist()
    colors = build_color_list(len(labels), "cividis")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, metrics_df["epes"], color=colors)
    ax.set_title(f"EPES ({metrics_df['efficiency_error_metric'].iat[0].upper()}-based)")
    ax.set_ylabel("Score (higher is better)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_cpls_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels = metrics_df["model"].tolist()
    colors = build_color_list(len(labels), "plasma")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, metrics_df["cpls"], color=colors)
    ax.set_title("CPLS")
    ax.set_ylabel("Pairwise parameter-efficiency score")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_latency_memory_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels = metrics_df["model"].tolist()
    colors = build_color_list(len(labels), "inferno")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, metrics_df["test_inference_ms_per_sample"], color=colors)
    axes[0].set_title("Inference Latency")
    axes[0].set_ylabel("ms / sample")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, metrics_df["test_peak_memory_mb"], color=colors)
    axes[1].set_title("Peak Memory")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_pareto_plot(
    metrics_df: pd.DataFrame,
    x_column: str,
    output_path: Path,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y_column = "test_mae_scaled" if "test_mae_scaled" in metrics_df.columns else "test_mae"
    y_label = "Test scaled MAE" if y_column == "test_mae_scaled" else "Test MAE"
    if x_column == "params":
        x_values = np.log10(metrics_df[x_column].astype(float).to_numpy())
        resolved_x_label = f"log10({x_label})"
    else:
        x_values = metrics_df[x_column].astype(float).to_numpy()
        resolved_x_label = x_label
    pareto_mask = (
        metrics_df["pareto_optimal"].astype(bool).to_numpy()
        if "pareto_optimal" in metrics_df.columns
        else np.full(len(metrics_df), True, dtype=bool)
    )
    ours_mask = metrics_df["model"].isin(OURS_MODEL_NAMES)
    ax.scatter(
        x_values[~ours_mask.to_numpy()],
        metrics_df.loc[~ours_mask, y_column],
        c="#4c566a",
        label="baseline",
        s=55,
    )
    ax.scatter(
        x_values[ours_mask.to_numpy()],
        metrics_df.loc[ours_mask, y_column],
        c="#c8553d",
        label="Ours",
        s=65,
    )
    ax.scatter(
        x_values[pareto_mask],
        metrics_df.loc[pareto_mask, y_column],
        facecolors="none",
        edgecolors="#2a9d8f",
        linewidths=1.5,
        s=120,
        label="Pareto-optimal",
    )
    if "accuracy_tolerance_threshold" in metrics_df.columns:
        ax.axhline(
            float(metrics_df["accuracy_tolerance_threshold"].iat[0]),
            color="#6d597a",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label="2% tolerance",
        )
    for idx, row in metrics_df.reset_index(drop=True).iterrows():
        ax.annotate(
            str(row["model"]),
            (float(x_values[idx]), float(row[y_column])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel(resolved_x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Pareto: {y_label} vs {resolved_x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_ours_branch_usage_summary(branch_raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_name, group_df in branch_raw_df.groupby("model", dropna=False):
        for branch in ("trend", "seasonal", "transient"):
            rows.append(
                {
                    "model": str(model_name),
                    "branch": branch,
                    "mean_router_weight": float(group_df[f"{branch}_weight"].mean()),
                    "mean_abs_coeff": float(group_df[f"{branch}_abs_coeff"].mean()),
                    "window_count": int(group_df["window_end_index"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "branch"]).reset_index(drop=True)


def save_ours_branch_usage_plot(branch_summary_df: pd.DataFrame, output_path: Path) -> None:
    models = sorted(branch_summary_df["model"].unique().tolist())
    branches = ["trend", "seasonal", "transient"]
    colors = {"trend": "#355070", "seasonal": "#b56576", "transient": "#2a9d8f"}
    pivot_weight = branch_summary_df.pivot(index="model", columns="branch", values="mean_router_weight").reindex(models).fillna(0.0)
    pivot_coeff = branch_summary_df.pivot(index="model", columns="branch", values="mean_abs_coeff").reindex(models).fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bottom_weight = np.zeros(len(models), dtype=np.float64)
    bottom_coeff = np.zeros(len(models), dtype=np.float64)
    for branch in branches:
        axes[0].bar(models, pivot_weight.get(branch, pd.Series(0.0, index=models)), bottom=bottom_weight, color=colors[branch], label=branch)
        axes[1].bar(models, pivot_coeff.get(branch, pd.Series(0.0, index=models)), bottom=bottom_coeff, color=colors[branch], label=branch)
        bottom_weight += pivot_weight.get(branch, pd.Series(0.0, index=models)).to_numpy(dtype=np.float64)
        bottom_coeff += pivot_coeff.get(branch, pd.Series(0.0, index=models)).to_numpy(dtype=np.float64)

    axes[0].set_title("Branch Usage: Router Weight")
    axes[0].set_ylabel("Mean weight")
    axes[1].set_title("Branch Usage: Mean Abs Coefficient")
    axes[1].set_ylabel("Mean |coefficient|")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_ours_frequency_summary(frequency_raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frequency_raw_df.groupby(["model", "mode_index"], dropna=False)
        .agg(
            omega_mean=("omega", "mean"),
            omega_std=("omega", "std"),
            effective_omega_mean=("effective_omega", "mean"),
            effective_omega_std=("effective_omega", "std"),
            delta_omega_mean_abs=("delta_omega", lambda values: float(np.mean(np.abs(values)))),
            window_count=("window_end_index", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "mode_index"])
        .reset_index(drop=True)
    )
    return summary


def save_ours_frequency_plot(frequency_summary_df: pd.DataFrame, output_path: Path) -> None:
    models = sorted(frequency_summary_df["model"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for model_name in models:
        model_df = frequency_summary_df.loc[frequency_summary_df["model"] == model_name].sort_values("mode_index")
        mode_index = model_df["mode_index"].to_numpy(dtype=np.int64)
        axes[0].errorbar(
            mode_index,
            model_df["omega_mean"].to_numpy(dtype=np.float64),
            yerr=model_df["omega_std"].fillna(0.0).to_numpy(dtype=np.float64),
            marker="o",
            linestyle="--",
            label=f"{model_name} omega",
            alpha=0.7,
        )
        axes[0].errorbar(
            mode_index,
            model_df["effective_omega_mean"].to_numpy(dtype=np.float64),
            yerr=model_df["effective_omega_std"].fillna(0.0).to_numpy(dtype=np.float64),
            marker="o",
            linestyle="-",
            label=f"{model_name} effective",
        )
        axes[1].plot(
            mode_index,
            model_df["delta_omega_mean_abs"].to_numpy(dtype=np.float64),
            marker="o",
            linestyle="-",
            label=model_name,
        )

    axes[0].set_title("Learned Frequency")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Omega")
    axes[1].set_title("Group Frequency Offset")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Mean |delta_omega|")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_tables(metrics_df: pd.DataFrame, summary_dir: Path) -> None:
    metrics_df.loc[:, ["model", "test_mae", "test_mse"]].to_csv(summary_dir / "mae_mse_comparison.csv", index=False)
    if {"test_mae_scaled", "test_rmse_scaled"}.issubset(metrics_df.columns):
        metrics_df.loc[:, ["model", "test_mae_scaled", "test_rmse_scaled"]].to_csv(
            summary_dir / "mae_mse_comparison_scaled.csv",
            index=False,
        )
    metrics_df.loc[:, ["model", "params"]].to_csv(summary_dir / "parameter_comparison.csv", index=False)
    parameter_efficiency_columns = [
        column
        for column in [
            "model",
            "test_mae_scaled",
            "test_mae",
            "test_rmse",
            "params",
            "pareto_optimal",
            "relative_error_vs_best",
            "within_2pct_accuracy_tolerance",
            "parameter_efficient_2pct",
            "accuracy_tolerance_threshold",
        ]
        if column in metrics_df.columns
    ]
    if parameter_efficiency_columns:
        metrics_df.loc[:, parameter_efficiency_columns].to_csv(
            summary_dir / "parameter_efficiency_tolerance.csv",
            index=False,
        )
    metrics_df.loc[:, ["model", "epes", "efficiency_error_metric"]].to_csv(
        summary_dir / "epes_comparison.csv",
        index=False,
    )
    metrics_df.loc[:, ["model", "cpls", "cpls_wins", "cpls_losses", "cpls_ties", "efficiency_error_metric"]].to_csv(
        summary_dir / "cpls_comparison.csv",
        index=False,
    )
    metrics_df.loc[:, ["model", "test_inference_ms_per_batch", "test_inference_ms_per_sample", "test_peak_memory_mb", "device_type"]].to_csv(
        summary_dir / "latency_memory_comparison.csv",
        index=False,
    )


def build_metrics_dataframe(
    rows: list[dict[str, float | int | str]],
    efficiency_error_metric: str,
) -> pd.DataFrame:
    metrics_df = pd.DataFrame(rows)
    metrics_df["model"] = pd.Categorical(metrics_df["model"], categories=MODEL_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("model").reset_index(drop=True)
    metrics_df = add_legacy_efficiency_columns(metrics_df)
    metrics_df = add_epes_cpls_columns(metrics_df, efficiency_error_metric=efficiency_error_metric)
    accuracy_error_col = "test_mae_scaled" if "test_mae_scaled" in metrics_df.columns else "test_mae"
    return add_parameter_efficiency_columns(
        metrics_df,
        error_col=accuracy_error_col,
        params_col="params",
        tolerance=DEFAULT_ACCURACY_TOLERANCE,
    )


def split_labels(end_indices: np.ndarray, horizon: int, train_end: int, val_end: int) -> np.ndarray:
    labels = np.full(end_indices.shape, "test", dtype=object)
    forecast_end = end_indices + horizon - 1
    train_mask = forecast_end < train_end
    val_mask = (end_indices >= train_end) & (forecast_end < val_end)
    cross_split_mask = ~(train_mask | val_mask | (end_indices >= val_end))
    labels[train_mask] = "train"
    labels[val_mask] = "val"
    labels[cross_split_mask] = CROSS_SPLIT_LABEL
    return labels


def build_prediction_export_frame(
    predictions_by_model: dict[str, np.ndarray],
    actual_scaled: np.ndarray,
    scaler: FeatureScaler,
    target_idx: int,
    end_indices: np.ndarray,
    split: np.ndarray,
    inverse_transform: bool = True,
) -> pd.DataFrame:
    actual = scaler.inverse_target(actual_scaled, target_idx) if inverse_transform else actual_scaled

    frames: list[pd.DataFrame] = []
    for model_name, pred_scaled in predictions_by_model.items():
        pred = scaler.inverse_target(pred_scaled, target_idx) if inverse_transform else pred_scaled
        payload: dict[str, object] = {
            "model": np.full(len(end_indices), model_name, dtype=object),
            "split": split,
            "window_end_index": end_indices,
            "forecast_start_index": end_indices,
            "forecast_end_index": end_indices + pred.shape[1] - 1,
        }
        for step in range(pred.shape[1]):
            payload[f"actual_t+{step + 1}"] = actual[:, step]
            payload[f"pred_t+{step + 1}"] = pred[:, step]
        frames.append(pd.DataFrame(payload))

    return pd.concat(frames, ignore_index=True)


def save_summary_artifacts(
    summary_dir: Path,
    metrics_df: pd.DataFrame,
    histories: dict[str, list[dict[str, float]]],
    eval_predictions_by_model: dict[str, np.ndarray] | None,
    eval_actual_scaled: np.ndarray | None,
    scaler: FeatureScaler,
    target_idx: int,
    eval_end_indices: np.ndarray,
    split_stats: dict[str, int],
    ours_branch_raw_df: pd.DataFrame | None = None,
    ours_frequency_raw_df: pd.DataFrame | None = None,
    all_predictions_by_model: dict[str, np.ndarray] | None = None,
    all_actual_scaled: np.ndarray | None = None,
    all_end_indices: np.ndarray | None = None,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    pairwise_cpl_df = build_pairwise_cpl_dataframe(
        metrics_df,
        efficiency_error_metric=str(metrics_df["efficiency_error_metric"].iat[0]),
    )

    history_rows: list[dict[str, float | int | str]] = []
    for model_name, history in histories.items():
        for row in history:
            history_rows.append({"model": model_name, **row})

    pd.DataFrame(history_rows).to_csv(summary_dir / "training_history.csv", index=False)
    pairwise_cpl_df.to_csv(summary_dir / "pairwise_cpl.csv", index=False)
    write_summary_tables(metrics_df, summary_dir)

    save_loss_summary_plot(histories, summary_dir / "train_val_loss.png")
    save_mae_mse_plot(metrics_df, summary_dir / "mae_mse_comparison.png")
    save_parameter_plot(metrics_df, summary_dir / "parameter_comparison.png")
    save_epes_plot(metrics_df, summary_dir / "epes_comparison.png")
    save_cpls_plot(metrics_df, summary_dir / "cpls_comparison.png")
    save_latency_memory_plot(metrics_df, summary_dir / "latency_memory_comparison.png")
    save_pareto_plot(metrics_df, "params", summary_dir / "pareto_accuracy_params.png", "Params")
    save_pareto_plot(metrics_df, "test_inference_ms_per_sample", summary_dir / "pareto_accuracy_latency.png", "Latency (ms/sample)")
    if eval_predictions_by_model is not None and eval_actual_scaled is not None:
        eval_split = split_labels(
            eval_end_indices,
            horizon=int(metrics_df["horizon"].iat[0]),
            train_end=split_stats["train_end"],
            val_end=split_stats["val_end"],
        )
        eval_frame = build_prediction_export_frame(
            predictions_by_model=eval_predictions_by_model,
            actual_scaled=eval_actual_scaled,
            scaler=scaler,
            target_idx=target_idx,
            end_indices=eval_end_indices,
            split=eval_split,
            inverse_transform=True,
        )
        eval_frame = eval_frame.loc[eval_frame["split"].isin(["val", "test"])].reset_index(drop=True)
        eval_frame.to_csv(summary_dir / "eval_model_predictions.csv", index=False)
        eval_frame_scaled = build_prediction_export_frame(
            predictions_by_model=eval_predictions_by_model,
            actual_scaled=eval_actual_scaled,
            scaler=scaler,
            target_idx=target_idx,
            end_indices=eval_end_indices,
            split=eval_split,
            inverse_transform=False,
        )
        eval_frame_scaled = eval_frame_scaled.loc[eval_frame_scaled["split"].isin(["val", "test"])].reset_index(drop=True)
        eval_frame_scaled.to_csv(summary_dir / "eval_model_predictions_scaled.csv", index=False)

    if ours_branch_raw_df is not None and not ours_branch_raw_df.empty:
        branch_summary_df = build_ours_branch_usage_summary(ours_branch_raw_df)
        ours_branch_raw_df.to_csv(summary_dir / "ours_branch_usage_raw.csv", index=False)
        branch_summary_df.to_csv(summary_dir / "ours_branch_usage_summary.csv", index=False)
        save_ours_branch_usage_plot(branch_summary_df, summary_dir / "ours_branch_usage.png")

    if ours_frequency_raw_df is not None and not ours_frequency_raw_df.empty:
        frequency_summary_df = build_ours_frequency_summary(ours_frequency_raw_df)
        ours_frequency_raw_df.to_csv(summary_dir / "ours_frequency_raw.csv", index=False)
        frequency_summary_df.to_csv(summary_dir / "ours_frequency_summary.csv", index=False)
        save_ours_frequency_plot(frequency_summary_df, summary_dir / "ours_frequency.png")

    if (
        all_predictions_by_model is not None
        and all_actual_scaled is not None
        and all_end_indices is not None
    ):
        all_split = split_labels(
            all_end_indices,
            horizon=int(metrics_df["horizon"].iat[0]),
            train_end=split_stats["train_end"],
            val_end=split_stats["val_end"],
        )
        all_frame = build_prediction_export_frame(
            predictions_by_model=all_predictions_by_model,
            actual_scaled=all_actual_scaled,
            scaler=scaler,
            target_idx=target_idx,
            end_indices=all_end_indices,
            split=all_split,
            inverse_transform=True,
        )
        all_frame.to_csv(summary_dir / "all_window_predictions.csv", index=False)
        all_frame_scaled = build_prediction_export_frame(
            predictions_by_model=all_predictions_by_model,
            actual_scaled=all_actual_scaled,
            scaler=scaler,
            target_idx=target_idx,
            end_indices=all_end_indices,
            split=all_split,
            inverse_transform=False,
        )
        all_frame_scaled.to_csv(summary_dir / "all_window_predictions_scaled.csv", index=False)


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    optimizer_name = str(config.optimizer_name).strip().lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name}")


def select_loss_fn(model_name: str, config: ExperimentConfig) -> nn.Module:
    loss_name = str(config.loss_name).strip().lower()
    if loss_name == "auto":
        return nn.MSELoss() if model_name in OURS_MODEL_NAMES else nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def tslib_type1_lr(base_lr: float, epoch: int) -> float:
    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")
    return base_lr * (0.5 ** (epoch - 1))


def apply_lr_schedule(optimizer: torch.optim.Optimizer, config: ExperimentConfig, epoch: int) -> float:
    schedule_name = str(config.lr_schedule).strip().lower()
    if schedule_name == "none":
        return float(optimizer.param_groups[0]["lr"])
    if schedule_name == "tslib_type1":
        lr = tslib_type1_lr(config.lr, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    raise ValueError(f"Unsupported lr_schedule: {config.lr_schedule}")


def resolve_monitored_metric_name(config: ExperimentConfig) -> str:
    early_stopping_metric = str(config.early_stopping_metric).strip().lower()
    if early_stopping_metric == "selection_metric":
        return validate_efficiency_error_metric(config.efficiency_error_metric)
    if early_stopping_metric == "val_loss":
        return "val_loss"
    raise ValueError(f"Unsupported early_stopping_metric: {config.early_stopping_metric}")


def monitored_metric_value(monitored_metric_name: str, val_loss: float, val_metrics: dict[str, float]) -> float:
    if monitored_metric_name == "val_loss":
        return val_loss
    return val_metrics[monitored_metric_name]


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: FeatureScaler,
    target_idx: int,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float], list[dict[str, float]]]:
    optimizer = build_optimizer(model, config)
    loss_fn = select_loss_fn(model_name, config)
    best_state = copy.deepcopy(model.state_dict())
    monitored_metric_name = resolve_monitored_metric_name(config)
    best_val_metric = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model_name, model, train_loader, optimizer, loss_fn, config, device)
        val_loss = compute_average_loss(model_name, model, val_loader, loss_fn, config, device)
        val_pred, val_true = collect_predictions(model_name, model, val_loader, device)
        val_metrics = compute_metrics(val_pred, val_true, scaler, target_idx)
        val_selection_metric = monitored_metric_value(monitored_metric_name, val_loss, val_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "selection_metric_name": monitored_metric_name,
                "val_selection_metric": val_selection_metric,
            }
        )

        if val_selection_metric < best_val_metric:
            best_val_metric = val_selection_metric
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                break
        apply_lr_schedule(optimizer, config, epoch)

    model.load_state_dict(best_state)
    val_pred, val_true = collect_predictions(model_name, model, val_loader, device)
    best_val_metrics = compute_metrics(val_pred, val_true, scaler, target_idx)
    return model, best_val_metrics, history


def prepare_run_dir(base_dir: Path, config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = build_run_dir(
        base_dir=base_dir,
        lookback=config.lookback,
        horizon=config.horizon,
        seed=config.seed,
        timestamp=timestamp,
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    (run_dir / "summary").mkdir(parents=True, exist_ok=False)
    return run_dir


def run_experiment(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    configure_torch_determinism(config.deterministic)
    dataset_name = infer_dataset_name(config)

    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    values_df, time_order_info = load_time_ordered_frame(data_path)
    if config.target_col not in values_df.columns:
        raise ValueError(f"{config.target_col} is not a valid column. Available: {list(values_df.columns)}")

    values = values_df.to_numpy(dtype=np.float32)
    target_idx = int(values_df.columns.get_loc(config.target_col))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, all_ds, scaler, split_stats, all_end_indices = make_datasets(
        values,
        target_idx,
        config,
        include_all_windows=config.export_all_window_predictions,
    )
    val_loader = build_eval_loader(val_ds, batch_size=config.batch_size)
    test_loader = build_eval_loader(test_ds, batch_size=config.batch_size)
    all_loader = None if all_ds is None else build_eval_loader(all_ds, batch_size=config.batch_size)
    run_dir = prepare_run_dir(Path(config.results_dir), config)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        payload = asdict(config)
        payload["dataset_name"] = dataset_name
        payload["device"] = str(device)
        payload["split_stats"] = split_stats
        payload["time_order_info"] = time_order_info
        payload["metric_storage_policy"] = "scaled_and_inverse"
        payload["eval_prediction_storage_policy"] = (
            "scaled_and_inverse" if config.export_eval_predictions else "disabled"
        )
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    rows: list[dict[str, float | int | str]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    eval_predictions_by_model: dict[str, np.ndarray] | None = (
        {} if config.export_eval_predictions else None
    )
    eval_true_scaled: np.ndarray | None = None
    ours_branch_frames: list[pd.DataFrame] = []
    ours_frequency_frames: list[pd.DataFrame] = []
    all_predictions_by_model: dict[str, np.ndarray] | None = (
        {} if config.export_all_window_predictions else None
    )
    all_true_scaled: np.ndarray | None = None
    eval_end_indices = np.concatenate([val_ds.end_indices, test_ds.end_indices])

    for model_name in config.models:
        model_seed = seed_for_model(config.seed, model_name)
        set_seed(model_seed)
        train_loader = build_train_loader(train_ds, batch_size=config.batch_size, seed=model_seed)
        model = build_model(model_name, input_dim=values.shape[1], target_idx=target_idx, config=config, device=device)
        trained_model, val_metrics, history = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=scaler,
            target_idx=target_idx,
            config=config,
            device=device,
        )
        val_pred, val_true = collect_predictions(model_name, trained_model, val_loader, device)
        test_pred, test_true = collect_predictions(model_name, trained_model, test_loader, device)
        test_metrics = compute_metrics(test_pred, test_true, scaler, target_idx)
        if config.export_ours_diagnostics:
            val_branch_df, val_frequency_df = collect_ours_diagnostics(
                model_name=model_name,
                model=trained_model,
                loader=val_loader,
                device=device,
                window_end_indices=val_ds.end_indices,
                split_label="val",
            )
            test_branch_df, test_frequency_df = collect_ours_diagnostics(
                model_name=model_name,
                model=trained_model,
                loader=test_loader,
                device=device,
                window_end_indices=test_ds.end_indices,
                split_label="test",
            )
            if val_branch_df is not None:
                ours_branch_frames.append(val_branch_df)
            if test_branch_df is not None:
                ours_branch_frames.append(test_branch_df)
            if val_frequency_df is not None:
                ours_frequency_frames.append(val_frequency_df)
            if test_frequency_df is not None:
                ours_frequency_frames.append(test_frequency_df)
        inference_profile = measure_inference_profile(
            model_name=model_name,
            model=trained_model,
            loader=test_loader,
            device=device,
        )
        if eval_predictions_by_model is not None:
            eval_pred = np.concatenate([val_pred, test_pred], axis=0)
            eval_true = np.concatenate([val_true, test_true], axis=0)
            if eval_true_scaled is None:
                eval_true_scaled = eval_true
            elif not np.allclose(eval_true_scaled, eval_true):
                raise RuntimeError("Collected evaluation targets differ across timing paths.")
            eval_predictions_by_model[model_name] = eval_pred

        if all_loader is not None and all_predictions_by_model is not None:
            all_pred, all_true = collect_predictions(
                model_name=model_name,
                model=trained_model,
                loader=all_loader,
                device=device,
            )
            all_predictions_by_model[model_name] = all_pred
            if all_true_scaled is None:
                all_true_scaled = all_true
            elif not np.allclose(all_true_scaled, all_true):
                raise RuntimeError("Collected targets differ across timing paths.")

        torch.save(trained_model.state_dict(), run_dir / "checkpoints" / f"{model_name}.pt")
        pd.DataFrame(history).to_csv(run_dir / f"{model_name}_history.csv", index=False)
        save_training_curve(history, run_dir / "plots" / f"{model_name}_train_curve.png")
        save_prediction_plot(
            pred_scaled=test_pred,
            target_scaled=test_true,
            scaler=scaler,
            target_idx=target_idx,
            output_path=run_dir / "plots" / f"{model_name}_forecast.png",
            title=f"{model_name} test forecast ({config.target_col})",
        )

        histories[model_name] = history

        rows.append(
            {
                "model": model_name,
                "dataset_name": dataset_name,
                "target_col": config.target_col,
                "lookback": config.lookback,
                "horizon": config.horizon,
                "seed": config.seed,
                "params": count_parameters(trained_model),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mae_scaled": val_metrics["mae_scaled"],
                "val_rmse_scaled": val_metrics["rmse_scaled"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mae_scaled": test_metrics["mae_scaled"],
                "test_rmse_scaled": test_metrics["rmse_scaled"],
                "deterministic": config.deterministic,
                "test_inference_ms_per_batch": inference_profile["test_inference_ms_per_batch"],
                "test_inference_ms_per_sample": inference_profile["test_inference_ms_per_sample"],
                "test_peak_memory_mb": inference_profile["test_peak_memory_mb"],
                "device_type": inference_profile["device_type"],
                "time_order_source": str(time_order_info["time_order_source"]),
                "time_order_column": (
                    "" if time_order_info["time_order_column"] is None else str(time_order_info["time_order_column"])
                ),
                "time_order_was_sorted": bool(time_order_info["time_order_was_sorted"]),
            }
        )

    if eval_predictions_by_model is not None and eval_true_scaled is None:
        raise RuntimeError("No evaluation predictions were collected.")

    metrics_df = build_metrics_dataframe(rows, efficiency_error_metric=config.efficiency_error_metric)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    save_summary_artifacts(
        summary_dir=run_dir / "summary",
        metrics_df=metrics_df,
        histories=histories,
        eval_predictions_by_model=eval_predictions_by_model,
        eval_actual_scaled=eval_true_scaled,
        scaler=scaler,
        target_idx=target_idx,
        eval_end_indices=eval_end_indices,
        split_stats=split_stats,
        ours_branch_raw_df=(pd.concat(ours_branch_frames, ignore_index=True) if ours_branch_frames else None),
        ours_frequency_raw_df=(pd.concat(ours_frequency_frames, ignore_index=True) if ours_frequency_frames else None),
        all_predictions_by_model=all_predictions_by_model,
        all_actual_scaled=all_true_scaled,
        all_end_indices=all_end_indices if config.export_all_window_predictions else None,
    )
    print(f"Saved run artifacts to: {run_dir}")
    print(f"Saved run summary to: {run_dir / 'summary'}")
    print(metrics_df.to_string(index=False))
    return run_dir


def parse_models(raw: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = sorted(set(models) - set(MODEL_ORDER))
    if invalid:
        raise ValueError(f"Invalid model names: {invalid}. Available: {MODEL_ORDER}")
    if not models:
        raise ValueError("At least one model must be selected.")
    return models


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_data = script_dir.parent / "dataset" / "ETTh1.csv"
    default_results = project_root / "runs" / "exchange_single_run"

    parser = argparse.ArgumentParser(description="Train Exchange models once and compare metrics.")
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--target-col", type=str, default="OT")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--lookback", type=int, default=56)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--stl-period", type=int, default=5)
    parser.add_argument("--decomposition-mode", type=str, choices=("stl", "mstl"), default="stl")
    parser.add_argument("--mstl-periods", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer-name", type=str, choices=OPTIMIZER_NAMES, default="adamw")
    parser.add_argument("--loss-name", type=str, choices=LOSS_NAMES, default="auto")
    parser.add_argument("--lr-schedule", type=str, choices=LR_SCHEDULES, default="none")
    parser.add_argument("--early-stopping-metric", type=str, choices=EARLY_STOPPING_METRICS, default="selection_metric")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--efficiency-error-metric", type=str, choices=EFFICIENCY_ERROR_METRICS, default="mae")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--export-all-window-predictions", action="store_true")
    parser.add_argument("--no-export-eval-predictions", dest="export_eval_predictions", action="store_false")
    parser.set_defaults(export_eval_predictions=True)
    parser.add_argument("--no-export-ours-diagnostics", dest="export_ours_diagnostics", action="store_false")
    parser.set_defaults(export_ours_diagnostics=True)
    parser.add_argument("--lstm-pure-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-proposed-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-num-layers", type=int, default=2)
    parser.add_argument("--gru-dropout", type=float, default=0.1)
    parser.add_argument("--stl-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--tcn-hidden-dim",
        type=int,
        default=32,
        help="Official OnlineTSF TCN wrapper only supports the default value 32.",
    )
    parser.add_argument(
        "--tcn-kernel-size",
        type=int,
        default=3,
        help="Official OnlineTSF TCN wrapper only supports the default value 3.",
    )
    parser.add_argument(
        "--tcn-dropout",
        type=float,
        default=0.1,
        help="Official OnlineTSF TCN wrapper only supports the default value 0.1.",
    )
    parser.add_argument(
        "--patchtst-patch-len",
        type=int,
        default=OFFICIAL_PATCHTST_PATCH_LEN,
        help="Official PatchTST supervised default is 16; dataset-specific scripts may override this.",
    )
    parser.add_argument(
        "--patchtst-patch-stride",
        type=int,
        default=OFFICIAL_PATCHTST_PATCH_STRIDE,
        help="Official PatchTST supervised default is 8; dataset-specific scripts may override this.",
    )
    parser.add_argument(
        "--patchtst-d-model",
        type=int,
        default=OFFICIAL_PATCHTST_D_MODEL,
        help="Official PatchTST supervised default is 512; dataset-specific scripts may override this.",
    )
    parser.add_argument(
        "--patchtst-num-layers",
        type=int,
        default=OFFICIAL_PATCHTST_NUM_LAYERS,
        help="Official PatchTST supervised default is 2; dataset-specific scripts may override this.",
    )
    parser.add_argument(
        "--patchtst-num-heads",
        type=int,
        default=OFFICIAL_PATCHTST_NUM_HEADS,
        help="Official PatchTST supervised default is 8; dataset-specific scripts may override this.",
    )
    parser.add_argument(
        "--patchtst-ff-dim",
        type=int,
        default=OFFICIAL_PATCHTST_FF_DIM,
        help="Official PatchTST supervised default is 2048; dataset-specific scripts may override this.",
    )
    parser.add_argument(
        "--patchtst-dropout",
        type=float,
        default=OFFICIAL_PATCHTST_DROPOUT,
        help="Official PatchTST supervised default is 0.1; dataset-specific scripts may override this.",
    )
    parser.add_argument("--dlinear-moving-avg-kernel", type=int, default=25)
    parser.add_argument("--dlinear-individual", action="store_true")
    parser.add_argument("--stl-disable-trend-branch", action="store_true")
    parser.add_argument("--stl-disable-season-branch", action="store_true")
    parser.add_argument("--stl-disable-resid-branch", action="store_true")
    parser.add_argument("--ours-latent-groups", type=int, default=16)
    parser.add_argument("--ours-summary-dim", type=int, default=32)
    parser.add_argument("--ours-depth", type=int, default=3)
    parser.add_argument("--ours-kernel-size", type=int, default=3)
    parser.add_argument("--ours-dilations", type=str, default="1,2,4")
    parser.add_argument("--ours-trend-basis-count", type=int, default=4)
    parser.add_argument("--ours-seasonal-mode-count", type=int, default=4)
    parser.add_argument("--ours-transient-basis-count", type=int, default=2)
    parser.add_argument("--ours-disable-router", action="store_true")
    parser.add_argument("--ours-use-fixed-bank", action="store_true")
    parser.add_argument("--ours-disable-trend-branch", action="store_true")
    parser.add_argument("--ours-disable-seasonal-branch", action="store_true")
    parser.add_argument("--ours-disable-transient-branch", action="store_true")
    parser.add_argument("--ours-coeff-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--ours-seasonal-diversity-weight", type=float, default=0.0)
    parser.add_argument("--ours-seasonal-diversity-tau", type=float, default=0.25)
    parser.add_argument("--ours-router-entropy-weight", type=float, default=0.0)
    parser.add_argument("--ours-num-clusters", type=int, default=3)
    parser.add_argument("--ours-use-cluster-bank", action="store_true")
    parser.add_argument("--ours-cluster-bank-fixed", action="store_true")
    parser.add_argument("--ours-use-local-correction", action="store_true")
    parser.add_argument("--ours-local-correction-hidden-dim", type=int, default=16)
    parser.add_argument("--ours-use-group-frequency-offset", action="store_true")
    parser.add_argument("--ours-group-frequency-offset-scale", type=float, default=0.10)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mstl_periods = tuple(int(part.strip()) for part in args.mstl_periods.split(",") if part.strip())
    ours_dilations = tuple(int(part.strip()) for part in args.ours_dilations.split(",") if part.strip())
    config = ExperimentConfig(
        data_path=str(args.data_path),
        results_dir=str(args.results_dir),
        target_col=args.target_col,
        dataset_name=args.dataset_name,
        lookback=args.lookback,
        horizon=args.horizon,
        stl_period=args.stl_period,
        decomposition_mode=args.decomposition_mode,
        mstl_periods=mstl_periods,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer_name,
        loss_name=args.loss_name,
        lr_schedule=args.lr_schedule,
        early_stopping_metric=args.early_stopping_metric,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        efficiency_error_metric=args.efficiency_error_metric,
        deterministic=args.deterministic,
        export_all_window_predictions=args.export_all_window_predictions,
        export_eval_predictions=args.export_eval_predictions,
        export_ours_diagnostics=args.export_ours_diagnostics,
        lstm_pure_hidden_dim=args.lstm_pure_hidden_dim,
        lstm_proposed_hidden_dim=args.lstm_proposed_hidden_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        gru_dropout=args.gru_dropout,
        stl_hidden_dim=args.stl_hidden_dim,
        tcn_hidden_dim=args.tcn_hidden_dim,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
        patchtst_patch_len=args.patchtst_patch_len,
        patchtst_patch_stride=args.patchtst_patch_stride,
        patchtst_d_model=args.patchtst_d_model,
        patchtst_num_layers=args.patchtst_num_layers,
        patchtst_num_heads=args.patchtst_num_heads,
        patchtst_ff_dim=args.patchtst_ff_dim,
        patchtst_dropout=args.patchtst_dropout,
        dlinear_moving_avg_kernel=args.dlinear_moving_avg_kernel,
        dlinear_individual=args.dlinear_individual,
        stl_use_trend_branch=not args.stl_disable_trend_branch,
        stl_use_season_branch=not args.stl_disable_season_branch,
        stl_use_resid_branch=not args.stl_disable_resid_branch,
        ours_latent_groups=args.ours_latent_groups,
        ours_summary_dim=args.ours_summary_dim,
        ours_depth=args.ours_depth,
        ours_kernel_size=args.ours_kernel_size,
        ours_dilations=ours_dilations,
        ours_trend_basis_count=args.ours_trend_basis_count,
        ours_seasonal_mode_count=args.ours_seasonal_mode_count,
        ours_transient_basis_count=args.ours_transient_basis_count,
        ours_use_router=not args.ours_disable_router,
        ours_adaptive_bank=not args.ours_use_fixed_bank,
        ours_use_trend_branch=not args.ours_disable_trend_branch,
        ours_use_seasonal_branch=not args.ours_disable_seasonal_branch,
        ours_use_transient_branch=not args.ours_disable_transient_branch,
        ours_coeff_sparsity_weight=args.ours_coeff_sparsity_weight,
        ours_seasonal_diversity_weight=args.ours_seasonal_diversity_weight,
        ours_seasonal_diversity_tau=args.ours_seasonal_diversity_tau,
        ours_router_entropy_weight=args.ours_router_entropy_weight,
        ours_num_clusters=args.ours_num_clusters,
        ours_use_cluster_bank=args.ours_use_cluster_bank,
        ours_cluster_bank_fixed=args.ours_cluster_bank_fixed,
        ours_use_local_correction=args.ours_use_local_correction,
        ours_local_correction_hidden_dim=args.ours_local_correction_hidden_dim,
        ours_use_group_frequency_offset=args.ours_use_group_frequency_offset,
        ours_group_frequency_offset_scale=args.ours_group_frequency_offset_scale,
        models=parse_models(args.models),
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
