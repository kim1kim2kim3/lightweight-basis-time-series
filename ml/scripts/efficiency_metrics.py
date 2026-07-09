from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

EFFICIENCY_ERROR_METRICS = ("mae", "rmse")
DEFAULT_ACCURACY_TOLERANCE = 0.02
PAIRWISE_CPL_COLUMNS = [
    "model_a",
    "model_b",
    "smaller_model",
    "larger_model",
    "error_metric",
    "error_smaller",
    "error_larger",
    "params_smaller",
    "params_larger",
    "state",
    "cpl",
    "winner",
]


def format_tolerance_label(tolerance: float) -> str:
    pct = float(tolerance) * 100.0
    if math.isclose(pct, round(pct), rel_tol=1e-12, abs_tol=1e-12):
        return f"{int(round(pct))}pct"
    return f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


def validate_efficiency_error_metric(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized not in EFFICIENCY_ERROR_METRICS:
        raise ValueError(
            f"Invalid efficiency_error_metric: {metric}. "
            f"Expected one of {EFFICIENCY_ERROR_METRICS}."
        )
    return normalized


def resolve_error_column(metric: str, prefix: str = "test") -> str:
    normalized = validate_efficiency_error_metric(metric)
    return f"{prefix}_{normalized}"


def add_legacy_efficiency_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    if "test_mse" not in enriched.columns:
        enriched["test_mse"] = enriched["test_rmse"] ** 2
    if "val_mse" not in enriched.columns:
        enriched["val_mse"] = enriched["val_rmse"] ** 2
    if "mae_param_product" not in enriched.columns:
        enriched["mae_param_product"] = enriched["params"] * enriched["test_mae"]
    if "mse_param_product" not in enriched.columns:
        enriched["mse_param_product"] = enriched["params"] * enriched["test_mse"]
    if "mae_efficiency_vs_best" not in enriched.columns:
        best_mae_row = select_best_error_row(enriched, "mae")
        enriched["mae_efficiency_vs_best"] = (
            (best_mae_row["params"] * best_mae_row["test_mae"]) / enriched["mae_param_product"]
        )
    if "mse_efficiency_vs_best" not in enriched.columns:
        best_mse_row = select_best_error_row(enriched, "rmse")
        enriched["mse_efficiency_vs_best"] = (
            (best_mse_row["params"] * best_mse_row["test_mse"]) / enriched["mse_param_product"]
        )
    return enriched


def select_best_error_row(df: pd.DataFrame, metric: str) -> pd.Series:
    error_col = resolve_error_column(metric)
    ordered = sort_for_best_error(df, [error_col, "params", "model"] if "model" in df.columns else [error_col, "params"])
    return ordered.iloc[0]


def sort_for_best_error(df: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    available = [column for column in by if column in df.columns]
    return df.sort_values(available, kind="mergesort").reset_index(drop=True)


def _validate_error_and_params_columns(df: pd.DataFrame, error_col: str, params_col: str) -> None:
    if error_col not in df.columns:
        raise KeyError(f"Missing required column: {error_col}")
    if params_col not in df.columns:
        raise KeyError(f"Missing required column: {params_col}")
    if df[error_col].isna().any():
        raise ValueError(f"{error_col} must not contain missing values.")
    if df[params_col].isna().any():
        raise ValueError(f"{params_col} must not contain missing values.")
    if (df[error_col].astype(float) <= 0).any():
        raise ValueError(f"{error_col} must be positive.")
    if (df[params_col].astype(float) <= 0).any():
        raise ValueError(f"{params_col} must be positive.")


def build_pareto_optimal_mask(
    df: pd.DataFrame,
    error_col: str,
    params_col: str = "params",
) -> pd.Series:
    """Return True for rows not dominated on both error and parameter count."""
    _validate_error_and_params_columns(df, error_col, params_col)
    errors = df[error_col].astype(float).to_numpy()
    params = df[params_col].astype(float).to_numpy()
    mask = np.ones(len(df), dtype=bool)
    for idx, (error_value, param_value) in enumerate(zip(errors, params)):
        dominates = (
            (errors <= error_value)
            & (params <= param_value)
            & ((errors < error_value) | (params < param_value))
        )
        mask[idx] = not bool(np.any(dominates))
    return pd.Series(mask, index=df.index, name="pareto_optimal")


def select_tolerance_efficient_row(
    df: pd.DataFrame,
    error_col: str,
    params_col: str = "params",
    tolerance: float = DEFAULT_ACCURACY_TOLERANCE,
) -> pd.Series:
    """Select the smallest model within an error tolerance of the best model."""
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
    _validate_error_and_params_columns(df, error_col, params_col)
    best_error = float(df[error_col].min())
    threshold = best_error * (1.0 + float(tolerance))
    candidates = df.loc[df[error_col].astype(float) <= threshold + 1e-12].copy()
    sort_columns = [params_col, error_col]
    if "model" in candidates.columns:
        sort_columns.append("model")
    return candidates.sort_values(sort_columns, kind="mergesort").iloc[0]


def add_parameter_efficiency_columns(
    df: pd.DataFrame,
    error_col: str,
    params_col: str = "params",
    tolerance: float = DEFAULT_ACCURACY_TOLERANCE,
) -> pd.DataFrame:
    """Add Pareto and tolerance-based parameter-efficiency annotations."""
    _validate_error_and_params_columns(df, error_col, params_col)
    label = format_tolerance_label(tolerance)
    enriched = df.copy()
    best_error = float(enriched[error_col].min())
    threshold = best_error * (1.0 + float(tolerance))
    efficient_row = select_tolerance_efficient_row(
        enriched,
        error_col=error_col,
        params_col=params_col,
        tolerance=tolerance,
    )
    if "model" in enriched.columns:
        efficient_model = str(efficient_row["model"])
        enriched[f"parameter_efficient_{label}"] = enriched["model"].astype(str) == efficient_model
    else:
        enriched[f"parameter_efficient_{label}"] = enriched.index == efficient_row.name
    enriched["pareto_optimal"] = build_pareto_optimal_mask(enriched, error_col, params_col).astype(bool)
    enriched["accuracy_efficiency_error_column"] = error_col
    enriched["accuracy_tolerance"] = float(tolerance)
    enriched["best_accuracy_error"] = best_error
    enriched["accuracy_tolerance_threshold"] = threshold
    enriched["relative_error_vs_best"] = (enriched[error_col].astype(float) / best_error) - 1.0
    enriched[f"within_{label}_accuracy_tolerance"] = (
        enriched[error_col].astype(float) <= threshold + 1e-12
    )
    return enriched


def build_pairwise_cpl_dataframe(df: pd.DataFrame, efficiency_error_metric: str = "mae") -> pd.DataFrame:
    metric = validate_efficiency_error_metric(efficiency_error_metric)
    error_col = resolve_error_column(metric)
    if "model" not in df.columns:
        raise KeyError("Missing required column: model")
    if error_col not in df.columns:
        raise KeyError(f"Missing required column: {error_col}")
    if "params" not in df.columns:
        raise KeyError("Missing required column: params")

    if len(df) < 2:
        return pd.DataFrame(columns=PAIRWISE_CPL_COLUMNS)

    ordered = sort_for_best_error(df, ["model"])
    rows: list[dict[str, object]] = []
    for left_idx, right_idx in combinations(range(len(ordered)), 2):
        left = ordered.iloc[left_idx]
        right = ordered.iloc[right_idx]
        model_a = str(left["model"])
        model_b = str(right["model"])
        error_a = float(left[error_col])
        error_b = float(right[error_col])
        params_a = float(left["params"])
        params_b = float(right["params"])
        if error_a <= 0 or error_b <= 0:
            raise ValueError(f"{error_col} must be positive to compute CPL, got {error_a} and {error_b}.")
        if params_a <= 0 or params_b <= 0:
            raise ValueError(f"params must be positive to compute CPL, got {params_a} and {params_b}.")

        if math.isclose(params_a, params_b, rel_tol=1e-12, abs_tol=1e-12):
            if math.isclose(error_a, error_b, rel_tol=1e-12, abs_tol=1e-12):
                state = "equal_models"
                winner: str | pd.NAType = pd.NA
            else:
                state = "equal_params"
                winner = model_a if error_a < error_b else model_b
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "smaller_model": pd.NA,
                    "larger_model": pd.NA,
                    "error_metric": metric,
                    "error_smaller": np.nan,
                    "error_larger": np.nan,
                    "params_smaller": params_a,
                    "params_larger": params_b,
                    "state": state,
                    "cpl": np.nan,
                    "winner": winner,
                }
            )
            continue

        if params_a < params_b:
            smaller_model = model_a
            larger_model = model_b
            error_smaller = error_a
            error_larger = error_b
            params_smaller = params_a
            params_larger = params_b
        else:
            smaller_model = model_b
            larger_model = model_a
            error_smaller = error_b
            error_larger = error_a
            params_smaller = params_b
            params_larger = params_a

        if error_smaller <= error_larger or math.isclose(error_smaller, error_larger, rel_tol=1e-12, abs_tol=1e-12):
            state = "dominates"
            cpl = math.inf
            winner = smaller_model
        else:
            state = "tradeoff"
            cpl = math.log(params_larger / params_smaller) / math.log(error_smaller / error_larger)
            if math.isclose(cpl, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                winner = pd.NA
            elif cpl > 1.0:
                winner = smaller_model
            else:
                winner = larger_model

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "smaller_model": smaller_model,
                "larger_model": larger_model,
                "error_metric": metric,
                "error_smaller": error_smaller,
                "error_larger": error_larger,
                "params_smaller": params_smaller,
                "params_larger": params_larger,
                "state": state,
                "cpl": cpl,
                "winner": winner,
            }
        )

    return pd.DataFrame(rows, columns=PAIRWISE_CPL_COLUMNS)


def add_epes_cpls_columns(df: pd.DataFrame, efficiency_error_metric: str = "mae") -> pd.DataFrame:
    metric = validate_efficiency_error_metric(efficiency_error_metric)
    error_col = resolve_error_column(metric)
    if error_col not in df.columns:
        raise KeyError(f"Missing required column: {error_col}")
    if "params" not in df.columns:
        raise KeyError("Missing required column: params")
    if "model" not in df.columns:
        raise KeyError("Missing required column: model")

    enriched = df.copy()
    best_row = select_best_error_row(enriched, metric)
    best_error_value = float(best_row[error_col])
    min_params = float(enriched["params"].min())
    if best_error_value <= 0:
        raise ValueError(f"{error_col} must be positive to compute EPES/CPLS, got {best_error_value}.")
    if min_params <= 0:
        raise ValueError(f"params must be positive to compute EPES/CPLS, got {min_params}.")

    enriched["efficiency_error_metric"] = metric
    enriched["efficiency_error_value"] = enriched[error_col].astype(float)
    enriched["best_error_value"] = best_error_value
    enriched["min_params"] = min_params
    enriched["error_ratio_vs_best"] = enriched["efficiency_error_value"] / best_error_value
    enriched["param_ratio_vs_smallest"] = enriched["params"].astype(float) / min_params
    enriched["epes"] = 1.0 / (
        enriched["error_ratio_vs_best"] + enriched["param_ratio_vs_smallest"] - 1.0
    )

    pairwise_df = build_pairwise_cpl_dataframe(enriched, efficiency_error_metric=metric)
    score_by_model: dict[str, dict[str, float | int]] = {
        str(model_name): {"cpls_wins": 0, "cpls_losses": 0, "cpls_ties": 0}
        for model_name in enriched["model"].astype(str).tolist()
    }
    for pair_row in pairwise_df.itertuples(index=False):
        model_a = str(pair_row.model_a)
        model_b = str(pair_row.model_b)
        winner = pair_row.winner
        if pd.isna(winner):
            score_by_model[model_a]["cpls_ties"] += 1
            score_by_model[model_b]["cpls_ties"] += 1
            continue

        winner_name = str(winner)
        loser_name = model_b if winner_name == model_a else model_a
        score_by_model[winner_name]["cpls_wins"] += 1
        score_by_model[loser_name]["cpls_losses"] += 1

    denominator = max(len(enriched) - 1, 1)
    enriched["cpls_wins"] = enriched["model"].astype(str).map(lambda name: score_by_model[name]["cpls_wins"]).astype(int)
    enriched["cpls_losses"] = enriched["model"].astype(str).map(lambda name: score_by_model[name]["cpls_losses"]).astype(int)
    enriched["cpls_ties"] = enriched["model"].astype(str).map(lambda name: score_by_model[name]["cpls_ties"]).astype(int)
    enriched["cpls"] = (
        enriched["model"].astype(str).map(
            lambda name: (score_by_model[name]["cpls_wins"] - score_by_model[name]["cpls_losses"]) / denominator
        )
    ).astype(float)
    return enriched


def add_epes_psl_columns(df: pd.DataFrame, efficiency_error_metric: str = "mae") -> pd.DataFrame:
    return add_epes_cpls_columns(df, efficiency_error_metric=efficiency_error_metric)
