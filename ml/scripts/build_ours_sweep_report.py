from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_benchmark_report import build_config_signatures
from efficiency_metrics import DEFAULT_ACCURACY_TOLERANCE, add_parameter_efficiency_columns, format_tolerance_label
from results_layout import iter_run_dirs, load_run_metadata

RAW_SORT_KEYS = ["sweep_name", "sweep_value", "dataset_name", "target_col", "lookback", "horizon", "seed", "model"]
AGG_KEYS = ["sweep_name", "sweep_value", "dataset_name", "target_col", "lookback", "horizon", "model"]
WINNER_KEYS = ["sweep_name", "dataset_name", "target_col", "lookback", "horizon"]
BASELINE_JOIN_KEYS = ["dataset_name", "target_col", "lookback", "horizon"]
SELECTION_GROUP_KEYS = ["sweep_name", "dataset_name", "target_col", "lookback"]
DEFAULT_SELECTION_HORIZONS = (336, 720)


def load_sweep_frame(run_dir: Path) -> pd.DataFrame | None:
    metadata_path = run_dir / "sweep_metadata.json"
    metrics_path = run_dir / "metrics.csv"
    if not metadata_path.exists() or not metrics_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as fp:
        sweep_metadata = json.load(fp)
    metadata = load_run_metadata(run_dir)
    frame = pd.read_csv(metrics_path)
    frame["sweep_name"] = str(sweep_metadata["sweep_name"])
    frame["sweep_value"] = str(sweep_metadata["sweep_value"])
    frame["dataset_name"] = metadata.get("dataset_name", frame.get("dataset_name", ""))
    frame["target_col"] = metadata.get("target_col", frame.get("target_col", ""))
    frame["lookback"] = int(metadata["lookback"])
    frame["horizon"] = int(metadata["horizon"])
    frame["seed"] = int(metadata["seed"])
    config_signature, benchmark_signature = build_config_signatures(run_dir)
    frame["config_signature"] = config_signature
    frame["benchmark_signature"] = benchmark_signature
    return frame


def load_analysis_frame(run_dir: Path, filename: str) -> pd.DataFrame | None:
    metadata_path = run_dir / "sweep_metadata.json"
    analysis_path = run_dir / "summary" / filename
    if not metadata_path.exists() or not analysis_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as fp:
        sweep_metadata = json.load(fp)
    metadata = load_run_metadata(run_dir)
    frame = pd.read_csv(analysis_path)
    frame["sweep_name"] = str(sweep_metadata["sweep_name"])
    frame["sweep_value"] = str(sweep_metadata["sweep_value"])
    frame["dataset_name"] = metadata.get("dataset_name", "")
    frame["target_col"] = metadata.get("target_col", "")
    frame["lookback"] = int(metadata["lookback"])
    frame["horizon"] = int(metadata["horizon"])
    frame["seed"] = int(metadata["seed"])
    return frame


def validate_group_consistency(frame: pd.DataFrame) -> None:
    inconsistent_configs = (
        frame.groupby(AGG_KEYS, dropna=False)["config_signature"].nunique(dropna=False).reset_index(name="count")
    )
    inconsistent_configs = inconsistent_configs[inconsistent_configs["count"] > 1]
    if not inconsistent_configs.empty:
        row = inconsistent_configs.iloc[0]
        raise ValueError(
            "Mixed config signatures found for one Ours sweep cell: "
            f"{ {key: row[key] for key in AGG_KEYS} }"
        )

    inconsistent_protocols = (
        frame.groupby(WINNER_KEYS, dropna=False)["benchmark_signature"].nunique(dropna=False).reset_index(name="count")
    )
    inconsistent_protocols = inconsistent_protocols[inconsistent_protocols["count"] > 1]
    if not inconsistent_protocols.empty:
        row = inconsistent_protocols.iloc[0]
        raise ValueError(
            "Mixed benchmark protocol signatures found across one Ours sweep: "
            f"{ {key: row[key] for key in WINNER_KEYS} }"
        )


def validate_seed_coverage(frame: pd.DataFrame) -> None:
    if "seed" not in frame.columns:
        return
    for winner_values, group in frame.groupby(WINNER_KEYS, dropna=False):
        per_cell = (
            group.groupby(["sweep_value", "model"], dropna=False)["seed"]
            .apply(lambda values: tuple(sorted(int(value) for value in pd.unique(values))))
        )
        unique_seed_sets = {seed_set for seed_set in per_cell.tolist()}
        if len(unique_seed_sets) > 1:
            winner_lookup = dict(zip(WINNER_KEYS, winner_values if isinstance(winner_values, tuple) else (winner_values,)))
            raise ValueError(f"Inconsistent seed coverage across Ours sweep cells for {winner_lookup}: {per_cell.to_dict()}")


def build_metric_winners(
    aggregated: pd.DataFrame,
    metric: str,
    prefix: str,
) -> pd.DataFrame | None:
    if metric not in aggregated.columns:
        return None
    winner_idx = aggregated.groupby(WINNER_KEYS, dropna=False)[metric].idxmin()
    winners = aggregated.loc[winner_idx, WINNER_KEYS + ["sweep_value", "model", metric]].copy()
    return winners.rename(
        columns={
            "sweep_value": f"{prefix}_sweep_value",
            "model": f"{prefix}_model",
            metric: f"{prefix}_value",
        }
    )


def choose_accuracy_metric(aggregated: pd.DataFrame) -> str:
    if "mean_test_mae_scaled" in aggregated.columns and aggregated["mean_test_mae_scaled"].notna().any():
        return "mean_test_mae_scaled"
    return "mean_test_mae"


def build_variant_label(frame: pd.DataFrame) -> pd.Series:
    return frame["sweep_value"].astype(str) + ":" + frame["model"].astype(str)


def add_variant_efficiency_columns(
    aggregated: pd.DataFrame,
    *,
    accuracy_metric: str,
    tolerance: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in aggregated.groupby(WINNER_KEYS, dropna=False, sort=False):
        labeled_group = group.copy()
        labeled_group["variant_label"] = build_variant_label(labeled_group)

        efficiency_input = labeled_group.copy()
        original_model = efficiency_input["model"].copy()
        efficiency_input["model"] = efficiency_input["variant_label"]
        enriched = add_parameter_efficiency_columns(
            efficiency_input,
            error_col=accuracy_metric,
            params_col="mean_params",
            tolerance=tolerance,
        )
        enriched["model"] = original_model
        frames.append(enriched)

    return pd.concat(frames, ignore_index=True).sort_values(AGG_KEYS).reset_index(drop=True)


def build_parameter_efficiency_winners(
    aggregated: pd.DataFrame,
    *,
    accuracy_metric: str,
    tolerance: float,
) -> pd.DataFrame:
    label = format_tolerance_label(tolerance)
    efficient_col = f"parameter_efficient_{label}"
    within_col = f"within_{label}_accuracy_tolerance"
    rows: list[dict[str, object]] = []
    for winner_values, group in aggregated.groupby(WINNER_KEYS, dropna=False, sort=False):
        key_values = winner_values if isinstance(winner_values, tuple) else (winner_values,)
        row: dict[str, object] = dict(zip(WINNER_KEYS, key_values))
        ordered = group.sort_values([accuracy_metric, "mean_params", "variant_label"], kind="mergesort")
        accuracy_row = ordered.iloc[0]
        efficient_candidates = group.loc[group[efficient_col].astype(bool)]
        if efficient_candidates.empty:
            raise ValueError(f"No tolerance-efficient Ours sweep variant found for {row}.")
        efficient_row = efficient_candidates.sort_values(["mean_params", accuracy_metric, "variant_label"], kind="mergesort").iloc[0]
        row.update(
            {
                "accuracy_winner_sweep_value": accuracy_row["sweep_value"],
                "accuracy_winner_model": accuracy_row["model"],
                "accuracy_winner_variant_label": accuracy_row["variant_label"],
                "accuracy_winner_metric": accuracy_metric,
                "accuracy_winner_error": float(accuracy_row[accuracy_metric]),
                "accuracy_winner_params": float(accuracy_row["mean_params"]),
                f"parameter_efficient_{label}_sweep_value": efficient_row["sweep_value"],
                f"parameter_efficient_{label}_model": efficient_row["model"],
                f"parameter_efficient_{label}_variant_label": efficient_row["variant_label"],
                "parameter_efficiency_rule": (
                    f"smallest mean_params within {tolerance:.2%} of the best {accuracy_metric}"
                ),
                "parameter_efficient_error": float(efficient_row[accuracy_metric]),
                "parameter_efficient_params": float(efficient_row["mean_params"]),
                "parameter_efficient_relative_error_vs_best": float(efficient_row["relative_error_vs_best"]),
                "parameter_efficient_pareto_optimal": bool(efficient_row["pareto_optimal"]),
                "accuracy_tolerance_threshold": float(efficient_row["accuracy_tolerance_threshold"]),
                "within_tolerance_column": within_col,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(WINNER_KEYS).reset_index(drop=True)


def parse_horizon_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    horizons = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not horizons:
        raise ValueError("At least one selection horizon must be provided.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"Selection horizons must be positive integers: {raw}")
    return horizons


def sanitize_column_label(value: object) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return text or "baseline"


def build_baseline_tolerance_frame(
    aggregated: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    *,
    baseline_model: str,
    accuracy_metric: str,
    tolerance: float,
) -> pd.DataFrame:
    missing = [column for column in [*BASELINE_JOIN_KEYS, "model", accuracy_metric] if column not in baseline_summary.columns]
    if missing:
        raise KeyError(f"Baseline summary is missing required columns: {missing}")

    baseline_candidates = baseline_summary.loc[
        baseline_summary["model"].astype(str).str.lower() == baseline_model.lower()
    ].copy()
    if baseline_candidates.empty:
        raise ValueError(f"No baseline rows found for model '{baseline_model}'.")
    if "mean_params" not in baseline_candidates.columns:
        baseline_candidates["mean_params"] = np.nan

    baseline_best = (
        baseline_candidates.sort_values(BASELINE_JOIN_KEYS + [accuracy_metric, "mean_params", "model"], kind="mergesort")
        .groupby(BASELINE_JOIN_KEYS, dropna=False, as_index=False)
        .first()
        .loc[:, [*BASELINE_JOIN_KEYS, "model", accuracy_metric, "mean_params"]]
        .rename(
            columns={
                "model": "baseline_model",
                accuracy_metric: "baseline_error",
                "mean_params": "baseline_params",
            }
        )
    )

    merged = aggregated.merge(baseline_best, on=BASELINE_JOIN_KEYS, how="left", validate="many_to_one")
    if merged["baseline_error"].isna().any():
        missing_groups = merged.loc[merged["baseline_error"].isna(), BASELINE_JOIN_KEYS].drop_duplicates()
        raise ValueError(f"Missing baseline rows for sweep groups: {missing_groups.to_dict(orient='records')}")

    label = format_tolerance_label(tolerance)
    baseline_label = sanitize_column_label(baseline_model)
    threshold_col = f"{baseline_label}_{label}_accuracy_tolerance_threshold"
    relative_col = f"relative_error_vs_{baseline_label}"
    within_col = f"within_{label}_{baseline_label}_tolerance"
    merged["baseline_accuracy_metric"] = accuracy_metric
    merged[threshold_col] = merged["baseline_error"].astype(float) * (1.0 + float(tolerance))
    merged[relative_col] = (merged[accuracy_metric].astype(float) / merged["baseline_error"].astype(float)) - 1.0
    merged[within_col] = merged[accuracy_metric].astype(float) <= merged[threshold_col].astype(float) + 1e-12
    merged[f"within_{label}_baseline_tolerance"] = merged[within_col]
    return merged.sort_values(AGG_KEYS).reset_index(drop=True)


def build_cross_horizon_selection(
    baseline_tolerance: pd.DataFrame,
    *,
    selection_horizons: tuple[int, ...],
    baseline_model: str,
    tolerance: float,
    accuracy_metric: str,
) -> pd.DataFrame:
    label = format_tolerance_label(tolerance)
    baseline_label = sanitize_column_label(baseline_model)
    within_col = f"within_{label}_{baseline_label}_tolerance"
    relative_col = f"relative_error_vs_{baseline_label}"
    required_horizons = tuple(dict.fromkeys(int(horizon) for horizon in selection_horizons))
    rows: list[dict[str, object]] = []

    for group_values, group in baseline_tolerance.groupby(SELECTION_GROUP_KEYS, dropna=False, sort=False):
        group_key = dict(zip(SELECTION_GROUP_KEYS, group_values if isinstance(group_values, tuple) else (group_values,)))
        for variant_values, variant_df in group.groupby(["sweep_value", "model", "variant_label"], dropna=False, sort=False):
            sweep_value, model, variant_label = variant_values if isinstance(variant_values, tuple) else (variant_values, "", "")
            selected_horizon_df = variant_df[variant_df["horizon"].astype(int).isin(required_horizons)].copy()
            horizon_set = {int(value) for value in selected_horizon_df["horizon"].tolist()}
            has_all_horizons = set(required_horizons).issubset(horizon_set)
            within_all = bool(has_all_horizons and selected_horizon_df[within_col].astype(bool).all())
            row: dict[str, object] = {
                **group_key,
                "sweep_value": sweep_value,
                "model": model,
                "variant_label": variant_label,
                "selection_horizons": ",".join(str(horizon) for horizon in required_horizons),
                "horizon_count": int(len(horizon_set)),
                "has_all_selection_horizons": bool(has_all_horizons),
                "within_all_selection_horizons": bool(within_all),
                "mean_params": float(selected_horizon_df["mean_params"].mean()) if not selected_horizon_df.empty else np.nan,
                "max_params": float(selected_horizon_df["mean_params"].max()) if not selected_horizon_df.empty else np.nan,
                "mean_accuracy_error": float(selected_horizon_df[accuracy_metric].mean()) if not selected_horizon_df.empty else np.nan,
                "max_relative_error_vs_baseline": float(selected_horizon_df[relative_col].max()) if not selected_horizon_df.empty else np.nan,
            }
            for horizon in required_horizons:
                horizon_rows = selected_horizon_df[selected_horizon_df["horizon"].astype(int) == horizon]
                if horizon_rows.empty:
                    row[f"h{horizon}_error"] = np.nan
                    row[f"h{horizon}_relative_error_vs_baseline"] = np.nan
                    row[f"h{horizon}_within_baseline_tolerance"] = False
                else:
                    horizon_row = horizon_rows.sort_values([accuracy_metric, "mean_params", "variant_label"], kind="mergesort").iloc[0]
                    row[f"h{horizon}_error"] = float(horizon_row[accuracy_metric])
                    row[f"h{horizon}_relative_error_vs_baseline"] = float(horizon_row[relative_col])
                    row[f"h{horizon}_within_baseline_tolerance"] = bool(horizon_row[within_col])
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    selection = pd.DataFrame(rows)
    selection["selected_variant"] = False
    selection["selection_rule"] = ""
    preferred_tie_breakers = [
        column
        for column in ["h720_error", "h336_error", "mean_accuracy_error", "variant_label"]
        if column in selection.columns
    ]
    for group_values, group in selection.groupby(SELECTION_GROUP_KEYS, dropna=False, sort=False):
        group_idx = group.index
        candidates = group.loc[group["within_all_selection_horizons"].astype(bool)].copy()
        if not candidates.empty:
            ordered = candidates.sort_values(["max_params", *preferred_tie_breakers], kind="mergesort")
            selected_idx = ordered.index[0]
            rule = f"smallest params within {label} of {baseline_model} on all selection horizons"
        else:
            fallback = group.loc[group["sweep_value"].astype(str) == "default_11k"].copy()
            if fallback.empty:
                fallback = group.copy()
                rule = "fallback_no_variant_satisfied_tolerance_default_11k_missing"
            else:
                rule = "fallback_default_11k_no_variant_satisfied_tolerance"
            selected_idx = fallback.sort_values(["max_params", *preferred_tie_breakers], kind="mergesort").index[0]
        selection.loc[group_idx, "selection_rule"] = rule
        selection.loc[selected_idx, "selected_variant"] = True

    return selection.sort_values(SELECTION_GROUP_KEYS + ["selected_variant", "max_params", "variant_label"], ascending=[True, True, True, True, False, True, True]).reset_index(drop=True)


def sanitize_group_stem(group_key: dict[str, object]) -> str:
    return (
        f"{group_key['sweep_name']}_{group_key['dataset_name']}_{group_key['target_col']}"
        f"_lb{int(group_key['lookback'])}_h{int(group_key['horizon'])}"
    ).replace(" ", "_")


def build_branch_usage_summary(branch_raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["sweep_name", "dataset_name", "target_col", "lookback", "horizon", "sweep_value", "model"]
    for group_values, group_df in branch_raw_df.groupby(group_columns, dropna=False):
        base = dict(zip(group_columns, group_values if isinstance(group_values, tuple) else (group_values,)))
        for branch in ("trend", "seasonal", "transient"):
            rows.append(
                {
                    **base,
                    "branch": branch,
                    "mean_router_weight": float(group_df[f"{branch}_weight"].mean()),
                    "mean_abs_coeff": float(group_df[f"{branch}_abs_coeff"].mean()),
                    "window_count": int(group_df["window_end_index"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(group_columns + ["branch"]).reset_index(drop=True)


def build_frequency_summary(frequency_raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        frequency_raw_df.groupby(
            ["sweep_name", "dataset_name", "target_col", "lookback", "horizon", "sweep_value", "model", "mode_index"],
            dropna=False,
        )
        .agg(
            omega_mean=("omega", "mean"),
            omega_std=("omega", "std"),
            effective_omega_mean=("effective_omega", "mean"),
            effective_omega_std=("effective_omega", "std"),
            delta_omega_mean_abs=("delta_omega", lambda values: float(np.mean(np.abs(values)))),
            window_count=("window_end_index", "nunique"),
        )
        .reset_index()
        .sort_values(["sweep_name", "dataset_name", "target_col", "lookback", "horizon", "sweep_value", "model", "mode_index"])
        .reset_index(drop=True)
    )


def save_group_pareto_plot(
    group_df: pd.DataFrame,
    output_path: Path,
    *,
    accuracy_metric: str,
    tolerance: float,
) -> None:
    has_latency = "mean_latency_ms_per_sample" in group_df.columns
    fig, axes_raw = plt.subplots(1, 2 if has_latency else 1, figsize=(12 if has_latency else 7.2, 4.8))
    axes = np.atleast_1d(axes_raw)
    labels = [f"{row['sweep_value']}:{row['model']}" for _, row in group_df.iterrows()]
    label = format_tolerance_label(tolerance)
    efficient_col = f"parameter_efficient_{label}"
    within_col = f"within_{label}_accuracy_tolerance"
    y_values = group_df[accuracy_metric].astype(float).to_numpy()
    pareto_mask = (
        group_df["pareto_optimal"].astype(bool).to_numpy()
        if "pareto_optimal" in group_df.columns
        else np.ones(len(group_df), dtype=bool)
    )
    efficient_mask = (
        group_df[efficient_col].astype(bool).to_numpy()
        if efficient_col in group_df.columns
        else np.zeros(len(group_df), dtype=bool)
    )
    within_mask = (
        group_df[within_col].astype(bool).to_numpy()
        if within_col in group_df.columns
        else np.zeros(len(group_df), dtype=bool)
    )

    x_params = np.log10(group_df["mean_params"].astype(float).to_numpy())
    axes[0].scatter(x_params[~pareto_mask], y_values[~pareto_mask], c="#8d99ae", s=58, label="dominated")
    axes[0].scatter(x_params[pareto_mask], y_values[pareto_mask], c="#2a9d8f", s=70, label="Pareto-optimal")
    if within_mask.any():
        axes[0].scatter(
            x_params[within_mask],
            y_values[within_mask],
            facecolors="none",
            edgecolors="#355070",
            linewidths=1.5,
            s=118,
            label=f"within {tolerance:.0%}",
        )
    if efficient_mask.any():
        axes[0].scatter(
            x_params[efficient_mask],
            y_values[efficient_mask],
            marker="*",
            c="#c8553d",
            edgecolors="#5f0f00",
            linewidths=0.7,
            s=180,
            label=f"efficient {label}",
        )
    if "accuracy_tolerance_threshold" in group_df.columns:
        axes[0].axhline(
            float(group_df["accuracy_tolerance_threshold"].iat[0]),
            color="#6d597a",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

    for idx, label in enumerate(labels):
        axes[0].annotate(label, (float(x_params[idx]), float(y_values[idx])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    y_label = "Mean scaled MAE" if accuracy_metric.endswith("_scaled") else "Mean test MAE"
    axes[0].set_title("Pareto: Accuracy vs Params")
    axes[0].set_xlabel("log10(mean params)")
    axes[0].set_ylabel(y_label)
    axes[0].legend(fontsize=8)

    if has_latency:
        axes[1].scatter(group_df["mean_latency_ms_per_sample"], y_values, c="#b56576", s=60)
        for idx, label in enumerate(labels):
            axes[1].annotate(
                label,
                (float(group_df["mean_latency_ms_per_sample"].iat[idx]), float(y_values[idx])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].set_title("Accuracy vs Latency")
        axes[1].set_xlabel("Mean latency (ms/sample)")
        axes[1].set_ylabel(y_label)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_group_branch_usage_plot(group_df: pd.DataFrame, output_path: Path) -> None:
    labels = sorted({f"{row['sweep_value']}:{row['model']}" for _, row in group_df.iterrows()})
    branches = ["trend", "seasonal", "transient"]
    colors = {"trend": "#355070", "seasonal": "#b56576", "transient": "#2a9d8f"}
    pivot_weight = (
        group_df.assign(label=group_df["sweep_value"] + ":" + group_df["model"])
        .pivot(index="label", columns="branch", values="mean_router_weight")
        .reindex(labels)
        .fillna(0.0)
    )
    pivot_coeff = (
        group_df.assign(label=group_df["sweep_value"] + ":" + group_df["model"])
        .pivot(index="label", columns="branch", values="mean_abs_coeff")
        .reindex(labels)
        .fillna(0.0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bottom_weight = np.zeros(len(labels), dtype=np.float64)
    bottom_coeff = np.zeros(len(labels), dtype=np.float64)
    for branch in branches:
        weight_values = pivot_weight.get(branch, pd.Series(0.0, index=labels)).to_numpy(dtype=np.float64)
        coeff_values = pivot_coeff.get(branch, pd.Series(0.0, index=labels)).to_numpy(dtype=np.float64)
        axes[0].bar(labels, weight_values, bottom=bottom_weight, color=colors[branch], label=branch)
        axes[1].bar(labels, coeff_values, bottom=bottom_coeff, color=colors[branch], label=branch)
        bottom_weight += weight_values
        bottom_coeff += coeff_values
    axes[0].set_title("Branch Usage: Router Weight")
    axes[1].set_title("Branch Usage: Mean Abs Coefficient")
    axes[0].set_ylabel("Mean weight")
    axes[1].set_ylabel("Mean |coefficient|")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_group_frequency_plot(group_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, label_df in group_df.assign(label=group_df["sweep_value"] + ":" + group_df["model"]).groupby("label", dropna=False):
        mode_index = label_df["mode_index"].to_numpy(dtype=np.int64)
        axes[0].errorbar(
            mode_index,
            label_df["omega_mean"].to_numpy(dtype=np.float64),
            yerr=label_df["omega_std"].fillna(0.0).to_numpy(dtype=np.float64),
            marker="o",
            linestyle="--",
            label=f"{label} omega",
            alpha=0.7,
        )
        axes[0].errorbar(
            mode_index,
            label_df["effective_omega_mean"].to_numpy(dtype=np.float64),
            yerr=label_df["effective_omega_std"].fillna(0.0).to_numpy(dtype=np.float64),
            marker="o",
            linestyle="-",
            label=f"{label} effective",
        )
        axes[1].plot(
            mode_index,
            label_df["delta_omega_mean_abs"].to_numpy(dtype=np.float64),
            marker="o",
            linestyle="-",
            label=label,
        )
    axes[0].set_title("Learned Frequency")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Omega")
    axes[1].set_title("Group Frequency Offset")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Mean |delta_omega|")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_group_analysis_outputs(
    results_dir: Path,
    aggregated: pd.DataFrame,
    branch_summary_df: pd.DataFrame | None,
    frequency_summary_df: pd.DataFrame | None,
    *,
    accuracy_metric: str,
    tolerance: float,
) -> None:
    pareto_dir = results_dir / "pareto_by_group"
    branch_dir = results_dir / "branch_usage_by_group"
    frequency_dir = results_dir / "frequency_by_group"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    branch_dir.mkdir(parents=True, exist_ok=True)
    frequency_dir.mkdir(parents=True, exist_ok=True)

    for winner_values, group_df in aggregated.groupby(WINNER_KEYS, dropna=False):
        group_key = dict(zip(WINNER_KEYS, winner_values if isinstance(winner_values, tuple) else (winner_values,)))
        stem = sanitize_group_stem(group_key)
        save_group_pareto_plot(
            group_df.sort_values([accuracy_metric, "mean_params", "variant_label"]).reset_index(drop=True),
            pareto_dir / f"{stem}.png",
            accuracy_metric=accuracy_metric,
            tolerance=tolerance,
        )

        if branch_summary_df is not None:
            mask = pd.Series(True, index=branch_summary_df.index)
            for key, value in group_key.items():
                mask &= branch_summary_df[key] == value
            branch_group_df = branch_summary_df.loc[mask].copy()
            if not branch_group_df.empty:
                branch_group_df.to_csv(branch_dir / f"{stem}.csv", index=False)
                save_group_branch_usage_plot(branch_group_df, branch_dir / f"{stem}.png")

        if frequency_summary_df is not None:
            mask = pd.Series(True, index=frequency_summary_df.index)
            for key, value in group_key.items():
                mask &= frequency_summary_df[key] == value
            frequency_group_df = frequency_summary_df.loc[mask].copy()
            if not frequency_group_df.empty:
                frequency_group_df.to_csv(frequency_dir / f"{stem}.csv", index=False)
                save_group_frequency_plot(frequency_group_df, frequency_dir / f"{stem}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Ours sweep runs across seeds.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--accuracy-tolerance", type=float, default=DEFAULT_ACCURACY_TOLERANCE)
    parser.add_argument("--baseline-summary-path", type=Path, default=None)
    parser.add_argument("--baseline-model", type=str, default="patchtst")
    parser.add_argument(
        "--selection-horizons",
        type=str,
        default=",".join(str(horizon) for horizon in DEFAULT_SELECTION_HORIZONS),
        help="Comma-separated horizons used for the cross-horizon defense selection rule.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accuracy_tolerance < 0:
        raise ValueError(f"--accuracy-tolerance must be non-negative, got {args.accuracy_tolerance}.")
    run_dirs = iter_run_dirs(args.results_dir)
    run_frames = [frame for frame in (load_sweep_frame(run_dir) for run_dir in run_dirs) if frame is not None]
    if not run_frames:
        raise FileNotFoundError(f"No Ours sweep run directories found in {args.results_dir}")

    summary = pd.concat(run_frames, ignore_index=True)
    validate_group_consistency(summary)
    validate_seed_coverage(summary)
    sort_keys = [column for column in RAW_SORT_KEYS if column in summary.columns]
    summary = summary.sort_values(sort_keys).reset_index(drop=True)

    aggregations: dict[str, tuple[str, str]] = {
        "seed_count": ("seed", "nunique"),
        "mean_test_mae": ("test_mae", "mean"),
        "mean_test_rmse": ("test_rmse", "mean"),
        "mean_params": ("params", "mean"),
    }
    optional_aggs = {
        "mean_test_mae_scaled": ("test_mae_scaled", "mean"),
        "mean_test_rmse_scaled": ("test_rmse_scaled", "mean"),
        "mean_val_mae_scaled": ("val_mae_scaled", "mean"),
        "mean_val_rmse_scaled": ("val_rmse_scaled", "mean"),
        "mean_latency_ms_per_sample": ("test_inference_ms_per_sample", "mean"),
        "mean_peak_memory_mb": ("test_peak_memory_mb", "mean"),
    }
    for output_name, (column_name, agg_fn) in optional_aggs.items():
        if column_name in summary.columns:
            aggregations[output_name] = (column_name, agg_fn)

    aggregated = (
        summary.groupby(AGG_KEYS, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(AGG_KEYS)
        .reset_index(drop=True)
    )

    accuracy_metric = choose_accuracy_metric(aggregated)
    aggregated = add_variant_efficiency_columns(
        aggregated,
        accuracy_metric=accuracy_metric,
        tolerance=args.accuracy_tolerance,
    )
    parameter_efficiency_winners = build_parameter_efficiency_winners(
        aggregated,
        accuracy_metric=accuracy_metric,
        tolerance=args.accuracy_tolerance,
    )

    winners = None
    winner_specs = (
        ("mean_test_mae_scaled", "best_mean_test_mae_scaled"),
        ("mean_test_mae", "best_mean_test_mae"),
        ("mean_test_rmse_scaled", "best_mean_test_rmse_scaled"),
        ("mean_test_rmse", "best_mean_test_rmse"),
        ("mean_params", "best_mean_params"),
        ("mean_latency_ms_per_sample", "best_mean_latency_ms_per_sample"),
        ("mean_peak_memory_mb", "best_mean_peak_memory_mb"),
    )
    for metric, prefix in winner_specs:
        metric_winners = build_metric_winners(aggregated, metric, prefix)
        if metric_winners is None:
            continue
        winners = metric_winners if winners is None else winners.merge(metric_winners, on=WINNER_KEYS, how="outer")

    if winners is None:
        raise ValueError("No winner metrics could be derived from the sweep results.")
    parameter_efficiency_merge_columns = [
        column
        for column in parameter_efficiency_winners.columns
        if column not in WINNER_KEYS + ["accuracy_winner_sweep_value", "accuracy_winner_model", "accuracy_winner_variant_label", "accuracy_winner_metric", "accuracy_winner_error", "accuracy_winner_params"]
    ]
    winners = winners.merge(
        parameter_efficiency_winners[WINNER_KEYS + parameter_efficiency_merge_columns],
        on=WINNER_KEYS,
        how="outer",
    )
    winners = winners.sort_values(WINNER_KEYS).reset_index(drop=True)

    summary_path = args.results_dir / "sweep_summary.csv"
    raw_metrics_path = args.results_dir / "sweep_raw_metrics.csv"
    winners_path = args.results_dir / "sweep_winners.csv"
    parameter_efficiency_path = args.results_dir / "sweep_parameter_efficiency.csv"
    summary.to_csv(raw_metrics_path, index=False)
    aggregated.to_csv(summary_path, index=False)
    winners.to_csv(winners_path, index=False)
    parameter_efficiency_winners.to_csv(parameter_efficiency_path, index=False)

    branch_frames = [frame for frame in (load_analysis_frame(run_dir, "ours_branch_usage_raw.csv") for run_dir in run_dirs) if frame is not None]
    frequency_frames = [frame for frame in (load_analysis_frame(run_dir, "ours_frequency_raw.csv") for run_dir in run_dirs) if frame is not None]
    branch_summary_df = None
    frequency_summary_df = None
    if branch_frames:
        branch_summary_df = build_branch_usage_summary(pd.concat(branch_frames, ignore_index=True))
    if frequency_frames:
        frequency_summary_df = build_frequency_summary(pd.concat(frequency_frames, ignore_index=True))
    write_group_analysis_outputs(
        args.results_dir,
        aggregated,
        branch_summary_df,
        frequency_summary_df,
        accuracy_metric=accuracy_metric,
        tolerance=args.accuracy_tolerance,
    )

    if args.baseline_summary_path is not None:
        selection_horizons = parse_horizon_list(args.selection_horizons)
        if selection_horizons is None:
            selection_horizons = tuple(sorted(int(value) for value in aggregated["horizon"].unique()))
        baseline_summary = pd.read_csv(args.baseline_summary_path)
        baseline_tolerance = build_baseline_tolerance_frame(
            aggregated,
            baseline_summary,
            baseline_model=args.baseline_model,
            accuracy_metric=accuracy_metric,
            tolerance=args.accuracy_tolerance,
        )
        baseline_tolerance_path = args.results_dir / "sweep_baseline_tolerance.csv"
        baseline_tolerance.to_csv(baseline_tolerance_path, index=False)
        variant_selection = build_cross_horizon_selection(
            baseline_tolerance,
            selection_horizons=selection_horizons,
            baseline_model=args.baseline_model,
            tolerance=args.accuracy_tolerance,
            accuracy_metric=accuracy_metric,
        )
        variant_selection_path = args.results_dir / "sweep_variant_selection.csv"
        selected_variant_path = args.results_dir / "sweep_selected_variant.csv"
        variant_selection.to_csv(variant_selection_path, index=False)
        variant_selection.loc[variant_selection["selected_variant"].astype(bool)].to_csv(selected_variant_path, index=False)
        print(f"Saved baseline tolerance table to: {baseline_tolerance_path}")
        print(f"Saved variant selection table to: {variant_selection_path}")
        print(f"Saved selected variant table to: {selected_variant_path}")

    print(f"Saved Ours sweep summary to: {summary_path}")
    print(f"Saved Ours raw metrics to: {raw_metrics_path}")
    print(f"Saved Ours sweep winners to: {winners_path}")
    print(f"Saved Ours parameter-efficiency winners to: {parameter_efficiency_path}")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
