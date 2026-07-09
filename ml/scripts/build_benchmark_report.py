from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from efficiency_metrics import (
    DEFAULT_ACCURACY_TOLERANCE,
    add_parameter_efficiency_columns,
    format_tolerance_label,
)
from results_layout import iter_run_dirs, load_run_metadata

GROUP_KEYS = ["dataset_name", "target_col", "lookback", "horizon", "model"]
WINNER_KEYS = ["dataset_name", "target_col", "lookback", "horizon"]
VOLATILE_CONFIG_FIELDS = {
    "data_path",
    "results_dir",
    "dataset_name",
    "target_col",
    "lookback",
    "horizon",
    "seed",
    "device",
    "split_stats",
    "time_order_info",
    "models",
    "export_all_window_predictions",
}
BENCHMARK_PROTOCOL_FIELDS = (
    "batch_size",
    "epochs",
    "patience",
    "lr",
    "weight_decay",
    "train_ratio",
    "val_ratio",
    "efficiency_error_metric",
    "deterministic",
    "stl_period",
    "decomposition_mode",
    "mstl_periods",
)


def build_config_signatures(run_dir: Path) -> tuple[str, str]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return "", ""
    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    config_signature_payload = {
        key: payload[key]
        for key in sorted(payload)
        if key not in VOLATILE_CONFIG_FIELDS
    }
    protocol_signature_payload = {
        key: payload.get(key)
        for key in BENCHMARK_PROTOCOL_FIELDS
        if key in payload
    }
    return (
        json.dumps(config_signature_payload, ensure_ascii=False, sort_keys=True),
        json.dumps(protocol_signature_payload, ensure_ascii=False, sort_keys=True),
    )


def load_metrics_frame(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {run_dir}")
    df = pd.read_csv(metrics_path)
    metadata = load_run_metadata(run_dir)
    for key in ("dataset_name", "target_col", "lookback", "horizon", "seed"):
        if key in metadata and str(metadata[key]).strip():
            df[key] = metadata[key]
    config_signature, benchmark_signature = build_config_signatures(run_dir)
    df["config_signature"] = config_signature
    df["benchmark_signature"] = benchmark_signature
    return df


def validate_group_consistency(frame: pd.DataFrame) -> None:
    group_keys = [column for column in GROUP_KEYS if column in frame.columns]
    inconsistent_configs = (
        frame.groupby(group_keys, dropna=False)["config_signature"].nunique(dropna=False).reset_index(name="count")
    )
    inconsistent_configs = inconsistent_configs[inconsistent_configs["count"] > 1]
    if not inconsistent_configs.empty:
        row = inconsistent_configs.iloc[0]
        raise ValueError(
            "Mixed config signatures found for one benchmark cell: "
            f"{ {key: row[key] for key in group_keys} }"
        )

    winner_keys = [column for column in WINNER_KEYS if column in frame.columns]
    inconsistent_protocols = (
        frame.groupby(winner_keys, dropna=False)["benchmark_signature"].nunique(dropna=False).reset_index(name="count")
    )
    inconsistent_protocols = inconsistent_protocols[inconsistent_protocols["count"] > 1]
    if not inconsistent_protocols.empty:
        row = inconsistent_protocols.iloc[0]
        raise ValueError(
            "Mixed benchmark protocol signatures found across models: "
            f"{ {key: row[key] for key in winner_keys} }"
        )


def validate_seed_coverage(frame: pd.DataFrame) -> None:
    if "seed" not in frame.columns:
        return
    winner_keys = [column for column in WINNER_KEYS if column in frame.columns]
    for winner_values, group in frame.groupby(winner_keys, dropna=False):
        per_model = (
            group.groupby("model", dropna=False)["seed"]
            .apply(lambda values: tuple(sorted(int(value) for value in pd.unique(values))))
        )
        unique_seed_sets = {seed_set for seed_set in per_model.tolist()}
        if len(unique_seed_sets) > 1:
            winner_lookup = dict(zip(winner_keys, winner_values if isinstance(winner_values, tuple) else (winner_values,)))
            raise ValueError(f"Inconsistent seed coverage across models for {winner_lookup}: {per_model.to_dict()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate benchmark runs across seeds.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--accuracy-tolerance", type=float, default=DEFAULT_ACCURACY_TOLERANCE)
    return parser


def choose_accuracy_metric(summary: pd.DataFrame) -> str:
    if "mean_test_mae_scaled" in summary.columns and summary["mean_test_mae_scaled"].notna().any():
        return "mean_test_mae_scaled"
    return "mean_test_mae"


def add_efficiency_analysis_columns(
    summary: pd.DataFrame,
    winner_keys: list[str],
    accuracy_metric: str,
    tolerance: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in summary.groupby(winner_keys, dropna=False, sort=False):
        frames.append(
            add_parameter_efficiency_columns(
                group,
                error_col=accuracy_metric,
                params_col="mean_params",
                tolerance=tolerance,
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(winner_keys + ["model"]).reset_index(drop=True)


def build_parameter_efficiency_winners(
    summary: pd.DataFrame,
    winner_keys: list[str],
    accuracy_metric: str,
    tolerance: float,
) -> pd.DataFrame:
    label = format_tolerance_label(tolerance)
    efficient_col = f"parameter_efficient_{label}"
    within_col = f"within_{label}_accuracy_tolerance"
    rows: list[dict[str, object]] = []
    for winner_values, group in summary.groupby(winner_keys, dropna=False, sort=False):
        key_values = winner_values if isinstance(winner_values, tuple) else (winner_values,)
        row: dict[str, object] = dict(zip(winner_keys, key_values))
        accuracy_row = group.sort_values([accuracy_metric, "mean_params", "model"], kind="mergesort").iloc[0]
        efficient_candidates = group.loc[group[efficient_col].astype(bool)]
        if efficient_candidates.empty:
            raise ValueError(f"No tolerance-efficient winner found for {row}.")
        efficient_row = efficient_candidates.sort_values(["mean_params", accuracy_metric, "model"], kind="mergesort").iloc[0]
        row.update(
            {
                "accuracy_winner": accuracy_row["model"],
                "accuracy_winner_metric": accuracy_metric,
                "accuracy_winner_error": float(accuracy_row[accuracy_metric]),
                "accuracy_winner_params": float(accuracy_row["mean_params"]),
                f"parameter_efficient_{label}_winner": efficient_row["model"],
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
    return pd.DataFrame(rows).sort_values(winner_keys).reset_index(drop=True)


def sanitize_filename_part(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text or "group"


def save_pareto_plot(
    group_df: pd.DataFrame,
    output_path: Path,
    *,
    accuracy_metric: str,
    tolerance: float,
) -> None:
    label = format_tolerance_label(tolerance)
    efficient_col = f"parameter_efficient_{label}"
    within_col = f"within_{label}_accuracy_tolerance"
    x_values = np.log10(group_df["mean_params"].astype(float).to_numpy())
    y_values = group_df[accuracy_metric].astype(float).to_numpy()
    pareto_mask = group_df["pareto_optimal"].astype(bool).to_numpy()
    efficient_mask = group_df[efficient_col].astype(bool).to_numpy()
    within_mask = group_df[within_col].astype(bool).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(x_values[~pareto_mask], y_values[~pareto_mask], c="#8d99ae", s=58, label="dominated")
    ax.scatter(x_values[pareto_mask], y_values[pareto_mask], c="#2a9d8f", s=70, label="Pareto-optimal")
    if within_mask.any():
        ax.scatter(
            x_values[within_mask],
            y_values[within_mask],
            facecolors="none",
            edgecolors="#355070",
            linewidths=1.5,
            s=118,
            label=f"within {tolerance:.0%}",
        )
    if efficient_mask.any():
        ax.scatter(
            x_values[efficient_mask],
            y_values[efficient_mask],
            marker="*",
            c="#c8553d",
            edgecolors="#5f0f00",
            linewidths=0.7,
            s=180,
            label=f"efficient {label}",
        )
    for idx, row in group_df.reset_index(drop=True).iterrows():
        ax.annotate(
            str(row["model"]),
            (float(x_values[idx]), float(y_values[idx])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    threshold = float(group_df["accuracy_tolerance_threshold"].iat[0])
    ax.axhline(threshold, color="#6d597a", linestyle="--", linewidth=1.0, alpha=0.75)
    y_label = "Mean scaled MAE" if accuracy_metric.endswith("_scaled") else "Mean test MAE"
    ax.set_xlabel("log10(trainable params)")
    ax.set_ylabel(y_label)
    ax.set_title("Pareto: Accuracy vs Parameters")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_pareto_outputs(
    summary: pd.DataFrame,
    results_dir: Path,
    winner_keys: list[str],
    accuracy_metric: str,
    tolerance: float,
) -> None:
    pareto_dir = results_dir / "pareto_by_group"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    for winner_values, group_df in summary.groupby(winner_keys, dropna=False, sort=False):
        key_values = winner_values if isinstance(winner_values, tuple) else (winner_values,)
        group_key = dict(zip(winner_keys, key_values))
        stem = "_".join(
            [
                sanitize_filename_part(group_key.get("dataset_name", "dataset")),
                f"lb{int(group_key.get('lookback', 0))}",
                f"h{int(group_key.get('horizon', 0))}",
            ]
        )
        save_pareto_plot(
            group_df.sort_values(["mean_params", accuracy_metric, "model"]).reset_index(drop=True),
            pareto_dir / f"{stem}.png",
            accuracy_metric=accuracy_metric,
            tolerance=tolerance,
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.accuracy_tolerance < 0:
        raise ValueError(f"--accuracy-tolerance must be non-negative, got {args.accuracy_tolerance}.")
    run_dirs = iter_run_dirs(args.results_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {args.results_dir}")

    frame = pd.concat([load_metrics_frame(run_dir) for run_dir in run_dirs], ignore_index=True)
    validate_group_consistency(frame)
    validate_seed_coverage(frame)

    group_keys = [column for column in GROUP_KEYS if column in frame.columns]
    if not group_keys:
        raise ValueError("No benchmark grouping columns were found.")

    aggregations: dict[str, tuple[str, str]] = {
        "seed_count": ("seed", "nunique"),
        "mean_test_mae": ("test_mae", "mean"),
        "std_test_mae": ("test_mae", "std"),
        "mean_test_rmse": ("test_rmse", "mean"),
        "std_test_rmse": ("test_rmse", "std"),
        "mean_val_mae": ("val_mae", "mean"),
        "std_val_mae": ("val_mae", "std"),
        "mean_val_rmse": ("val_rmse", "mean"),
        "std_val_rmse": ("val_rmse", "std"),
        "mean_params": ("params", "mean"),
    }
    optional_aggs = {
        "mean_test_mae_scaled": ("test_mae_scaled", "mean"),
        "std_test_mae_scaled": ("test_mae_scaled", "std"),
        "mean_test_rmse_scaled": ("test_rmse_scaled", "mean"),
        "std_test_rmse_scaled": ("test_rmse_scaled", "std"),
        "mean_val_mae_scaled": ("val_mae_scaled", "mean"),
        "std_val_mae_scaled": ("val_mae_scaled", "std"),
        "mean_val_rmse_scaled": ("val_rmse_scaled", "mean"),
        "std_val_rmse_scaled": ("val_rmse_scaled", "std"),
        "mean_epes": ("epes", "mean"),
        "std_epes": ("epes", "std"),
        "mean_cpls": ("cpls", "mean"),
        "std_cpls": ("cpls", "std"),
        "mean_latency_ms_per_sample": ("test_inference_ms_per_sample", "mean"),
        "std_latency_ms_per_sample": ("test_inference_ms_per_sample", "std"),
        "mean_peak_memory_mb": ("test_peak_memory_mb", "mean"),
        "std_peak_memory_mb": ("test_peak_memory_mb", "std"),
    }
    for output_name, (column_name, agg_fn) in optional_aggs.items():
        if column_name in frame.columns:
            aggregations[output_name] = (column_name, agg_fn)

    summary = (
        frame.groupby(group_keys, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    winner_keys = [column for column in WINNER_KEYS if column in summary.columns]
    mae_winner_metric = choose_accuracy_metric(summary)
    summary = add_efficiency_analysis_columns(
        summary,
        winner_keys=winner_keys,
        accuracy_metric=mae_winner_metric,
        tolerance=args.accuracy_tolerance,
    )
    parameter_efficiency_winners = build_parameter_efficiency_winners(
        summary,
        winner_keys=winner_keys,
        accuracy_metric=mae_winner_metric,
        tolerance=args.accuracy_tolerance,
    )
    mae_value_columns = [
        column
        for column in [
            "mean_test_mae",
            "mean_test_mae_scaled",
            "mean_test_rmse",
            "mean_test_rmse_scaled",
            "mean_params",
            "pareto_optimal",
            f"within_{format_tolerance_label(args.accuracy_tolerance)}_accuracy_tolerance",
        ]
        if column in summary.columns
    ]
    best_mae = summary.loc[
        summary.groupby(winner_keys)[mae_winner_metric].idxmin(),
        winner_keys + ["model"] + mae_value_columns,
    ].rename(columns={"model": "mean_test_mae_winner"})
    best_mae["mae_winner_metric"] = mae_winner_metric
    winners = best_mae.copy()
    parameter_efficiency_columns = [
        column
        for column in parameter_efficiency_winners.columns
        if column not in winner_keys + ["accuracy_winner", "accuracy_winner_metric", "accuracy_winner_error", "accuracy_winner_params"]
    ]
    winners = winners.merge(
        parameter_efficiency_winners[winner_keys + parameter_efficiency_columns],
        on=winner_keys,
        how="outer",
    )
    winners = winners.sort_values(winner_keys).reset_index(drop=True)

    summary_path = args.results_dir / "benchmark_seed_summary.csv"
    winners_path = args.results_dir / "benchmark_winners.csv"
    parameter_efficiency_path = args.results_dir / "benchmark_parameter_efficiency.csv"
    pareto_summary_path = args.results_dir / "benchmark_pareto_summary.csv"
    legacy_scores_path = args.results_dir / "benchmark_legacy_efficiency_scores.csv"
    summary.to_csv(summary_path, index=False)
    winners.to_csv(winners_path, index=False)
    parameter_efficiency_winners.to_csv(parameter_efficiency_path, index=False)
    pareto_columns = [
        column
        for column in [
            *winner_keys,
            "model",
            "mean_test_mae_scaled",
            "mean_test_mae",
            "mean_test_rmse",
            "mean_params",
            "pareto_optimal",
            "relative_error_vs_best",
            f"within_{format_tolerance_label(args.accuracy_tolerance)}_accuracy_tolerance",
            f"parameter_efficient_{format_tolerance_label(args.accuracy_tolerance)}",
        ]
        if column in summary.columns
    ]
    summary.loc[:, pareto_columns].to_csv(pareto_summary_path, index=False)
    if "mean_epes" in summary.columns:
        legacy_columns = [
            column
            for column in [*winner_keys, "model", "mean_epes", "std_epes", "mean_cpls", "std_cpls"]
            if column in summary.columns
        ]
        summary.loc[:, legacy_columns].to_csv(legacy_scores_path, index=False)
    write_pareto_outputs(
        summary,
        args.results_dir,
        winner_keys=winner_keys,
        accuracy_metric=mae_winner_metric,
        tolerance=args.accuracy_tolerance,
    )

    print(f"Saved benchmark seed summary to: {summary_path}")
    print(f"Saved benchmark winners to: {winners_path}")
    print(f"Saved parameter-efficiency winners to: {parameter_efficiency_path}")
    print(f"Saved Pareto summary to: {pareto_summary_path}")
    if "mean_epes" in summary.columns:
        print(f"Saved legacy efficiency scores to: {legacy_scores_path}")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
