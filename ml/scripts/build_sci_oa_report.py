from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a paper-ready SCI OA benchmark report.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--accuracy-tolerance", type=float, default=DEFAULT_ACCURACY_TOLERANCE)
    return parser


def has_usable_column(frame: pd.DataFrame, column_name: str) -> bool:
    return column_name in frame.columns and frame[column_name].notna().any()


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


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
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
        "mean_latency_ms_per_sample": ("test_inference_ms_per_sample", "mean"),
        "std_latency_ms_per_sample": ("test_inference_ms_per_sample", "std"),
        "mean_peak_memory_mb": ("test_peak_memory_mb", "mean"),
        "std_peak_memory_mb": ("test_peak_memory_mb", "std"),
    }
    for output_name, (column_name, agg_fn) in optional_aggs.items():
        if has_usable_column(frame, column_name):
            aggregations[output_name] = (column_name, agg_fn)

    return (
        frame.groupby(group_keys, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(group_keys)
        .reset_index(drop=True)
    )


def choose_accuracy_metric(summary: pd.DataFrame) -> str:
    if "mean_test_mae_scaled" in summary.columns and summary["mean_test_mae_scaled"].notna().any():
        return "mean_test_mae_scaled"
    return "mean_test_mae"


def add_efficiency_analysis_columns(
    summary: pd.DataFrame,
    accuracy_metric: str,
    tolerance: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in summary.groupby(WINNER_KEYS, dropna=False, sort=False):
        frames.append(
            add_parameter_efficiency_columns(
                group,
                error_col=accuracy_metric,
                params_col="mean_params",
                tolerance=tolerance,
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(GROUP_KEYS).reset_index(drop=True)


def select_winner_rows(
    summary: pd.DataFrame,
    metric_column: str,
    output_column: str,
    *,
    lower_is_better: bool = True,
    allow_missing: bool = False,
) -> pd.DataFrame:
    if metric_column not in summary.columns:
        if allow_missing:
            return pd.DataFrame(columns=WINNER_KEYS + [output_column])
        raise ValueError(f"Missing metric column: {metric_column}")

    candidate_columns = WINNER_KEYS + ["model", metric_column]
    tie_break_columns = ["model"]
    if metric_column != "mean_test_mae":
        candidate_columns.append("mean_test_mae")
        tie_break_columns.insert(0, "mean_test_mae")
    candidate_frame = summary.loc[summary[metric_column].notna(), candidate_columns]
    if candidate_frame.empty:
        return pd.DataFrame(columns=WINNER_KEYS + [output_column])

    sort_columns = WINNER_KEYS + [metric_column] + tie_break_columns
    ascending = [True] * len(WINNER_KEYS) + [lower_is_better] + [True] * len(tie_break_columns)
    winners = (
        candidate_frame.sort_values(sort_columns, ascending=ascending, na_position="last")
        .groupby(WINNER_KEYS, as_index=False)
        .first()
        .loc[:, WINNER_KEYS + ["model"]]
        .rename(columns={"model": output_column})
    )
    return winners


def build_winner_summary(summary: pd.DataFrame, accuracy_metric: str, tolerance: float) -> pd.DataFrame:
    winners = select_winner_rows(summary, accuracy_metric, "best_mae_model")
    winners["accuracy_winner_metric"] = accuracy_metric
    winners = winners.merge(
        select_winner_rows(summary, "mean_test_rmse", "best_rmse_model"),
        on=WINNER_KEYS,
        how="outer",
    )
    winners = winners.merge(
        select_winner_rows(summary, "mean_params", "smallest_params_model"),
        on=WINNER_KEYS,
        how="outer",
    )
    latency_winners = select_winner_rows(
        summary,
        "mean_latency_ms_per_sample",
        "lowest_latency_model",
        allow_missing=True,
    )
    if not latency_winners.empty:
        winners = winners.merge(latency_winners, on=WINNER_KEYS, how="outer")

    memory_winners = select_winner_rows(
        summary,
        "mean_peak_memory_mb",
        "lowest_memory_model",
        allow_missing=True,
    )
    if not memory_winners.empty:
        winners = winners.merge(memory_winners, on=WINNER_KEYS, how="outer")
    label = format_tolerance_label(tolerance)
    efficient_col = f"parameter_efficient_{label}"
    efficient_rows = (
        summary.loc[summary[efficient_col].astype(bool), WINNER_KEYS + ["model", accuracy_metric, "mean_params"]]
        .rename(
            columns={
                "model": f"parameter_efficient_{label}_model",
                accuracy_metric: "parameter_efficient_error",
                "mean_params": "parameter_efficient_params",
            }
        )
        .sort_values(WINNER_KEYS)
        .reset_index(drop=True)
    )
    if not efficient_rows.empty:
        efficient_rows["parameter_efficiency_rule"] = (
            f"smallest params within {tolerance:.2%} of best {accuracy_metric}"
        )
        winners = winners.merge(efficient_rows, on=WINNER_KEYS, how="outer")
    return winners.sort_values(WINNER_KEYS).reset_index(drop=True)


def sanitize_filename_part(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text or "group"


def format_mean_std(mean_value: float, std_value: float, precision: int = 4) -> str:
    mean_text = f"{mean_value:.{precision}f}"
    if pd.isna(std_value):
        return mean_text
    return f"{mean_text} +- {std_value:.{precision}f}"


def format_scalar(value: float, precision: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if precision == 0:
        return str(int(round(float(value))))
    return f"{float(value):.{precision}f}"


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def group_display_columns(group_df: pd.DataFrame) -> list[str]:
    columns = [
        "model",
        "seed_count",
        "mean_test_mae_scaled",
        "std_test_mae_scaled",
        "mean_test_mae",
        "std_test_mae",
        "mean_test_rmse",
        "std_test_rmse",
        "mean_val_mae",
        "std_val_mae",
        "mean_val_rmse",
        "std_val_rmse",
        "mean_params",
        "pareto_optimal",
        "relative_error_vs_best",
        "within_2pct_accuracy_tolerance",
        "parameter_efficient_2pct",
    ]
    if has_usable_column(group_df, "mean_latency_ms_per_sample"):
        columns.extend(["mean_latency_ms_per_sample", "std_latency_ms_per_sample"])
    if has_usable_column(group_df, "mean_peak_memory_mb"):
        columns.extend(["mean_peak_memory_mb", "std_peak_memory_mb"])
    return [column for column in columns if column in group_df.columns]


def build_table_rows(group_df: pd.DataFrame, winners_row: pd.Series) -> tuple[list[str], list[list[str]]]:
    include_scaled_mae = has_usable_column(group_df, "mean_test_mae_scaled")
    headers = ["Model"]
    if include_scaled_mae:
        headers.append("Scaled MAE")
    headers.extend(["Test MAE", "Test RMSE", "Params"])
    include_latency = has_usable_column(group_df, "mean_latency_ms_per_sample")
    include_memory = has_usable_column(group_df, "mean_peak_memory_mb")
    if include_latency:
        headers.append("Latency (ms/sample)")
    if include_memory:
        headers.append("Peak Memory (MB)")

    sort_metric = "mean_test_mae_scaled" if include_scaled_mae else "mean_test_mae"
    sorted_group = group_df.sort_values([sort_metric, "mean_test_rmse", "model"]).reset_index(drop=True)
    rows: list[list[str]] = []
    for _, row in sorted_group.iterrows():
        scaled_mae_cell = (
            format_mean_std(float(row["mean_test_mae_scaled"]), float(row["std_test_mae_scaled"]), precision=4)
            if include_scaled_mae
            else ""
        )
        mae_cell = format_mean_std(float(row["mean_test_mae"]), float(row["std_test_mae"]), precision=4)
        rmse_cell = format_mean_std(float(row["mean_test_rmse"]), float(row["std_test_rmse"]), precision=4)
        if include_scaled_mae and row["model"] == winners_row["best_mae_model"]:
            scaled_mae_cell = f"**{scaled_mae_cell}**"
        elif row["model"] == winners_row["best_mae_model"]:
            mae_cell = f"**{mae_cell}**"
        if row["model"] == winners_row["best_rmse_model"]:
            rmse_cell = f"**{rmse_cell}**"

        table_row = [str(row["model"])]
        if include_scaled_mae:
            table_row.append(scaled_mae_cell)
        table_row.extend([mae_cell, rmse_cell, format_scalar(float(row["mean_params"]), precision=0)])
        if include_latency:
            table_row.append(
                format_mean_std(
                    float(row["mean_latency_ms_per_sample"]),
                    float(row["std_latency_ms_per_sample"]),
                    precision=4,
                )
            )
        if include_memory:
            table_row.append(
                format_mean_std(
                    float(row["mean_peak_memory_mb"]),
                    float(row["std_peak_memory_mb"]),
                    precision=2,
                )
            )
        rows.append(table_row)
    return headers, rows


def build_winners_line(winners_row: pd.Series) -> str:
    accuracy_label = "Scaled MAE" if winners_row.get("accuracy_winner_metric") == "mean_test_mae_scaled" else "MAE"
    parts = [
        f"{accuracy_label}=`{winners_row['best_mae_model']}`",
        f"RMSE=`{winners_row['best_rmse_model']}`",
        f"Params=`{winners_row['smallest_params_model']}`",
    ]
    efficient_columns = [column for column in winners_row.index if column.startswith("parameter_efficient_") and column.endswith("_model")]
    if efficient_columns:
        efficient_column = efficient_columns[0]
        if pd.notna(winners_row[efficient_column]):
            parts.append(f"Efficient@2%=`{winners_row[efficient_column]}`")
    if "lowest_latency_model" in winners_row and pd.notna(winners_row["lowest_latency_model"]):
        parts.append(f"Latency=`{winners_row['lowest_latency_model']}`")
    if "lowest_memory_model" in winners_row and pd.notna(winners_row["lowest_memory_model"]):
        parts.append(f"Memory=`{winners_row['lowest_memory_model']}`")
    return "Winners: " + ", ".join(parts)


def write_group_outputs(summary: pd.DataFrame, winners: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for winner_values, winners_group in winners.groupby(WINNER_KEYS, dropna=False):
        winner_lookup = dict(zip(WINNER_KEYS, winner_values))
        mask = pd.Series(True, index=summary.index)
        for key, value in winner_lookup.items():
            mask &= summary[key] == value
        group_df = summary.loc[mask].copy()
        if group_df.empty:
            continue

        winners_row = winners_group.iloc[0]
        dataset_slug = sanitize_filename_part(winner_lookup["dataset_name"])
        stem = f"{dataset_slug}_lb{int(winner_lookup['lookback'])}_h{int(winner_lookup['horizon'])}"

        group_df.loc[:, group_display_columns(group_df)].to_csv(tables_dir / f"{stem}.csv", index=False)
        headers, rows = build_table_rows(group_df, winners_row)
        title = (
            f"## {winner_lookup['dataset_name']} | target `{winner_lookup['target_col']}` | "
            f"lookback {winner_lookup['lookback']} | horizon {winner_lookup['horizon']}"
        )
        markdown = "\n\n".join(
            [
                title,
                render_markdown_table(headers, rows),
                build_winners_line(winners_row),
            ]
        )
        (tables_dir / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")


def build_coverage_lines(summary: pd.DataFrame) -> list[str]:
    datasets = ", ".join(str(value) for value in sorted(summary["dataset_name"].dropna().unique()))
    horizons = ", ".join(str(int(value)) for value in sorted(summary["horizon"].dropna().unique()))
    models = ", ".join(str(value) for value in sorted(summary["model"].dropna().unique()))
    seed_min = int(summary["seed_count"].min())
    seed_max = int(summary["seed_count"].max())
    seed_text = str(seed_min) if seed_min == seed_max else f"{seed_min}-{seed_max}"
    return [
        "# SCI OA Benchmark Report",
        "",
        f"- Datasets: {datasets}",
        f"- Horizons: {horizons}",
        f"- Models: {models}",
        f"- Seeds per group: {seed_text}",
        "",
    ]


def render_group_markdown(summary: pd.DataFrame, winners: pd.DataFrame) -> list[str]:
    lines: list[str] = ["## Per Dataset-Horizon Tables", ""]
    for winner_values, winners_group in winners.groupby(WINNER_KEYS, dropna=False):
        winner_lookup = dict(zip(WINNER_KEYS, winner_values))
        mask = pd.Series(True, index=summary.index)
        for key, value in winner_lookup.items():
            mask &= summary[key] == value
        group_df = summary.loc[mask].copy()
        if group_df.empty:
            continue

        headers, rows = build_table_rows(group_df, winners_group.iloc[0])
        lines.append(
            f"### {winner_lookup['dataset_name']} | target `{winner_lookup['target_col']}` | "
            f"lookback {winner_lookup['lookback']} | horizon {winner_lookup['horizon']}"
        )
        lines.append("")
        lines.append(render_markdown_table(headers, rows))
        lines.append("")
        lines.append(build_winners_line(winners_group.iloc[0]))
        lines.append("")
    return lines


def render_winner_summary_markdown(winners: pd.DataFrame) -> list[str]:
    headers = ["Dataset", "Target", "Lookback", "Horizon", "Accuracy Metric", "Best MAE", "Best RMSE", "Smallest Params"]
    efficient_columns = [column for column in winners.columns if column.startswith("parameter_efficient_") and column.endswith("_model")]
    if efficient_columns:
        headers.append("Efficient @ 2%")
    include_latency = "lowest_latency_model" in winners.columns and winners["lowest_latency_model"].notna().any()
    include_memory = "lowest_memory_model" in winners.columns and winners["lowest_memory_model"].notna().any()
    if include_latency:
        headers.append("Lowest Latency")
    if include_memory:
        headers.append("Lowest Memory")

    rows: list[list[str]] = []
    for _, row in winners.sort_values(WINNER_KEYS).iterrows():
        row_values = [
            str(row["dataset_name"]),
            str(row["target_col"]),
            str(int(row["lookback"])),
            str(int(row["horizon"])),
            str(row.get("accuracy_winner_metric", "mean_test_mae")),
            str(row["best_mae_model"]),
            str(row["best_rmse_model"]),
            str(row["smallest_params_model"]),
        ]
        if efficient_columns:
            efficient_column = efficient_columns[0]
            row_values.append("NA" if pd.isna(row[efficient_column]) else str(row[efficient_column]))
        if include_latency:
            row_values.append("NA" if pd.isna(row["lowest_latency_model"]) else str(row["lowest_latency_model"]))
        if include_memory:
            row_values.append("NA" if pd.isna(row["lowest_memory_model"]) else str(row["lowest_memory_model"]))
        rows.append(row_values)

    return [
        "## Winner Summary",
        "",
        render_markdown_table(headers, rows),
        "",
    ]


def write_markdown_report(summary: pd.DataFrame, winners: pd.DataFrame, output_path: Path) -> None:
    lines = build_coverage_lines(summary)
    lines.extend(render_group_markdown(summary, winners))
    lines.extend(render_winner_summary_markdown(winners))
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.accuracy_tolerance < 0:
        raise ValueError(f"--accuracy-tolerance must be non-negative, got {args.accuracy_tolerance}.")
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "sci_oa_report")).resolve()

    run_dirs = iter_run_dirs(results_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")

    frame = pd.concat([load_metrics_frame(run_dir) for run_dir in run_dirs], ignore_index=True)
    validate_group_consistency(frame)
    validate_seed_coverage(frame)
    summary = aggregate_metrics(frame)
    accuracy_metric = choose_accuracy_metric(summary)
    summary = add_efficiency_analysis_columns(summary, accuracy_metric, args.accuracy_tolerance)
    winners = build_winner_summary(summary, accuracy_metric, args.accuracy_tolerance)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    winners.to_csv(output_dir / "winner_summary.csv", index=False)
    write_group_outputs(summary, winners, output_dir)
    write_markdown_report(summary, winners, output_dir / "paper_summary.md")

    print(f"Saved aggregate metrics to: {output_dir / 'aggregate_metrics.csv'}")
    print(f"Saved winner summary to: {output_dir / 'winner_summary.csv'}")
    print(f"Saved paper summary to: {output_dir / 'paper_summary.md'}")


if __name__ == "__main__":
    main()
