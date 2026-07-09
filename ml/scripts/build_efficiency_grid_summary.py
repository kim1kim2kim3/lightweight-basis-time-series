from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from efficiency_metrics import (
    add_epes_cpls_columns,
    add_legacy_efficiency_columns,
    validate_efficiency_error_metric,
)

from results_layout import iter_run_dirs, load_run_metadata


def load_run_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_with_mse = run_dir / "metrics_with_mse.csv"
    metrics = run_dir / "metrics.csv"
    if metrics_with_mse.exists():
        df = pd.read_csv(metrics_with_mse)
    elif metrics.exists():
        df = pd.read_csv(metrics)
        df["test_mse"] = df["test_rmse"] ** 2
        df["val_mse"] = df["val_rmse"] ** 2
    else:
        raise FileNotFoundError(f"No metrics file found in {run_dir}")

    metadata = load_run_metadata(run_dir)
    df["lookback"] = int(metadata["lookback"])
    df["horizon"] = int(metadata["horizon"])
    if "seed" in metadata:
        df["seed"] = int(metadata["seed"])
    if "target_col" in metadata:
        df["target_col"] = str(metadata["target_col"])
    if "dataset_name" in metadata and str(metadata["dataset_name"]).strip():
        df["dataset_name"] = str(metadata["dataset_name"])

    df = add_legacy_efficiency_columns(df)
    if "efficiency_error_metric" in df.columns:
        normalized_metrics = {
            validate_efficiency_error_metric(str(value))
            for value in df["efficiency_error_metric"].dropna().tolist()
        }
        if len(normalized_metrics) != 1:
            raise ValueError(f"Inconsistent efficiency_error_metric values in {run_dir}: {sorted(normalized_metrics)}")
        efficiency_error_metric = next(iter(normalized_metrics))
    else:
        efficiency_error_metric = "mae"
    return add_epes_cpls_columns(df, efficiency_error_metric=efficiency_error_metric)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate efficiency metrics across Exchange sweep runs.")
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir
    run_dirs = iter_run_dirs(results_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")

    frames = [load_run_metrics(run_dir) for run_dir in run_dirs]
    summary_df = pd.concat(frames, ignore_index=True)
    group_keys = [column for column in ["dataset_name", "target_col", "seed", "lookback", "horizon"] if column in summary_df.columns]
    summary_df = summary_df.sort_values(group_keys + ["model"]).reset_index(drop=True)

    mae_winners = summary_df.loc[
        summary_df.groupby(group_keys)["mae_efficiency_vs_best"].idxmax(),
        group_keys + ["model", "mae_efficiency_vs_best", "test_mae", "params"],
    ].rename(columns={"model": "mae_efficiency_winner"})

    mse_winners = summary_df.loc[
        summary_df.groupby(group_keys)["mse_efficiency_vs_best"].idxmax(),
        group_keys + ["model", "mse_efficiency_vs_best", "test_mse", "params"],
    ].rename(columns={"model": "mse_efficiency_winner"})

    epes_winners = summary_df.loc[
        summary_df.groupby(group_keys)["epes"].idxmax(),
        group_keys + ["model", "efficiency_error_metric", "epes", "efficiency_error_value", "params"],
    ].rename(columns={"model": "epes_winner"})

    cpls_winners = summary_df.loc[
        summary_df.groupby(group_keys)["cpls"].idxmax(),
        group_keys + ["model", "cpls", "cpls_wins", "cpls_losses", "cpls_ties"],
    ].rename(columns={"model": "cpls_winner"})

    winner_df = mae_winners.merge(mse_winners, on=group_keys, how="outer")
    winner_df = winner_df.merge(epes_winners, on=group_keys, how="outer")
    winner_df = winner_df.merge(cpls_winners, on=group_keys, how="outer")
    winner_df = winner_df.sort_values(group_keys).reset_index(drop=True)

    summary_path = results_dir / "efficiency_summary.csv"
    winners_path = results_dir / "efficiency_winners.csv"
    summary_df.to_csv(summary_path, index=False)
    winner_df.to_csv(winners_path, index=False)

    print(f"Saved efficiency summary to: {summary_path}")
    print(f"Saved efficiency winners to: {winners_path}")
    print(winner_df.to_string(index=False))


if __name__ == "__main__":
    main()
