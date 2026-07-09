from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from efficiency_metrics import (
    EFFICIENCY_ERROR_METRICS,
    add_epes_cpls_columns,
    add_legacy_efficiency_columns,
    build_pairwise_cpl_dataframe,
)

MODEL_ORDER = [
    "stl_tcn",
    "dlinear",
    "tcn",
    "gru",
    "lstm_pure",
    "patchtst",
    "lstm_proposed",
]


def build_related_output_path(output_path: Path, label: str) -> Path:
    stem = output_path.stem
    if "summary" in stem:
        derived_stem = stem.replace("summary", label)
    else:
        derived_stem = f"{stem}_{label}"
    return output_path.with_name(f"{derived_stem}{output_path.suffix}")


def annotate_bars(ax: plt.Axes, fmt: str) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
        )


def save_mae_mse_plot(df: pd.DataFrame, output_path: Path, colors: list[str]) -> None:
    labels = df["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(labels, df["test_mae"], color=colors)
    axes[0].set_title("Test MAE")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=20)
    annotate_bars(axes[0], "{:.4f}")

    axes[1].bar(labels, df["test_mse"], color=colors)
    axes[1].set_title("Test MSE")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=20)
    annotate_bars(axes[1], "{:.5f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_parameter_plot(df: pd.DataFrame, output_path: Path, colors: list[str]) -> None:
    labels = df["model"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.bar(labels, df["params"], color=colors)
    ax.set_title("Parameter Count")
    ax.set_ylabel("Trainable params")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax, "{:.0f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_epes_plot(df: pd.DataFrame, output_path: Path, colors: list[str], efficiency_error_metric: str) -> None:
    labels = df["model"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.bar(labels, df["epes"], color=colors)
    ax.set_title(f"EPES ({efficiency_error_metric.upper()}-based)")
    ax.set_ylabel("Score (higher is better)")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax, "{:.3f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_cpls_plot(df: pd.DataFrame, output_path: Path, colors: list[str]) -> None:
    labels = df["model"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.bar(labels, df["cpls"], color=colors)
    ax.set_title("CPLS")
    ax.set_ylabel("Pairwise parameter-efficiency score")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax, "{:.3f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_latency_memory_plot(df: pd.DataFrame, output_path: Path, colors: list[str]) -> None:
    labels = df["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(labels, df["test_inference_ms_per_sample"], color=colors)
    axes[0].set_title("Inference Latency")
    axes[0].set_ylabel("ms / sample")
    axes[0].tick_params(axis="x", rotation=20)
    annotate_bars(axes[0], "{:.3f}")

    axes[1].bar(labels, df["test_peak_memory_mb"], color=colors)
    axes[1].set_title("Peak Memory")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=20)
    annotate_bars(axes[1], "{:.2f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    df.loc[:, ["model", "test_mae", "test_mse"]].to_csv(output_dir / "mae_mse_comparison.csv", index=False)
    df.loc[:, ["model", "params"]].to_csv(output_dir / "parameter_comparison.csv", index=False)
    df.loc[:, ["model", "epes", "efficiency_error_metric"]].to_csv(output_dir / "epes_comparison.csv", index=False)
    df.loc[:, ["model", "cpls", "cpls_wins", "cpls_losses", "cpls_ties", "efficiency_error_metric"]].to_csv(
        output_dir / "cpls_comparison.csv",
        index=False,
    )
    if {"test_inference_ms_per_batch", "test_inference_ms_per_sample", "test_peak_memory_mb", "device_type"}.issubset(df.columns):
        df.loc[:, ["model", "test_inference_ms_per_batch", "test_inference_ms_per_sample", "test_peak_memory_mb", "device_type"]].to_csv(
            output_dir / "latency_memory_comparison.csv",
            index=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot parameter/error summary for an Exchange run.")
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--efficiency-error-metric", type=str, choices=EFFICIENCY_ERROR_METRICS, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_path = args.metrics_path
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    df = add_legacy_efficiency_columns(df)
    efficiency_error_metric = (
        args.efficiency_error_metric
        or (str(df["efficiency_error_metric"].iat[0]) if "efficiency_error_metric" in df.columns else "mae")
    )
    df = add_epes_cpls_columns(df, efficiency_error_metric=efficiency_error_metric)
    pairwise_cpl_df = build_pairwise_cpl_dataframe(df, efficiency_error_metric=efficiency_error_metric)

    output_path = args.output_path
    if output_path is None:
        output_dir = metrics_path.parent / "plots"
        output_path = output_dir / "mae_mse_comparison.png"
        parameter_output_path = output_dir / "parameter_comparison.png"
        epes_output_path = output_dir / "epes_comparison.png"
        cpls_output_path = output_dir / "cpls_comparison.png"
        latency_output_path = output_dir / "latency_memory_comparison.png"
    else:
        parameter_output_path = build_related_output_path(output_path, "parameter_comparison")
        epes_output_path = build_related_output_path(output_path, "epes_comparison")
        cpls_output_path = build_related_output_path(output_path, "cpls_comparison")
        latency_output_path = build_related_output_path(output_path, "latency_memory_comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    palette = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#457b9d", "#2a9d8f"]
    colors = palette[: len(df)]

    save_mae_mse_plot(df, output_path, colors)
    save_parameter_plot(df, parameter_output_path, colors)
    save_epes_plot(df, epes_output_path, colors, efficiency_error_metric)
    save_cpls_plot(df, cpls_output_path, colors)
    if {"test_inference_ms_per_sample", "test_peak_memory_mb"}.issubset(df.columns):
        save_latency_memory_plot(df, latency_output_path, colors)

    summary_path = metrics_path.parent / "metrics_with_mse.csv"
    pairwise_cpl_path = metrics_path.parent / "pairwise_cpl.csv"
    df.to_csv(summary_path, index=False)
    pairwise_cpl_df.to_csv(pairwise_cpl_path, index=False)
    write_summary_tables(df, metrics_path.parent)

    print(f"Saved MAE/MSE plot to: {output_path}")
    print(f"Saved parameter plot to: {parameter_output_path}")
    print(f"Saved EPES plot to: {epes_output_path}")
    print(f"Saved CPLS plot to: {cpls_output_path}")
    if {"test_inference_ms_per_sample", "test_peak_memory_mb"}.issubset(df.columns):
        print(f"Saved latency/memory plot to: {latency_output_path}")
    print(f"Saved metrics with MSE to: {summary_path}")
    print(f"Saved pairwise CPL table to: {pairwise_cpl_path}")


if __name__ == "__main__":
    main()
