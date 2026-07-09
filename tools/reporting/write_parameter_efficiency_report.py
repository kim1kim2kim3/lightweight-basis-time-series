from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "runs" / "etth1_ettm2_ours_dlinear_patchtst_tslib_3seed_all_horizons"
OUT_DIR = ROOT / "reports" / "paper"


def format_float(value: float, precision: int = 4) -> str:
    return f"{float(value):.{precision}f}"


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def build_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "target_col",
        "lookback",
        "horizon",
        "model",
        "mean_test_mae_scaled",
        "mean_test_mae",
        "mean_test_rmse",
        "mean_params",
        "pareto_optimal",
        "within_2pct_accuracy_tolerance",
        "parameter_efficient_2pct",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary.loc[:, available].sort_values(["dataset_name", "horizon", "model"]).reset_index(drop=True)


def build_winner_table(winners: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "target_col",
        "lookback",
        "horizon",
        "accuracy_winner",
        "accuracy_winner_metric",
        "accuracy_winner_error",
        "accuracy_winner_params",
        "parameter_efficient_2pct_winner",
        "parameter_efficient_error",
        "parameter_efficient_params",
        "parameter_efficient_relative_error_vs_best",
        "accuracy_tolerance_threshold",
    ]
    available = [column for column in columns if column in winners.columns]
    return winners.loc[:, available].sort_values(["dataset_name", "horizon"]).reset_index(drop=True)


def build_markdown(winner_table: pd.DataFrame, results_dir: Path) -> str:
    rows: list[list[str]] = []
    for _, row in winner_table.iterrows():
        rows.append(
            [
                str(row["dataset_name"]),
                str(int(row["horizon"])),
                str(row["accuracy_winner"]),
                format_float(float(row["accuracy_winner_error"])),
                str(row["parameter_efficient_2pct_winner"]),
                format_float(float(row["parameter_efficient_error"])),
                format_float(float(row["parameter_efficient_relative_error_vs_best"]) * 100.0, precision=2) + "%",
                format_float(float(row["parameter_efficient_params"]), precision=0),
            ]
        )

    return "\n".join(
        [
            "# Parameter Efficiency Analysis",
            "",
            f"Source: `{results_dir.relative_to(ROOT)}`",
            "",
            "Method:",
            "- Accuracy winner: lowest `mean_test_mae_scaled`.",
            "- Parameter-efficient winner: smallest parameter count among models within 2% of the best scaled MAE.",
            "- Pareto analysis uses `x = log10(params)` and `y = scaled MAE`.",
            "",
            render_markdown_table(
                [
                    "Dataset",
                    "Horizon",
                    "Accuracy winner",
                    "Best scaled MAE",
                    "Efficient @ 2%",
                    "Efficient scaled MAE",
                    "Gap vs best",
                    "Efficient params",
                ],
                rows,
            ),
            "",
            "Paper-safe claims:",
            "- PatchTST is the scaled-MAE accuracy winner on ETTh1 across the reported horizons.",
            "- DLinear is the scaled-MAE accuracy winner on ETTm2 across the reported horizons.",
            "- Under the 2% scaled-MAE tolerance, Ours is parameter-efficient on ETTh1 horizons 336 and 720.",
            "- Under the same tolerance, DLinear remains parameter-efficient on all ETTm2 horizons.",
            "- Parameter efficiency should be reported through Pareto and tolerance analysis, not as a single universal composite score.",
            "",
        ]
    )


def main() -> None:
    results_dir = DEFAULT_RESULTS_DIR
    summary_path = results_dir / "benchmark_seed_summary.csv"
    winners_path = results_dir / "benchmark_parameter_efficiency.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")
    if not winners_path.exists():
        raise FileNotFoundError(f"Missing parameter-efficiency winners: {winners_path}")

    summary = pd.read_csv(summary_path)
    winners = pd.read_csv(winners_path)
    main_table = build_main_table(summary)
    winner_table = build_winner_table(winners)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(OUT_DIR / "benchmark_main_table.csv", index=False)
    winner_table.to_csv(OUT_DIR / "parameter_efficiency_winners.csv", index=False)
    (OUT_DIR / "parameter_efficiency_analysis.md").write_text(
        build_markdown(winner_table, results_dir),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'benchmark_main_table.csv'}")
    print(f"Wrote {OUT_DIR / 'parameter_efficiency_winners.csv'}")
    print(f"Wrote {OUT_DIR / 'parameter_efficiency_analysis.md'}")


if __name__ == "__main__":
    main()
