from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "runs" / "etth1_ettm2_ours_dlinear_patchtst_tslib_3seed_all_horizons"
SUMMARY_PATH = RESULTS_DIR / "benchmark_seed_summary.csv"
WINNERS_PATH = RESULTS_DIR / "benchmark_parameter_efficiency.csv"
OUT = ROOT / "reports" / "paper" / "etth1_ours_dlinear_summary.md"


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def fmt(value: float, precision: int = 4) -> str:
    return f"{float(value):.{precision}f}"


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {SUMMARY_PATH}")
    if not WINNERS_PATH.exists():
        raise FileNotFoundError(f"Missing efficiency winners: {WINNERS_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)
    winners = pd.read_csv(WINNERS_PATH)
    etth1 = summary.loc[summary["dataset_name"] == "ETTh1"].copy()
    etth1_winners = winners.loc[winners["dataset_name"] == "ETTh1"].copy()
    model_order = ["dlinear", "patchtst", "ours"]
    etth1["model"] = pd.Categorical(etth1["model"], categories=model_order, ordered=True)
    etth1 = etth1.sort_values(["horizon", "model"]).reset_index(drop=True)

    metric_rows: list[list[str]] = []
    for _, row in etth1.iterrows():
        metric_rows.append(
            [
                str(int(row["horizon"])),
                str(row["model"]),
                fmt(row["mean_test_mae_scaled"]),
                fmt(row["std_test_mae_scaled"]),
                fmt(row["mean_test_rmse_scaled"]),
                fmt(row["std_test_rmse_scaled"]),
                fmt(row["mean_test_mae"]),
                fmt(row["mean_test_rmse"]),
                fmt(row["mean_params"], precision=0),
                str(bool(row["pareto_optimal"])),
                str(bool(row["within_2pct_accuracy_tolerance"])),
                str(bool(row["parameter_efficient_2pct"])),
            ]
        )

    winner_rows: list[list[str]] = []
    for _, row in etth1_winners.sort_values("horizon").iterrows():
        winner_rows.append(
            [
                str(int(row["horizon"])),
                str(row["accuracy_winner"]),
                fmt(row["accuracy_winner_error"]),
                str(row["parameter_efficient_2pct_winner"]),
                fmt(row["parameter_efficient_error"]),
                fmt(float(row["parameter_efficient_relative_error_vs_best"]) * 100.0, precision=2) + "%",
                fmt(row["parameter_efficient_params"], precision=0),
            ]
        )

    parts = [
        "# ETTh1 Current Benchmark Summary",
        "",
        f"Source: `{RESULTS_DIR.relative_to(ROOT)}`",
        "",
        "This file is generated from the current 3-seed benchmark summary. Accuracy winner is selected by lowest scaled MAE. Parameter-efficient winner uses the 2% scaled-MAE tolerance rule.",
        "",
        "## Metrics",
        "",
        render_markdown_table(
            [
                "Horizon",
                "Model",
                "Scaled MAE",
                "Scaled MAE std",
                "Scaled RMSE",
                "Scaled RMSE std",
                "MAE",
                "RMSE",
                "Params",
                "Pareto",
                "Within 2%",
                "Efficient @ 2%",
            ],
            metric_rows,
        ),
        "",
        "## Winners",
        "",
        render_markdown_table(
            [
                "Horizon",
                "Accuracy winner",
                "Best scaled MAE",
                "Efficient @ 2%",
                "Efficient scaled MAE",
                "Gap vs best",
                "Efficient params",
            ],
            winner_rows,
        ),
        "",
        "Summary:",
        "- PatchTST is the scaled-MAE accuracy winner on ETTh1.",
        "- Ours is the parameter-efficient winner under the 2% rule at horizons 336 and 720.",
        "- Ours keeps 11,149 trainable parameters for all ETTh1 horizons.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
