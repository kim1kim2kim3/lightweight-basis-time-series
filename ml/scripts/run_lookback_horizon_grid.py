from __future__ import annotations

import argparse
from pathlib import Path

from efficiency_metrics import EFFICIENCY_ERROR_METRICS
from train_exchange_once import DEFAULT_BENCHMARK_MODELS, ExperimentConfig, MODEL_ORDER, parse_models, run_experiment


def parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not horizons:
        raise ValueError("At least one horizon must be provided.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be positive integers.")
    return horizons


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_data = script_dir.parent / "dataset" / "ETTh1.csv"
    default_results = project_root / "runs" / "exchange_grid"

    parser = argparse.ArgumentParser(description="Run Exchange experiments for a lookback / horizon grid.")
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--target-col", type=str, default="OT")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--lookback", type=int, default=70)
    parser.add_argument("--horizons", type=str, default="7,14,21,28")
    parser.add_argument("--stl-period", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--efficiency-error-metric", type=str, choices=EFFICIENCY_ERROR_METRICS, default="mae")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--export-all-window-predictions", action="store_true")
    parser.add_argument("--lstm-proposed-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-pure-hidden-dim", type=int, default=64)
    parser.add_argument("--stl-hidden-dim", type=int, default=32)
    parser.add_argument("--dlinear-moving-avg-kernel", type=int, default=25)
    parser.add_argument("--dlinear-individual", action="store_true")
    parser.add_argument("--stl-disable-trend-branch", action="store_true")
    parser.add_argument("--stl-disable-season-branch", action="store_true")
    parser.add_argument("--stl-disable-resid-branch", action="store_true")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    horizons = parse_horizons(args.horizons)

    for horizon in horizons:
        config = ExperimentConfig(
            data_path=str(args.data_path),
            results_dir=str(args.results_dir),
            target_col=args.target_col,
            dataset_name=args.dataset_name,
            lookback=args.lookback,
            horizon=horizon,
            stl_period=args.stl_period,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            efficiency_error_metric=args.efficiency_error_metric,
            deterministic=args.deterministic,
            export_all_window_predictions=args.export_all_window_predictions,
            lstm_proposed_hidden_dim=args.lstm_proposed_hidden_dim,
            lstm_pure_hidden_dim=args.lstm_pure_hidden_dim,
            stl_hidden_dim=args.stl_hidden_dim,
            dlinear_moving_avg_kernel=args.dlinear_moving_avg_kernel,
            dlinear_individual=args.dlinear_individual,
            stl_use_trend_branch=not args.stl_disable_trend_branch,
            stl_use_season_branch=not args.stl_disable_season_branch,
            stl_use_resid_branch=not args.stl_disable_resid_branch,
            models=parse_models(args.models),
        )
        run_experiment(config)


if __name__ == "__main__":
    main()
