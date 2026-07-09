import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = (ROOT / "scripts").resolve()
sys.path.append(str(SCRIPTS_DIR))

from build_efficiency_grid_summary import load_run_metrics
from efficiency_metrics import (
    add_epes_cpls_columns,
    add_legacy_efficiency_columns,
    add_parameter_efficiency_columns,
    build_pairwise_cpl_dataframe,
    select_tolerance_efficient_row,
)


def make_metrics_frame(
    test_mae: list[float],
    test_rmse: list[float],
    params: list[int],
) -> pd.DataFrame:
    models = ["stl_tcn", "lstm_proposed", "lstm_pure"]
    return pd.DataFrame(
        {
            "model": models,
            "target_col": ["AUD"] * 3,
            "lookback": [56] * 3,
            "horizon": [14] * 3,
            "seed": [42] * 3,
            "params": params,
            "val_mae": [value * 1.1 for value in test_mae],
            "val_rmse": [value * 1.1 for value in test_rmse],
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }
    )


class EfficiencyMetricTests(unittest.TestCase):
    def test_epes_and_cpls_use_mae_by_default(self) -> None:
        base_df = make_metrics_frame(
            test_mae=[12.0, 10.0, 13.0],
            test_rmse=[15.0, 11.0, 16.0],
            params=[50, 100, 80],
        )

        enriched = add_epes_cpls_columns(add_legacy_efficiency_columns(base_df), "mae")
        by_model = enriched.set_index("model")
        pairwise = build_pairwise_cpl_dataframe(enriched, "mae")
        stl_vs_lstm = pairwise[(pairwise["model_a"] == "lstm_proposed") & (pairwise["model_b"] == "stl_tcn")].iloc[0]
        stl_vs_legacy = pairwise[
            (pairwise["model_a"] == "lstm_pure") & (pairwise["model_b"] == "stl_tcn")
        ].iloc[0]

        self.assertTrue((enriched["efficiency_error_metric"] == "mae").all())
        self.assertAlmostEqual(by_model.loc["stl_tcn", "epes"], 1.0 / 1.2, places=6)
        self.assertAlmostEqual(by_model.loc["lstm_proposed", "epes"], 1.0 / 2.0, places=6)
        self.assertAlmostEqual(by_model.loc["lstm_pure", "epes"], 1.0 / 1.9, places=6)
        self.assertAlmostEqual(by_model.loc["stl_tcn", "cpls"], 1.0, places=6)
        self.assertAlmostEqual(by_model.loc["lstm_proposed", "cpls"], 0.0, places=6)
        self.assertAlmostEqual(by_model.loc["lstm_pure", "cpls"], -1.0, places=6)
        self.assertEqual(stl_vs_lstm["winner"], "stl_tcn")
        self.assertAlmostEqual(
            float(stl_vs_lstm["cpl"]),
            3.801784,
            places=5,
        )
        self.assertEqual(stl_vs_legacy["state"], "dominates")

    def test_epes_and_cpls_can_switch_to_rmse(self) -> None:
        base_df = make_metrics_frame(
            test_mae=[0.50, 0.60, 0.65],
            test_rmse=[1.20, 1.00, 1.10],
            params=[100, 60, 40],
        )

        enriched = add_epes_cpls_columns(add_legacy_efficiency_columns(base_df), "rmse")
        by_model = enriched.set_index("model")

        self.assertTrue((enriched["efficiency_error_metric"] == "rmse").all())
        self.assertAlmostEqual(float(enriched["best_error_value"].iat[0]), 1.0, places=6)
        self.assertAlmostEqual(by_model.loc["stl_tcn", "epes"], 1.0 / (1.2 + (100 / 40) - 1.0), places=6)
        self.assertAlmostEqual(by_model.loc["lstm_proposed", "epes"], 1.0 / (1.0 + (60 / 40) - 1.0), places=6)
        self.assertAlmostEqual(by_model.loc["lstm_pure", "epes"], 1.0 / (1.1 + 1.0 - 1.0), places=6)
        self.assertAlmostEqual(by_model.loc["stl_tcn", "cpls"], -1.0, places=6)
        self.assertAlmostEqual(by_model.loc["lstm_proposed", "cpls"], 0.0, places=6)
        self.assertAlmostEqual(by_model.loc["lstm_pure", "cpls"], 1.0, places=6)

    def test_pairwise_cpl_marks_equal_params_pairs(self) -> None:
        base_df = pd.DataFrame(
            {
                "model": ["a", "b"],
                "params": [100, 100],
                "test_mae": [1.0, 1.2],
                "test_rmse": [1.1, 1.3],
            }
        )

        pairwise = build_pairwise_cpl_dataframe(base_df, "mae")

        self.assertEqual(pairwise["state"].iat[0], "equal_params")
        self.assertEqual(pairwise["winner"].iat[0], "a")
        self.assertTrue(pd.isna(pairwise["cpl"].iat[0]))

    def test_parameter_efficiency_columns_mark_pareto_and_two_percent_winner(self) -> None:
        base_df = pd.DataFrame(
            {
                "model": ["large_best", "small_close", "small_bad", "dominated"],
                "params": [1000, 100, 50, 2000],
                "test_mae_scaled": [1.00, 1.015, 1.10, 1.05],
            }
        )

        enriched = add_parameter_efficiency_columns(base_df, "test_mae_scaled", tolerance=0.02)
        efficient = select_tolerance_efficient_row(base_df, "test_mae_scaled", tolerance=0.02)
        by_model = enriched.set_index("model")

        self.assertEqual(efficient["model"], "small_close")
        self.assertTrue(bool(by_model.loc["large_best", "pareto_optimal"]))
        self.assertTrue(bool(by_model.loc["small_close", "pareto_optimal"]))
        self.assertTrue(bool(by_model.loc["small_bad", "pareto_optimal"]))
        self.assertFalse(bool(by_model.loc["dominated", "pareto_optimal"]))
        self.assertTrue(bool(by_model.loc["small_close", "within_2pct_accuracy_tolerance"]))
        self.assertTrue(bool(by_model.loc["small_close", "parameter_efficient_2pct"]))
        self.assertFalse(bool(by_model.loc["large_best", "parameter_efficient_2pct"]))


class EfficiencyScriptTests(unittest.TestCase):
    def test_build_efficiency_grid_summary_backfills_epes_and_cpls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            run_dir = results_dir / "lb56" / "hz14" / "seed42_20260327_120000"
            run_dir.mkdir(parents=True)
            make_metrics_frame(
                test_mae=[12.0, 10.0, 13.0],
                test_rmse=[15.0, 11.0, 16.0],
                params=[50, 100, 80],
            ).to_csv(run_dir / "metrics.csv", index=False)

            with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
                json.dump({"lookback": 56, "horizon": 14, "seed": 42, "target_col": "AUD"}, fp)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_efficiency_grid_summary.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            summary_df = pd.read_csv(results_dir / "efficiency_summary.csv")
            winners_df = pd.read_csv(results_dir / "efficiency_winners.csv")

            self.assertIn("epes", summary_df.columns)
            self.assertIn("cpls", summary_df.columns)
            self.assertIn("efficiency_error_metric", summary_df.columns)
            self.assertIn("epes_winner", winners_df.columns)
            self.assertIn("cpls_winner", winners_df.columns)
            self.assertIn("seed", winners_df.columns)
            self.assertIn("target_col", winners_df.columns)

    def test_build_efficiency_grid_summary_keeps_seeds_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            seed_to_best_model = {42: "lstm_proposed", 43: "stl_tcn"}
            for seed, best_model in seed_to_best_model.items():
                run_dir = results_dir / "lb56" / "hz14" / f"seed{seed}_20260327_120000"
                run_dir.mkdir(parents=True)
                frame = make_metrics_frame(
                    test_mae=[12.0, 10.0, 13.0] if best_model == "lstm_proposed" else [9.0, 11.0, 13.0],
                    test_rmse=[15.0, 11.0, 16.0] if best_model == "lstm_proposed" else [10.0, 12.0, 16.0],
                    params=[50, 100, 80],
                )
                frame["seed"] = seed
                frame.to_csv(run_dir / "metrics.csv", index=False)
                with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
                    json.dump({"lookback": 56, "horizon": 14, "seed": seed, "target_col": "AUD"}, fp)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_efficiency_grid_summary.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            winners_df = pd.read_csv(results_dir / "efficiency_winners.csv")
            self.assertEqual(sorted(winners_df["seed"].tolist()), [42, 43])
            self.assertEqual(len(winners_df), 2)

    def test_load_run_metrics_rejects_mixed_efficiency_metric_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "lb56" / "hz14" / "seed42_20260327_120000"
            run_dir.mkdir(parents=True)
            base_df = make_metrics_frame(
                test_mae=[12.0, 10.0, 13.0],
                test_rmse=[15.0, 11.0, 16.0],
                params=[50, 100, 80],
            )
            enriched = add_epes_cpls_columns(add_legacy_efficiency_columns(base_df), "mae")
            enriched.loc[1, "efficiency_error_metric"] = "rmse"
            enriched.to_csv(run_dir / "metrics_with_mse.csv", index=False)
            with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
                json.dump({"lookback": 56, "horizon": 14, "seed": 42, "target_col": "AUD"}, fp)

            with self.assertRaises(ValueError):
                load_run_metrics(run_dir)

    def test_plot_exchange_run_summary_supports_legacy_metrics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            metrics_path = work_dir / "metrics.csv"
            output_path = work_dir / "model_summary_test.png"
            make_metrics_frame(
                test_mae=[12.0, 10.0, 13.0],
                test_rmse=[15.0, 11.0, 16.0],
                params=[50, 100, 80],
            ).to_csv(metrics_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "plot_exchange_run_summary.py"),
                    "--metrics-path",
                    str(metrics_path),
                    "--output-path",
                    str(output_path),
                ],
                check=True,
            )

            metrics_with_mse = pd.read_csv(work_dir / "metrics_with_mse.csv")
            pairwise_cpl = pd.read_csv(work_dir / "pairwise_cpl.csv")
            mae_mse_comparison = pd.read_csv(work_dir / "mae_mse_comparison.csv")
            parameter_comparison = pd.read_csv(work_dir / "parameter_comparison.csv")
            epes_comparison = pd.read_csv(work_dir / "epes_comparison.csv")
            cpls_comparison = pd.read_csv(work_dir / "cpls_comparison.csv")

            self.assertTrue(output_path.exists())
            self.assertTrue((work_dir / "model_parameter_comparison_test.png").exists())
            self.assertTrue((work_dir / "model_epes_comparison_test.png").exists())
            self.assertTrue((work_dir / "model_cpls_comparison_test.png").exists())
            self.assertIn("epes", metrics_with_mse.columns)
            self.assertIn("cpls", metrics_with_mse.columns)
            self.assertTrue((metrics_with_mse["efficiency_error_metric"] == "mae").all())
            self.assertEqual(mae_mse_comparison.columns.tolist(), ["model", "test_mae", "test_mse"])
            self.assertEqual(parameter_comparison.columns.tolist(), ["model", "params"])
            self.assertEqual(epes_comparison.columns.tolist(), ["model", "epes", "efficiency_error_metric"])
            self.assertEqual(
                cpls_comparison.columns.tolist(),
                ["model", "cpls", "cpls_wins", "cpls_losses", "cpls_ties", "efficiency_error_metric"],
            )
            self.assertEqual(len(pairwise_cpl), 3)

    def test_plot_exchange_run_summary_respects_precomputed_rmse_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            metrics_path = work_dir / "metrics.csv"
            output_path = work_dir / "rmse_summary_test.png"
            base_df = make_metrics_frame(
                test_mae=[0.50, 0.60, 0.65],
                test_rmse=[1.20, 1.00, 1.10],
                params=[100, 60, 40],
            )
            enriched = add_epes_cpls_columns(add_legacy_efficiency_columns(base_df), "rmse")
            enriched.to_csv(metrics_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "plot_exchange_run_summary.py"),
                    "--metrics-path",
                    str(metrics_path),
                    "--output-path",
                    str(output_path),
                ],
                check=True,
            )

            metrics_with_mse = pd.read_csv(work_dir / "metrics_with_mse.csv")

            self.assertTrue(output_path.exists())
            self.assertTrue((work_dir / "rmse_parameter_comparison_test.png").exists())
            self.assertTrue((work_dir / "rmse_epes_comparison_test.png").exists())
            self.assertTrue((work_dir / "rmse_cpls_comparison_test.png").exists())
            self.assertTrue((metrics_with_mse["efficiency_error_metric"] == "rmse").all())

    def test_plot_exchange_run_summary_uses_distinct_filenames_without_summary_substring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            metrics_path = work_dir / "metrics.csv"
            output_path = work_dir / "custom.png"
            make_metrics_frame(
                test_mae=[12.0, 10.0, 13.0],
                test_rmse=[15.0, 11.0, 16.0],
                params=[50, 100, 80],
            ).to_csv(metrics_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "plot_exchange_run_summary.py"),
                    "--metrics-path",
                    str(metrics_path),
                    "--output-path",
                    str(output_path),
                ],
                check=True,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue((work_dir / "custom_parameter_comparison.png").exists())
            self.assertTrue((work_dir / "custom_epes_comparison.png").exists())
            self.assertTrue((work_dir / "custom_cpls_comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
