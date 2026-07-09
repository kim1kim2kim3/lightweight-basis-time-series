import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = (ROOT / "scripts").resolve()


class SciOAReportTests(unittest.TestCase):
    def _write_run(
        self,
        results_dir: Path,
        *,
        dataset_name: str,
        target_col: str,
        lookback: int,
        horizon: int,
        seed: int,
        rows: list[dict[str, object]],
        config_extra: dict[str, object] | None = None,
    ) -> None:
        run_dir = results_dir / f"lb{lookback}" / f"hz{horizon}" / f"seed{seed}_20260410_000000"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_name": dataset_name,
            "target_col": target_col,
            "lookback": lookback,
            "horizon": horizon,
            "seed": seed,
        }
        if config_extra:
            payload.update(config_extra)
        with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
            json.dump(payload, fp)
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)

    def test_build_sci_oa_report_generates_csv_and_markdown_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"

            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=1,
                rows=[
                    {
                        "model": "stl_tcn",
                        "test_mae": 0.52,
                        "test_rmse": 0.75,
                        "val_mae": 0.50,
                        "val_rmse": 0.73,
                        "params": 100,
                        "test_inference_ms_per_sample": 0.30,
                        "test_peak_memory_mb": 30.0,
                        "epes": 99.0,
                        "cpls": 5.0,
                    },
                    {
                        "model": "dlinear",
                        "test_mae": 0.50,
                        "test_rmse": 0.80,
                        "val_mae": 0.49,
                        "val_rmse": 0.79,
                        "params": 40,
                        "test_inference_ms_per_sample": 0.10,
                        "test_peak_memory_mb": 20.0,
                        "epes": 999.0,
                        "cpls": 7.0,
                    },
                    {
                        "model": "patchtst",
                        "test_mae": 0.50,
                        "test_rmse": 0.70,
                        "val_mae": 0.48,
                        "val_rmse": 0.69,
                        "params": 200,
                        "test_inference_ms_per_sample": 0.60,
                        "test_peak_memory_mb": 40.0,
                        "epes": 111.0,
                        "cpls": 1.0,
                    },
                ],
            )
            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=2,
                rows=[
                    {
                        "model": "stl_tcn",
                        "test_mae": 0.54,
                        "test_rmse": 0.77,
                        "val_mae": 0.53,
                        "val_rmse": 0.76,
                        "params": 100,
                        "test_inference_ms_per_sample": 0.35,
                        "test_peak_memory_mb": 31.0,
                        "epes": 88.0,
                        "cpls": 4.0,
                    },
                    {
                        "model": "dlinear",
                        "test_mae": 0.52,
                        "test_rmse": 0.82,
                        "val_mae": 0.51,
                        "val_rmse": 0.81,
                        "params": 40,
                        "test_inference_ms_per_sample": 0.11,
                        "test_peak_memory_mb": 21.0,
                        "epes": 777.0,
                        "cpls": 8.0,
                    },
                    {
                        "model": "patchtst",
                        "test_mae": 0.50,
                        "test_rmse": 0.72,
                        "val_mae": 0.49,
                        "val_rmse": 0.71,
                        "params": 200,
                        "test_inference_ms_per_sample": 0.62,
                        "test_peak_memory_mb": 42.0,
                        "epes": 100.0,
                        "cpls": 2.0,
                    },
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_sci_oa_report.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            output_dir = results_dir / "sci_oa_report"
            aggregate_df = pd.read_csv(output_dir / "aggregate_metrics.csv")
            winners_df = pd.read_csv(output_dir / "winner_summary.csv")
            table_df = pd.read_csv(output_dir / "tables" / "toy_exchange_lb8_h2.csv")
            paper_summary = (output_dir / "paper_summary.md").read_text(encoding="utf-8")
            table_md = (output_dir / "tables" / "toy_exchange_lb8_h2.md").read_text(encoding="utf-8")

            self.assertEqual(set(aggregate_df["model"]), {"stl_tcn", "dlinear", "patchtst"})
            self.assertEqual(int(aggregate_df.loc[aggregate_df["model"] == "dlinear", "seed_count"].iat[0]), 2)
            self.assertNotIn("mean_epes", aggregate_df.columns)
            self.assertNotIn("mean_cpls", aggregate_df.columns)
            self.assertEqual(winners_df["best_mae_model"].iat[0], "patchtst")
            self.assertEqual(winners_df["best_rmse_model"].iat[0], "patchtst")
            self.assertEqual(winners_df["smallest_params_model"].iat[0], "dlinear")
            self.assertEqual(winners_df["parameter_efficient_2pct_model"].iat[0], "dlinear")
            self.assertEqual(winners_df["lowest_latency_model"].iat[0], "dlinear")
            self.assertEqual(winners_df["lowest_memory_model"].iat[0], "dlinear")
            self.assertIn("mean_latency_ms_per_sample", table_df.columns)
            self.assertIn("mean_peak_memory_mb", table_df.columns)
            self.assertIn("pareto_optimal", table_df.columns)
            self.assertIn("parameter_efficient_2pct", table_df.columns)
            self.assertIn("+-", paper_summary)
            self.assertIn(
                "Winners: MAE=`patchtst`, RMSE=`patchtst`, Params=`dlinear`, Efficient@2%=`dlinear`, Latency=`dlinear`, Memory=`dlinear`",
                paper_summary,
            )
            self.assertIn("**0.5000 +- 0.0000**", paper_summary)
            self.assertIn("Latency (ms/sample)", table_md)
            self.assertNotIn("epes", paper_summary.lower())
            self.assertNotIn("cpls", paper_summary.lower())

    def test_build_sci_oa_report_omits_latency_and_memory_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"

            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=1,
                rows=[
                    {
                        "model": "dlinear",
                        "test_mae": 0.50,
                        "test_rmse": 0.80,
                        "val_mae": 0.49,
                        "val_rmse": 0.79,
                        "params": 40,
                    },
                    {
                        "model": "patchtst",
                        "test_mae": 0.48,
                        "test_rmse": 0.70,
                        "val_mae": 0.47,
                        "val_rmse": 0.69,
                        "params": 200,
                    },
                ],
            )
            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=2,
                rows=[
                    {
                        "model": "dlinear",
                        "test_mae": 0.52,
                        "test_rmse": 0.82,
                        "val_mae": 0.51,
                        "val_rmse": 0.81,
                        "params": 40,
                    },
                    {
                        "model": "patchtst",
                        "test_mae": 0.49,
                        "test_rmse": 0.71,
                        "val_mae": 0.48,
                        "val_rmse": 0.70,
                        "params": 200,
                    },
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_sci_oa_report.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            output_dir = results_dir / "sci_oa_report"
            table_df = pd.read_csv(output_dir / "tables" / "toy_exchange_lb8_h2.csv")
            winners_df = pd.read_csv(output_dir / "winner_summary.csv")
            paper_summary = (output_dir / "paper_summary.md").read_text(encoding="utf-8")

            self.assertNotIn("mean_latency_ms_per_sample", table_df.columns)
            self.assertNotIn("mean_peak_memory_mb", table_df.columns)
            self.assertNotIn("lowest_latency_model", winners_df.columns)
            self.assertNotIn("lowest_memory_model", winners_df.columns)
            self.assertNotIn("Latency (ms/sample)", paper_summary)
            self.assertNotIn("Peak Memory (MB)", paper_summary)

    def test_build_sci_oa_report_rejects_mixed_config_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"

            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=1,
                config_extra={"decomposition_mode": "stl", "stl_period": 3},
                rows=[{"model": "dlinear", "test_mae": 0.5, "test_rmse": 0.8, "val_mae": 0.49, "val_rmse": 0.79, "params": 40}],
            )
            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=2,
                config_extra={"decomposition_mode": "mstl", "mstl_periods": [24, 168]},
                rows=[{"model": "dlinear", "test_mae": 0.52, "test_rmse": 0.82, "val_mae": 0.51, "val_rmse": 0.81, "params": 40}],
            )

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "build_sci_oa_report.py"),
                        "--results-dir",
                        str(results_dir),
                    ],
                    check=True,
                )

    def test_build_sci_oa_report_rejects_inconsistent_seed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"

            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=1,
                config_extra={"decomposition_mode": "stl", "stl_period": 3},
                rows=[
                    {"model": "dlinear", "test_mae": 0.50, "test_rmse": 0.80, "val_mae": 0.49, "val_rmse": 0.79, "params": 40},
                    {"model": "patchtst", "test_mae": 0.48, "test_rmse": 0.70, "val_mae": 0.47, "val_rmse": 0.69, "params": 200},
                ],
            )
            self._write_run(
                results_dir,
                dataset_name="toy_exchange",
                target_col="AUD",
                lookback=8,
                horizon=2,
                seed=2,
                config_extra={"decomposition_mode": "stl", "stl_period": 3},
                rows=[
                    {"model": "dlinear", "test_mae": 0.52, "test_rmse": 0.82, "val_mae": 0.51, "val_rmse": 0.81, "params": 40},
                ],
            )

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "build_sci_oa_report.py"),
                        "--results-dir",
                        str(results_dir),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    unittest.main()
