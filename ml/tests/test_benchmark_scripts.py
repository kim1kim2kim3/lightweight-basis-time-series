import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = (ROOT / "scripts").resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_benchmark_manifest import EXPERIMENT_FIELD_NAMES, load_model_presets, merge_experiment_kwargs_for_models


class BenchmarkScriptTests(unittest.TestCase):
    def test_model_presets_load_for_selected_models(self) -> None:
        presets = load_model_presets(("ours", "dlinear", "patchtst"))

        self.assertEqual(presets["ours_latent_groups"], 16)
        self.assertEqual(presets["ours_summary_dim"], 32)
        self.assertEqual(presets["ours_dilations"], [1, 2, 4])
        self.assertEqual(presets["dlinear_moving_avg_kernel"], 25)
        self.assertFalse(presets["dlinear_individual"])
        self.assertEqual(presets["patchtst_patch_len"], 16)
        self.assertEqual(presets["patchtst_patch_stride"], 8)
        self.assertEqual(presets["patchtst_d_model"], 512)
        self.assertEqual(presets["patchtst_num_layers"], 2)
        self.assertEqual(presets["patchtst_num_heads"], 8)
        self.assertEqual(presets["patchtst_ff_dim"], 2048)
        self.assertEqual(presets["patchtst_dropout"], 0.1)

    def test_ours_variant_uses_ours_preset(self) -> None:
        presets = load_model_presets(("ours_direct_head",))

        self.assertEqual(presets["ours_latent_groups"], 16)
        self.assertEqual(presets["ours_summary_dim"], 32)
        self.assertNotIn("patchtst_patch_len", presets)
        self.assertNotIn("dlinear_moving_avg_kernel", presets)

    def test_manifest_kwargs_override_model_presets(self) -> None:
        merged = merge_experiment_kwargs_for_models(
            ("ours", "patchtst"),
            {"ours_latent_groups": 8, "patchtst_patch_len": 12},
        )

        self.assertEqual(merged["ours_latent_groups"], 8)
        self.assertEqual(merged["patchtst_patch_len"], 12)
        self.assertEqual(merged["ours_summary_dim"], 32)
        self.assertEqual(merged["patchtst_patch_stride"], 8)

    def test_ours_parameter_optimization_defense_manifest_shape(self) -> None:
        manifest_path = ROOT.parent / "configs" / "sweeps" / "benchmark_manifest.ours_parameter_optimization_defense.json"
        with manifest_path.open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        self.assertEqual(manifest["seeds"], [42])
        self.assertEqual(manifest["datasets"][0]["name"], "ETTh1")
        self.assertEqual(manifest["datasets"][0]["horizons"], [336, 720])
        variants = manifest["sweeps"][0]["variants"]
        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "default_11k",
                "tiny_encoder",
                "small_encoder",
                "basis_light",
                "summary_light",
                "seasonal_light",
                "no_router_default",
                "fixed_bank_default",
            ],
        )
        override_keys = {
            key
            for variant in variants
            for key in variant.get("overrides", {})
        }
        self.assertFalse(override_keys - EXPERIMENT_FIELD_NAMES)

    def _make_toy_dataset(self, tmp_path: Path, rows: int = 120) -> Path:
        dataset_path = tmp_path / "toy_exchange.csv"
        t = np.arange(rows, dtype=np.float32)
        pd.DataFrame(
            {
                "AUD": 0.5 + 0.01 * t + 0.05 * np.sin(t / 3.0),
                "GBP": 0.7 + 0.02 * np.cos(t / 5.0),
                "CAD": 0.9 + 0.03 * np.sin(t / 4.0),
            }
        ).to_csv(dataset_path, index=False)
        return dataset_path

    def _write_ours_sweep_run(
        self,
        results_dir: Path,
        *,
        sweep_name: str,
        sweep_value: str,
        model: str,
        seed: int,
        horizon: int = 2,
        metrics_overrides: dict[str, object] | None = None,
        config_overrides: dict[str, object] | None = None,
    ) -> None:
        run_dir = results_dir / "toy_exchange" / sweep_name / sweep_value / "lb8" / f"hz{horizon}" / f"seed{seed}_20260410_000000"
        run_dir.mkdir(parents=True)
        metrics_payload = {
            "model": model,
            "test_mae": 1.0,
            "test_rmse": 1.1,
            "val_mae": 1.0,
            "val_rmse": 1.1,
            "params": 10,
            "test_inference_ms_per_sample": 0.5,
            "test_peak_memory_mb": 64.0,
        }
        if metrics_overrides:
            metrics_payload.update(metrics_overrides)
        pd.DataFrame([metrics_payload]).to_csv(run_dir / "metrics.csv", index=False)
        config_payload = {
            "dataset_name": "toy_exchange",
            "target_col": "AUD",
            "lookback": 8,
            "horizon": horizon,
            "seed": seed,
            "decomposition_mode": "stl",
            "stl_period": 3,
            "ours_dilations": [1, 2],
            "ours_latent_groups": 4,
        }
        if config_overrides:
            config_payload.update(config_overrides)
        (run_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
        (run_dir / "sweep_metadata.json").write_text(
            json.dumps(
                {
                    "dataset_name": "toy_exchange",
                    "sweep_name": sweep_name,
                    "sweep_value": sweep_value,
                    "variant_overrides": {},
                }
            ),
            encoding="utf-8",
        )

    def test_manifest_runner_and_benchmark_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path)
            manifest_path = tmp_path / "benchmark_manifest.json"
            results_dir = tmp_path / "results"
            with manifest_path.open("w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "defaults": {
                            "batch_size": 8,
                            "epochs": 1,
                            "patience": 1,
                            "lookback": 8,
                            "stl_period": 3,
                            "train_ratio": 0.6,
                            "val_ratio": 0.2,
                            "deterministic": True,
                            "models": ["dlinear"],
                        },
                        "seeds": [1, 2],
                        "datasets": [
                            {
                                "name": "toy_exchange",
                                "data_path": str(dataset_path),
                                "target_col": "AUD",
                                "horizons": [2],
                            }
                        ],
                    },
                    fp,
                )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "run_benchmark_manifest.py"),
                    "--manifest-path",
                    str(manifest_path),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_benchmark_report.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            summary_df = pd.read_csv(results_dir / "benchmark_seed_summary.csv")
            winners_df = pd.read_csv(results_dir / "benchmark_winners.csv")
            efficiency_df = pd.read_csv(results_dir / "benchmark_parameter_efficiency.csv")
            pareto_df = pd.read_csv(results_dir / "benchmark_pareto_summary.csv")
            self.assertEqual(summary_df["seed_count"].iat[0], 2)
            self.assertEqual(summary_df["dataset_name"].iat[0], "toy_exchange")
            self.assertEqual(summary_df["model"].iat[0], "dlinear")
            self.assertIn("mean_latency_ms_per_sample", summary_df.columns)
            self.assertIn("mean_peak_memory_mb", summary_df.columns)
            self.assertIn("pareto_optimal", summary_df.columns)
            self.assertIn("parameter_efficient_2pct", summary_df.columns)
            self.assertEqual(winners_df["mean_test_mae_winner"].iat[0], "dlinear")
            self.assertEqual(winners_df["parameter_efficient_2pct_winner"].iat[0], "dlinear")
            self.assertEqual(efficiency_df["parameter_efficient_2pct_winner"].iat[0], "dlinear")
            self.assertIn("pareto_optimal", pareto_df.columns)
            self.assertTrue(any((results_dir / "pareto_by_group").glob("*.png")))

    def test_manifest_runner_supports_ours_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path)
            manifest_path = tmp_path / "benchmark_manifest_ours.json"
            results_dir = tmp_path / "results_ours"
            with manifest_path.open("w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "defaults": {
                            "batch_size": 8,
                            "epochs": 1,
                            "patience": 1,
                            "lookback": 8,
                            "train_ratio": 0.6,
                            "val_ratio": 0.2,
                            "deterministic": True,
                            "ours_latent_groups": 4,
                            "ours_summary_dim": 8,
                            "ours_depth": 2,
                            "ours_kernel_size": 3,
                            "ours_dilations": [1, 2],
                            "ours_trend_basis_count": 4,
                            "ours_seasonal_mode_count": 2,
                            "ours_transient_basis_count": 1,
                            "models": ["ours", "ours_fixed_bank", "ours_cluster_bank", "ours_cluster_bank_fixed", "ours_extended", "ours_direct_head"],
                        },
                        "seeds": [1],
                        "datasets": [
                            {
                                "name": "toy_exchange",
                                "data_path": str(dataset_path),
                                "target_col": "AUD",
                                "horizons": [2],
                            }
                        ],
                    },
                    fp,
                )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "run_benchmark_manifest.py"),
                    "--manifest-path",
                    str(manifest_path),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_benchmark_report.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            summary_df = pd.read_csv(results_dir / "benchmark_seed_summary.csv")
            self.assertEqual(
                set(summary_df["model"]),
                {"ours", "ours_fixed_bank", "ours_cluster_bank", "ours_cluster_bank_fixed", "ours_extended", "ours_direct_head"},
            )
            self.assertIn("mean_latency_ms_per_sample", summary_df.columns)
            self.assertIn("mean_peak_memory_mb", summary_df.columns)

    def test_ours_sweep_runner_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path)
            manifest_path = tmp_path / "ours_sweep_manifest.json"
            results_dir = tmp_path / "ours_sweep_results"
            with manifest_path.open("w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "defaults": {
                            "batch_size": 8,
                            "epochs": 1,
                            "patience": 1,
                            "lookback": 8,
                            "train_ratio": 0.6,
                            "val_ratio": 0.2,
                            "deterministic": True,
                            "ours_latent_groups": 4,
                            "ours_summary_dim": 8,
                            "ours_depth": 2,
                            "ours_kernel_size": 3,
                            "ours_dilations": [1, 2],
                            "ours_trend_basis_count": 4,
                            "ours_seasonal_mode_count": 2,
                            "ours_transient_basis_count": 1,
                            "models": ["ours"],
                        },
                        "seeds": [1],
                        "datasets": [
                            {
                                "name": "toy_exchange",
                                "data_path": str(dataset_path),
                                "target_col": "AUD",
                                "horizons": [2],
                            }
                        ],
                        "sweeps": [
                            {
                                "name": "group_size",
                                "variants": [
                                    {"label": "G4", "overrides": {"ours_latent_groups": 4}},
                                    {"label": "G8", "overrides": {"ours_latent_groups": 8}},
                                ],
                            },
                            {
                                "name": "basis_count",
                                "variants": [
                                    {
                                        "label": "Kse2_Kre1",
                                        "overrides": {
                                            "ours_seasonal_mode_count": 2,
                                            "ours_transient_basis_count": 1,
                                        },
                                    }
                                ],
                            },
                            {
                                "name": "multiscale_ablation",
                                "variants": [
                                    {"label": "single_scale", "overrides": {"ours_dilations": [1]}},
                                    {"label": "multi_scale", "overrides": {"ours_dilations": [1, 2]}},
                                ],
                            },
                        ],
                    },
                    fp,
                )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "run_ours_sweep_manifest.py"),
                    "--manifest-path",
                    str(manifest_path),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_ours_sweep_report.py"),
                    "--results-dir",
                    str(results_dir),
                ],
                check=True,
            )

            summary_df = pd.read_csv(results_dir / "sweep_summary.csv")
            winners_df = pd.read_csv(results_dir / "sweep_winners.csv")
            self.assertEqual(set(summary_df["sweep_name"]), {"group_size", "basis_count", "multiscale_ablation"})
            self.assertEqual(set(summary_df["sweep_value"]), {"G4", "G8", "Kse2_Kre1", "single_scale", "multi_scale"})
            self.assertEqual(set(summary_df["model"]), {"ours"})
            self.assertEqual(set(winners_df["sweep_name"]), {"group_size", "basis_count", "multiscale_ablation"})
            self.assertIn("best_mean_test_mae_sweep_value", winners_df.columns)
            self.assertIn("best_mean_peak_memory_mb_model", winners_df.columns)
            sweep_metadata_files = list(results_dir.rglob("sweep_metadata.json"))
            self.assertEqual(len(sweep_metadata_files), 5)
            self.assertTrue(any((results_dir / "pareto_by_group").glob("*.png")))
            self.assertTrue(any((results_dir / "branch_usage_by_group").glob("*.csv")))
            self.assertTrue(any((results_dir / "branch_usage_by_group").glob("*.png")))
            self.assertTrue(any((results_dir / "frequency_by_group").glob("*.csv")))
            self.assertTrue(any((results_dir / "frequency_by_group").glob("*.png")))

    def test_ours_sweep_report_selects_variant_against_baseline_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            baseline_path = tmp_path / "baseline_summary.csv"
            sweep_name = "parameter_optimization_defense"
            variant_specs = {
                "default_11k": {336: (1.010, 10), 720: (2.020, 10)},
                "small_encoder": {336: (1.015, 5), 720: (2.030, 5)},
                "tiny_encoder": {336: (1.030, 3), 720: (2.100, 3)},
            }
            for variant, horizon_specs in variant_specs.items():
                for horizon, (scaled_mae, params) in horizon_specs.items():
                    self._write_ours_sweep_run(
                        results_dir,
                        sweep_name=sweep_name,
                        sweep_value=variant,
                        model="ours",
                        seed=42,
                        horizon=horizon,
                        metrics_overrides={
                            "test_mae_scaled": scaled_mae,
                            "test_rmse_scaled": scaled_mae + 0.1,
                            "val_mae_scaled": scaled_mae + 0.2,
                            "val_rmse_scaled": scaled_mae + 0.3,
                            "params": params,
                        },
                    )

            pd.DataFrame(
                [
                    {
                        "dataset_name": "toy_exchange",
                        "target_col": "AUD",
                        "lookback": 8,
                        "horizon": 336,
                        "model": "patchtst",
                        "mean_test_mae_scaled": 1.000,
                        "mean_params": 1000,
                    },
                    {
                        "dataset_name": "toy_exchange",
                        "target_col": "AUD",
                        "lookback": 8,
                        "horizon": 720,
                        "model": "patchtst",
                        "mean_test_mae_scaled": 2.000,
                        "mean_params": 1000,
                    },
                ]
            ).to_csv(baseline_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "build_ours_sweep_report.py"),
                    "--results-dir",
                    str(results_dir),
                    "--baseline-summary-path",
                    str(baseline_path),
                    "--selection-horizons",
                    "336,720",
                ],
                check=True,
            )

            summary_df = pd.read_csv(results_dir / "sweep_summary.csv")
            baseline_df = pd.read_csv(results_dir / "sweep_baseline_tolerance.csv")
            selected_df = pd.read_csv(results_dir / "sweep_selected_variant.csv")

            self.assertIn("mean_test_mae_scaled", summary_df.columns)
            self.assertIn("pareto_optimal", summary_df.columns)
            self.assertIn("within_2pct_accuracy_tolerance", summary_df.columns)
            self.assertIn("parameter_efficient_2pct", summary_df.columns)
            self.assertIn("within_2pct_patchtst_tolerance", baseline_df.columns)
            tiny_rows = baseline_df[baseline_df["sweep_value"] == "tiny_encoder"]
            self.assertFalse(tiny_rows["within_2pct_patchtst_tolerance"].all())
            self.assertEqual(selected_df["sweep_value"].iat[0], "small_encoder")
            self.assertTrue(bool(selected_df["selected_variant"].iat[0]))

    def test_ours_sweep_report_rejects_mixed_config_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            self._write_ours_sweep_run(results_dir, sweep_name="group_size", sweep_value="G4", model="ours", seed=1)
            self._write_ours_sweep_run(
                results_dir,
                sweep_name="group_size",
                sweep_value="G4",
                model="ours",
                seed=2,
                config_overrides={"ours_dilations": [1]},
            )

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "build_ours_sweep_report.py"),
                        "--results-dir",
                        str(results_dir),
                    ],
                    check=True,
                )

    def test_ours_sweep_report_rejects_inconsistent_seed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            self._write_ours_sweep_run(results_dir, sweep_name="group_size", sweep_value="G4", model="ours", seed=1)
            self._write_ours_sweep_run(results_dir, sweep_name="group_size", sweep_value="G4", model="ours", seed=2)
            self._write_ours_sweep_run(results_dir, sweep_name="group_size", sweep_value="G8", model="ours", seed=1)

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "build_ours_sweep_report.py"),
                        "--results-dir",
                        str(results_dir),
                    ],
                    check=True,
                )

    def test_benchmark_report_rejects_mixed_config_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            run_dir_a = results_dir / "lb8" / "hz2" / "seed1_20260410_000000"
            run_dir_b = results_dir / "lb8" / "hz2" / "seed2_20260410_000000"
            run_dir_a.mkdir(parents=True)
            run_dir_b.mkdir(parents=True)

            metrics = pd.DataFrame(
                [{"model": "dlinear", "test_mae": 1.0, "test_rmse": 1.1, "val_mae": 1.0, "val_rmse": 1.1, "params": 10}]
            )
            metrics.to_csv(run_dir_a / "metrics.csv", index=False)
            metrics.to_csv(run_dir_b / "metrics.csv", index=False)
            (run_dir_a / "config.json").write_text(
                json.dumps(
                    {
                        "dataset_name": "toy_exchange",
                        "target_col": "AUD",
                        "lookback": 8,
                        "horizon": 2,
                        "seed": 1,
                        "decomposition_mode": "stl",
                        "stl_period": 3,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir_b / "config.json").write_text(
                json.dumps(
                    {
                        "dataset_name": "toy_exchange",
                        "target_col": "AUD",
                        "lookback": 8,
                        "horizon": 2,
                        "seed": 2,
                        "decomposition_mode": "mstl",
                        "mstl_periods": [24, 168],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "build_benchmark_report.py"),
                        "--results-dir",
                        str(results_dir),
                    ],
                    check=True,
                )

    def test_benchmark_report_rejects_inconsistent_seed_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            for seed, rows in (
                (1, [{"model": "dlinear", "test_mae": 1.0, "test_rmse": 1.1, "val_mae": 1.0, "val_rmse": 1.1, "params": 10},
                     {"model": "patchtst", "test_mae": 0.9, "test_rmse": 1.0, "val_mae": 0.9, "val_rmse": 1.0, "params": 20}]),
                (2, [{"model": "dlinear", "test_mae": 1.1, "test_rmse": 1.2, "val_mae": 1.1, "val_rmse": 1.2, "params": 10}]),
            ):
                run_dir = results_dir / "lb8" / "hz2" / f"seed{seed}_20260410_000000"
                run_dir.mkdir(parents=True)
                pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "dataset_name": "toy_exchange",
                            "target_col": "AUD",
                            "lookback": 8,
                            "horizon": 2,
                            "seed": seed,
                            "decomposition_mode": "stl",
                            "stl_period": 3,
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "build_benchmark_report.py"),
                        "--results-dir",
                        str(results_dir),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    unittest.main()
