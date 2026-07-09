import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = (ROOT / "scripts").resolve()
sys.path.append(str(SCRIPTS_DIR))

import train_exchange_once as teo
from models.patchtst import (
    OFFICIAL_PATCHTST_D_MODEL,
    OFFICIAL_PATCHTST_DROPOUT,
    OFFICIAL_PATCHTST_FF_DIM,
    OFFICIAL_PATCHTST_NUM_HEADS,
    OFFICIAL_PATCHTST_NUM_LAYERS,
    OFFICIAL_PATCHTST_PATCH_LEN,
    OFFICIAL_PATCHTST_PATCH_STRIDE,
)
from models.tcn import OFFICIAL_TCN_DILATIONS


class TrainExchangeUtilityTests(unittest.TestCase):
    def test_default_benchmark_models_exclude_internal_modified_lstm(self) -> None:
        self.assertEqual(
            teo.DEFAULT_BENCHMARK_MODELS,
            ("dlinear", "gru", "lstm_pure", "patchtst"),
        )
        default_config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
        )
        self.assertEqual(default_config.models, teo.DEFAULT_BENCHMARK_MODELS)

    def test_patchtst_defaults_match_official_supervised_entrypoint(self) -> None:
        expected_patchtst_defaults = {
            "patchtst_patch_len": 16,
            "patchtst_patch_stride": 8,
            "patchtst_d_model": 512,
            "patchtst_num_layers": 2,
            "patchtst_num_heads": 8,
            "patchtst_ff_dim": 2048,
            "patchtst_dropout": 0.1,
        }
        constant_defaults = {
            "patchtst_patch_len": OFFICIAL_PATCHTST_PATCH_LEN,
            "patchtst_patch_stride": OFFICIAL_PATCHTST_PATCH_STRIDE,
            "patchtst_d_model": OFFICIAL_PATCHTST_D_MODEL,
            "patchtst_num_layers": OFFICIAL_PATCHTST_NUM_LAYERS,
            "patchtst_num_heads": OFFICIAL_PATCHTST_NUM_HEADS,
            "patchtst_ff_dim": OFFICIAL_PATCHTST_FF_DIM,
            "patchtst_dropout": OFFICIAL_PATCHTST_DROPOUT,
        }
        self.assertEqual(constant_defaults, expected_patchtst_defaults)

        default_config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
        )
        for key, expected_value in expected_patchtst_defaults.items():
            self.assertEqual(getattr(default_config, key), expected_value)

        parser = teo.build_parser()
        args = parser.parse_args([])
        for key, expected_value in expected_patchtst_defaults.items():
            self.assertEqual(getattr(args, key), expected_value)

    def test_load_time_ordered_frame_drops_explicit_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "toy_with_date.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-01"],
                    "OT": [2.0, 1.0],
                    "HUFL": [0.2, 0.1],
                }
            ).to_csv(dataset_path, index=False)

            frame, metadata = teo.load_time_ordered_frame(dataset_path)

            self.assertEqual(metadata["time_order_column"], "date")
            self.assertNotIn("date", frame.columns)
            self.assertEqual(frame.columns.tolist(), ["OT", "HUFL"])
            self.assertEqual(frame["OT"].tolist(), [1.0, 2.0])

    def test_split_labels_marks_boundary_crossing_windows(self) -> None:
        end_indices = np.array([56, 57, 70, 72, 85])
        labels = teo.split_labels(end_indices, horizon=14, train_end=70, val_end=85)
        self.assertEqual(labels.tolist(), ["train", "cross_split", "val", "cross_split", "test"])

    def test_resolve_split_boundaries_uses_official_ett_boundaries(self) -> None:
        train_end, val_end = teo.resolve_split_boundaries(
            dataset_name="ETTh1",
            total_len=17420,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        self.assertEqual(train_end, 12 * 30 * 24)
        self.assertEqual(val_end, (12 + 4) * 30 * 24)

        train_end, val_end = teo.resolve_split_boundaries(
            dataset_name="ETTm2",
            total_len=69680,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        self.assertEqual(train_end, 12 * 30 * 24 * 4)
        self.assertEqual(val_end, (12 + 4) * 30 * 24 * 4)

    def test_resolve_split_boundaries_keeps_ratio_for_non_ett_datasets(self) -> None:
        train_end, val_end = teo.resolve_split_boundaries(
            dataset_name="toy_exchange",
            total_len=100,
            train_ratio=0.6,
            val_ratio=0.2,
        )
        self.assertEqual(train_end, 60)
        self.assertEqual(val_end, 80)

    def test_train_model_uses_configured_efficiency_metric_for_checkpoint_selection(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.0)
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            epochs=2,
            patience=2,
            efficiency_error_metric="rmse",
            models=("lstm_proposed",),
        )
        epoch_weights = iter([1.0, 2.0])

        def fake_run_epoch(*args, **kwargs) -> float:
            args[1].weight.data.fill_(next(epoch_weights))
            return 0.0

        def fake_collect_predictions(model_name, current_model, loader, device):
            value = float(current_model.weight.item())
            return np.array([[value]], dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

        def fake_compute_metrics(pred_scaled, target_scaled, scaler, target_idx):
            value = float(pred_scaled[0, 0])
            if value < 1.5:
                return {"mae": 1.0, "rmse": 2.0}
            return {"mae": 2.0, "rmse": 1.0}

        with patch.object(teo, "run_epoch", side_effect=fake_run_epoch), patch.object(
            teo, "compute_average_loss", return_value=0.0
        ), patch.object(teo, "collect_predictions", side_effect=fake_collect_predictions), patch.object(
            teo, "compute_metrics", side_effect=fake_compute_metrics
        ):
            trained_model, best_val_metrics, history = teo.train_model(
                model_name="lstm_proposed",
                model=model,
                train_loader=None,
                val_loader=None,
                scaler=None,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )

        self.assertAlmostEqual(float(trained_model.weight.item()), 2.0, places=6)
        self.assertEqual(best_val_metrics["rmse"], 1.0)
        self.assertEqual(history[-1]["selection_metric_name"], "rmse")

    def test_tslib_optimizer_and_mse_loss_options_are_selectable(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            optimizer_name="adam",
            loss_name="mse",
        )
        model = torch.nn.Linear(1, 1)

        optimizer = teo.build_optimizer(model, config)

        self.assertIsInstance(optimizer, torch.optim.Adam)
        for model_name in ("ours", "dlinear", "patchtst"):
            self.assertIsInstance(teo.select_loss_fn(model_name, config), torch.nn.MSELoss)

    def test_tslib_type1_lr_schedule_matches_tslib_post_epoch_timing(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            epochs=3,
            patience=99,
            lr=0.1,
            lr_schedule="tslib_type1",
            models=("lstm_proposed",),
        )
        observed_lrs: list[float] = []
        epoch_weights = iter([1.0, 2.0, 3.0])

        def fake_run_epoch(model_name, current_model, loader, optimizer, loss_fn, current_config, device) -> float:
            observed_lrs.append(float(optimizer.param_groups[0]["lr"]))
            current_model.weight.data.fill_(next(epoch_weights))
            return 0.0

        def fake_collect_predictions(model_name, current_model, loader, device):
            value = float(current_model.weight.item())
            return np.array([[value]], dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

        def fake_compute_metrics(pred_scaled, target_scaled, scaler, target_idx):
            value = float(pred_scaled[0, 0])
            return {"mae": value, "rmse": value}

        with patch.object(teo, "run_epoch", side_effect=fake_run_epoch), patch.object(
            teo, "compute_average_loss", return_value=0.0
        ), patch.object(teo, "collect_predictions", side_effect=fake_collect_predictions), patch.object(
            teo, "compute_metrics", side_effect=fake_compute_metrics
        ):
            teo.train_model(
                model_name="lstm_proposed",
                model=model,
                train_loader=None,
                val_loader=None,
                scaler=None,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(observed_lrs), 3)
        self.assertAlmostEqual(observed_lrs[0], 0.1, places=8)
        self.assertAlmostEqual(observed_lrs[1], 0.1, places=8)
        self.assertAlmostEqual(observed_lrs[2], 0.05, places=8)

    def test_train_model_can_select_best_state_by_validation_loss(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.0)
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            epochs=2,
            patience=2,
            early_stopping_metric="val_loss",
            efficiency_error_metric="mae",
            models=("lstm_proposed",),
        )
        epoch_weights = iter([1.0, 2.0])
        val_losses = iter([0.2, 0.1])

        def fake_run_epoch(*args, **kwargs) -> float:
            args[1].weight.data.fill_(next(epoch_weights))
            return 0.0

        def fake_collect_predictions(model_name, current_model, loader, device):
            value = float(current_model.weight.item())
            return np.array([[value]], dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

        def fake_compute_metrics(pred_scaled, target_scaled, scaler, target_idx):
            value = float(pred_scaled[0, 0])
            if value < 1.5:
                return {"mae": 1.0, "rmse": 1.0}
            return {"mae": 2.0, "rmse": 2.0}

        with patch.object(teo, "run_epoch", side_effect=fake_run_epoch), patch.object(
            teo, "compute_average_loss", side_effect=lambda *args, **kwargs: next(val_losses)
        ), patch.object(teo, "collect_predictions", side_effect=fake_collect_predictions), patch.object(
            teo, "compute_metrics", side_effect=fake_compute_metrics
        ):
            trained_model, _best_val_metrics, history = teo.train_model(
                model_name="lstm_proposed",
                model=model,
                train_loader=None,
                val_loader=None,
                scaler=None,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )

        self.assertAlmostEqual(float(trained_model.weight.item()), 2.0, places=6)
        self.assertEqual(history[-1]["selection_metric_name"], "val_loss")
        self.assertAlmostEqual(float(history[-1]["val_selection_metric"]), 0.1, places=6)

    def test_compute_metrics_returns_scaled_and_inverse_values(self) -> None:
        scaler = teo.FeatureScaler()
        scaler.fit(np.array([[10.0], [14.0]], dtype=np.float32))
        pred_scaled = np.array([[0.0], [1.0]], dtype=np.float32)
        target_scaled = np.array([[1.0], [0.0]], dtype=np.float32)

        metrics = teo.compute_metrics(pred_scaled, target_scaled, scaler, target_idx=0)

        self.assertEqual(set(metrics), {"mae", "rmse", "mae_scaled", "rmse_scaled"})
        self.assertAlmostEqual(metrics["mae_scaled"], 1.0, places=6)
        self.assertAlmostEqual(metrics["rmse_scaled"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mae"], 2.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 2.0, places=6)

    def test_build_prediction_export_frame_supports_scaled_export(self) -> None:
        scaler = teo.FeatureScaler()
        scaler.fit(np.array([[10.0], [14.0]], dtype=np.float32))
        predictions = {"demo": np.array([[0.0, 1.0]], dtype=np.float32)}
        actual_scaled = np.array([[1.0, 0.0]], dtype=np.float32)
        end_indices = np.array([8], dtype=np.int64)
        split = np.array(["val"], dtype=object)

        inverse_frame = teo.build_prediction_export_frame(
            predictions_by_model=predictions,
            actual_scaled=actual_scaled,
            scaler=scaler,
            target_idx=0,
            end_indices=end_indices,
            split=split,
            inverse_transform=True,
        )
        scaled_frame = teo.build_prediction_export_frame(
            predictions_by_model=predictions,
            actual_scaled=actual_scaled,
            scaler=scaler,
            target_idx=0,
            end_indices=end_indices,
            split=split,
            inverse_transform=False,
        )

        self.assertEqual(float(inverse_frame["actual_t+1"].iat[0]), 14.0)
        self.assertEqual(float(inverse_frame["pred_t+2"].iat[0]), 14.0)
        self.assertEqual(float(scaled_frame["actual_t+1"].iat[0]), 1.0)
        self.assertEqual(float(scaled_frame["pred_t+2"].iat[0]), 1.0)

    def test_build_model_rejects_stl_tcn_with_all_branches_disabled(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            stl_use_trend_branch=False,
            stl_use_season_branch=False,
            stl_use_resid_branch=False,
        )
        with self.assertRaises(ValueError):
            teo.build_model(
                "stl_tcn",
                input_dim=3,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )

    def test_build_model_supports_standard_baseline_set(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=8,
            horizon=2,
            stl_period=3,
            patchtst_patch_len=4,
            patchtst_patch_stride=2,
            patchtst_d_model=16,
            patchtst_num_heads=4,
            patchtst_ff_dim=32,
            patchtst_num_layers=1,
            models=("dlinear", "gru", "lstm_pure", "patchtst"),
        )
        for model_name in config.models:
            model = teo.build_model(
                model_name,
                input_dim=3,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )
            self.assertGreater(teo.count_parameters(model), 0)

    def test_tcn_wrapper_returns_target_slice_from_backbone_output(self) -> None:
        torch.manual_seed(0)
        model = teo.TCNForecast(
            lookback=8,
            input_dim=3,
            horizon=2,
            target_idx=1,
        )
        model.eval()
        batch = torch.randn(4, 8, 3)
        with torch.no_grad():
            wrapped = model(batch)
            raw = model.backbone(batch)
        self.assertEqual(tuple(wrapped.shape), (4, 2))
        self.assertEqual(tuple(raw.shape), (4, 2, 3))
        self.assertTrue(torch.allclose(wrapped, raw[:, :, 1]))

    def test_tcn_parameter_counts_match_official_regression_values(self) -> None:
        expected_params_by_horizon = {
            96: 853792,
            192: 1069504,
            336: 1393072,
            720: 2255920,
        }
        for horizon, expected_params in expected_params_by_horizon.items():
            with self.subTest(horizon=horizon):
                model = teo.TCNForecast(
                    lookback=96,
                    input_dim=7,
                    horizon=horizon,
                    target_idx=0,
                )
                self.assertEqual(teo.count_parameters(model), expected_params)

    def test_build_model_rejects_non_default_tcn_hyperparameters(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            tcn_hidden_dim=64,
        )
        with self.assertRaisesRegex(ValueError, "Official OnlineTSF TCN wrapper only supports"):
            teo.build_model(
                "tcn",
                input_dim=3,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )

    def test_tcn_wrapper_rejects_non_default_direct_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "Official OnlineTSF TCN wrapper only supports"):
            teo.TCNForecast(
                lookback=8,
                input_dim=3,
                horizon=2,
                target_idx=0,
                dilations=OFFICIAL_TCN_DILATIONS + (8,),
            )

    def test_exchange_target_dataset_emits_target_only_future(self) -> None:
        values = np.stack(
            [
                np.linspace(0.0, 1.0, 24, dtype=np.float32),
                np.linspace(1.0, 2.0, 24, dtype=np.float32),
                np.linspace(2.0, 3.0, 24, dtype=np.float32),
            ],
            axis=1,
        )
        dataset = teo.ExchangeTargetDataset(
            values=values,
            end_indices=np.array([8, 9], dtype=np.int64),
            target_idx=0,
            lookback=6,
            horizon=3,
            stl_period=3,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["y"].shape), (3,))
        self.assertNotIn("y_full", sample)

    def test_make_datasets_skips_decomposition_when_no_selected_model_needs_it(self) -> None:
        values = np.stack(
            [
                np.linspace(0.0, 1.0, 32, dtype=np.float32),
                np.linspace(1.0, 2.0, 32, dtype=np.float32),
                np.linspace(2.0, 3.0, 32, dtype=np.float32),
            ],
            axis=1,
        )
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=6,
            horizon=3,
            stl_period=3,
            train_ratio=0.6,
            val_ratio=0.2,
            models=("ours", "dlinear", "patchtst"),
        )

        with patch.object(
            teo,
            "decompose_multichannel_history",
            side_effect=AssertionError("STL precompute should be skipped"),
        ), patch.object(
            teo,
            "decompose_history",
            side_effect=AssertionError("STL precompute should be skipped"),
        ):
            train_ds, val_ds, test_ds, _all_ds, _scaler, stats, _all_end_indices = teo.make_datasets(
                values,
                target_idx=0,
                config=config,
            )

        self.assertEqual(len(train_ds), stats["train_windows"])
        self.assertEqual(len(val_ds), stats["val_windows"])
        self.assertEqual(len(test_ds), stats["test_windows"])
        self.assertEqual(stats["train_decomposition_fallback_events"], 0)
        self.assertEqual(tuple(train_ds[0]["x_hist_full"].shape), (6, 3))
        self.assertEqual(tuple(train_ds[0]["y"].shape), (3,))

    def test_build_model_rejects_ours_with_all_branches_disabled(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            horizon=3,
            ours_use_trend_branch=False,
            ours_use_seasonal_branch=False,
            ours_use_transient_branch=False,
        )
        with self.assertRaises(ValueError):
            teo.build_model(
                "ours",
                input_dim=3,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )

    def test_build_model_allows_ours_direct_head_with_all_structural_branches_disabled(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            horizon=3,
            ours_use_trend_branch=False,
            ours_use_seasonal_branch=False,
            ours_use_transient_branch=False,
        )
        model = teo.build_model(
            "ours_direct_head",
            input_dim=3,
            target_idx=0,
            config=config,
            device=torch.device("cpu"),
        )
        output = teo.model_forward("ours_direct_head", model, {"x_hist_full": torch.randn(2, 8, 3)})
        self.assertEqual(tuple(output["target_pred"].shape), (2, 3))

    def test_build_model_supports_ours_variants(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=8,
            horizon=3,
            ours_latent_groups=4,
            ours_summary_dim=8,
            ours_depth=2,
            ours_kernel_size=3,
            ours_dilations=(1, 2),
            ours_trend_basis_count=4,
            ours_seasonal_mode_count=2,
            ours_transient_basis_count=1,
            models=(
                "ours",
                "ours_no_router",
                "ours_fixed_bank",
                "ours_cluster_bank",
                "ours_cluster_bank_fixed",
                "ours_extended",
                "ours_no_transient",
                "ours_no_seasonal",
                "ours_trend_only",
                "ours_direct_head",
            ),
        )
        batch = {"x_hist_full": torch.randn(2, 8, 3)}
        for model_name in config.models:
            model = teo.build_model(
                model_name,
                input_dim=3,
                target_idx=0,
                config=config,
                device=torch.device("cpu"),
            )
            output = teo.model_forward(model_name, model, batch)
            self.assertIsInstance(output, dict)
            target_pred = output["target_pred"]
            self.assertEqual(tuple(target_pred.shape), (2, 3))
            if model_name in {"ours_cluster_bank", "ours_cluster_bank_fixed", "ours_extended"}:
                cluster_weights = output["cluster_weights"]
                self.assertEqual(tuple(cluster_weights.shape), (2, 3))
                self.assertTrue(torch.allclose(cluster_weights.sum(dim=-1), torch.ones(2), atol=1e-5))
            else:
                self.assertIsNone(output["cluster_weights"])
            if model_name == "ours_extended":
                local_correction = output["local_correction"]
                delta_omega = output["delta_omega"]
                effective_omega = output["effective_omega"]
                omega = output["omega"]
                self.assertEqual(tuple(local_correction.shape), (2, 3))
                self.assertEqual(tuple(delta_omega.shape), (2, 4, 2))
                self.assertEqual(tuple(effective_omega.shape), (2, 4, 2))
                self.assertEqual(tuple(omega.shape), (2, 2))
                self.assertTrue(torch.all((effective_omega > 0.0) & (effective_omega < math.pi)))
                self.assertTrue(torch.isfinite(local_correction).all())
            elif model_name == "ours_direct_head":
                self.assertIsNone(output["local_correction"])
                self.assertIsNone(output["delta_omega"])
                self.assertIsNone(output["effective_omega"])
            else:
                self.assertIsNone(output["local_correction"])
                self.assertIsNone(output["delta_omega"])
                effective_omega = output["effective_omega"]
                omega = output["omega"]
                self.assertEqual(tuple(effective_omega.shape), (2, 4, 2))
                self.assertTrue(torch.allclose(effective_omega, omega.unsqueeze(1).expand(-1, 4, -1)))
            self.assertTrue(torch.isfinite(target_pred).all())
            self.assertGreater(teo.count_parameters(model), 0)

    def test_ours_fixed_bank_parameters_are_input_invariant(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=8,
            horizon=3,
            ours_latent_groups=4,
            ours_summary_dim=8,
            ours_depth=2,
            ours_kernel_size=3,
            ours_dilations=(1, 2),
            ours_trend_basis_count=4,
            ours_seasonal_mode_count=2,
            ours_transient_basis_count=1,
        )
        model = teo.build_model(
            "ours_fixed_bank",
            input_dim=3,
            target_idx=0,
            config=config,
            device=torch.device("cpu"),
        )
        out_a = teo.model_forward("ours_fixed_bank", model, {"x_hist_full": torch.randn(2, 8, 3)})
        out_b = teo.model_forward("ours_fixed_bank", model, {"x_hist_full": torch.randn(2, 8, 3)})
        for key in ("gamma", "rho", "omega", "beta"):
            self.assertTrue(torch.allclose(out_a[key], out_b[key]))

    def test_ours_fixed_bank_initialization_is_spread(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=8,
            horizon=3,
            ours_latent_groups=4,
            ours_summary_dim=8,
            ours_depth=2,
            ours_kernel_size=3,
            ours_dilations=(1, 2),
            ours_trend_basis_count=4,
            ours_seasonal_mode_count=3,
            ours_transient_basis_count=2,
        )
        model = teo.build_model(
            "ours_fixed_bank",
            input_dim=3,
            target_idx=0,
            config=config,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(model.fixed_gamma)
        gamma = torch.sigmoid(model.fixed_gamma.detach())
        rho = torch.sigmoid(model.fixed_rho.detach())
        omega = math.pi * torch.sigmoid(model.fixed_omega.detach())
        beta = torch.sigmoid(model.fixed_beta.detach())
        self.assertGreater(float(gamma.max() - gamma.min()), 0.05)
        self.assertGreater(float(rho.max() - rho.min()), 0.05)
        self.assertGreater(float(beta.max() - beta.min()), 0.05)
        self.assertTrue(bool(torch.all((gamma >= 0.69) & (gamma <= 0.99))))
        self.assertTrue(bool(torch.all((rho >= 0.69) & (rho <= 0.99))))
        self.assertTrue(bool(torch.all((beta >= 0.69) & (beta <= 0.99))))
        self.assertTrue(bool(torch.all((omega > 0.0) & (omega < math.pi))))

    def test_ours_trend_basis_two_omits_gamma_head(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            horizon=3,
            ours_trend_basis_count=2,
            ours_seasonal_mode_count=2,
            ours_transient_basis_count=1,
        )
        model = teo.build_model(
            "ours",
            input_dim=3,
            target_idx=0,
            config=config,
            device=torch.device("cpu"),
        )
        self.assertIsNone(model.gamma_head)

    def test_compute_model_loss_supports_ours_regularizers(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=8,
            horizon=3,
            ours_latent_groups=4,
            ours_summary_dim=8,
            ours_depth=2,
            ours_kernel_size=3,
            ours_dilations=(1, 2),
            ours_trend_basis_count=4,
            ours_seasonal_mode_count=2,
            ours_transient_basis_count=1,
            ours_coeff_sparsity_weight=0.01,
            ours_seasonal_diversity_weight=0.01,
            ours_router_entropy_weight=0.01,
        )
        model = teo.build_model(
            "ours",
            input_dim=3,
            target_idx=0,
            config=config,
            device=torch.device("cpu"),
        )
        batch = {
            "x_hist_full": torch.randn(2, 8, 3),
            "y": torch.randn(2, 3),
        }
        output = teo.model_forward("ours", model, batch)
        loss = teo.compute_model_loss("ours", output, batch, torch.nn.MSELoss(), config)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss))

    def test_compute_model_loss_uses_omega_pairwise_separation(self) -> None:
        batch = {"y": torch.zeros(1, 3)}
        common_output = {
            "target_pred": torch.zeros(1, 3),
            "router_weights": None,
            "router_enabled": False,
            "seasonal_coeff": None,
            "transient_coeff": None,
        }
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            ours_seasonal_diversity_weight=1.0,
            ours_seasonal_diversity_tau=0.25,
        )

        close_loss = teo.compute_model_loss(
            "ours",
            common_output | {"omega": torch.tensor([[0.10, 0.20]], dtype=torch.float32)},
            batch,
            torch.nn.MSELoss(),
            config,
        )
        far_loss = teo.compute_model_loss(
            "ours",
            common_output | {"omega": torch.tensor([[0.10, 2.80]], dtype=torch.float32)},
            batch,
            torch.nn.MSELoss(),
            config,
        )

        self.assertGreater(float(close_loss), float(far_loss))

    def test_compute_model_loss_seasonal_diversity_tau_controls_decay(self) -> None:
        batch = {"y": torch.zeros(1, 3)}
        output = {
            "target_pred": torch.zeros(1, 3),
            "router_weights": None,
            "router_enabled": False,
            "seasonal_coeff": None,
            "transient_coeff": None,
            "omega": torch.tensor([[0.10, 0.60]], dtype=torch.float32),
        }
        config_small_tau = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            ours_seasonal_diversity_weight=1.0,
            ours_seasonal_diversity_tau=0.10,
        )
        config_large_tau = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            ours_seasonal_diversity_weight=1.0,
            ours_seasonal_diversity_tau=1.00,
        )

        small_tau_loss = teo.compute_model_loss("ours", output, batch, torch.nn.MSELoss(), config_small_tau)
        large_tau_loss = teo.compute_model_loss("ours", output, batch, torch.nn.MSELoss(), config_large_tau)

        self.assertLess(float(small_tau_loss), float(large_tau_loss))

    def test_ours_extended_stage3_fields_are_opt_in(self) -> None:
        config = teo.ExperimentConfig(
            data_path="unused.csv",
            results_dir="unused",
            target_col="AUD",
            lookback=8,
            horizon=3,
            ours_latent_groups=4,
            ours_summary_dim=8,
            ours_depth=2,
            ours_kernel_size=3,
            ours_dilations=(1, 2),
            ours_trend_basis_count=4,
            ours_seasonal_mode_count=2,
            ours_transient_basis_count=1,
        )
        batch = {"x_hist_full": torch.randn(2, 8, 3)}
        base_model = teo.build_model("ours", input_dim=3, target_idx=0, config=config, device=torch.device("cpu"))
        base_output = teo.model_forward("ours", base_model, batch)
        self.assertIsNone(base_output["local_correction"])
        self.assertIsNone(base_output["delta_omega"])
        self.assertTrue(torch.allclose(base_output["effective_omega"], base_output["omega"].unsqueeze(1).expand(-1, 4, -1)))

        extended_model = teo.build_model("ours_extended", input_dim=3, target_idx=0, config=config, device=torch.device("cpu"))
        extended_output = teo.model_forward("ours_extended", extended_model, batch)
        self.assertIsNotNone(extended_output["local_correction"])
        self.assertIsNotNone(extended_output["delta_omega"])
        self.assertFalse(torch.allclose(extended_output["effective_omega"], extended_output["omega"].unsqueeze(1).expand(-1, 4, -1)))


class TrainExchangeReproTests(unittest.TestCase):
    def _make_toy_dataset(self, tmp_path: Path, rows: int = 80) -> Path:
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

    def _common_kwargs(self, dataset_path: Path, seed: int = 123) -> dict[str, object]:
        return dict(
            data_path=str(dataset_path),
            target_col="AUD",
            lookback=8,
            horizon=2,
            batch_size=8,
            epochs=1,
            patience=1,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=seed,
            deterministic=True,
            lstm_proposed_hidden_dim=8,
            lstm_pure_hidden_dim=8,
            gru_hidden_dim=8,
            gru_num_layers=1,
            gru_dropout=0.0,
            stl_hidden_dim=4,
            tcn_hidden_dim=8,
            tcn_kernel_size=3,
            tcn_dropout=0.0,
            patchtst_patch_len=4,
            patchtst_patch_stride=2,
            patchtst_d_model=16,
            patchtst_num_layers=1,
            patchtst_num_heads=4,
            patchtst_ff_dim=32,
            patchtst_dropout=0.0,
            ours_latent_groups=4,
            ours_summary_dim=8,
            ours_depth=2,
            ours_kernel_size=3,
            ours_dilations=(1, 2),
            ours_trend_basis_count=4,
            ours_seasonal_mode_count=2,
            ours_transient_basis_count=1,
        )

    def test_run_experiment_reseeds_per_model_for_order_independent_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path)
            common_kwargs = self._common_kwargs(dataset_path)
            single_run = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "single"),
                    models=("lstm_proposed",),
                    **common_kwargs,
                )
            )
            multi_run = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "multi"),
                    models=("stl_tcn", "lstm_proposed"),
                    **common_kwargs,
                )
            )

            single_metrics = pd.read_csv(single_run / "metrics.csv").set_index("model")
            multi_metrics = pd.read_csv(multi_run / "metrics.csv").set_index("model")
            for column in ["val_mae", "val_rmse", "test_mae", "test_rmse"]:
                self.assertAlmostEqual(
                    float(single_metrics.loc["lstm_proposed", column]),
                    float(multi_metrics.loc["lstm_proposed", column]),
                    places=7,
                )
            self.assertIn("cpls", single_metrics.columns)
            self.assertIn("pareto_optimal", single_metrics.columns)
            self.assertIn("parameter_efficient_2pct", single_metrics.columns)
            summary_dir = single_run / "summary"
            self.assertTrue((summary_dir / "mae_mse_comparison.csv").exists())
            self.assertTrue((summary_dir / "parameter_comparison.csv").exists())
            self.assertTrue((summary_dir / "parameter_efficiency_tolerance.csv").exists())
            self.assertTrue((summary_dir / "epes_comparison.csv").exists())
            self.assertTrue((summary_dir / "cpls_comparison.csv").exists())
            self.assertTrue((summary_dir / "mae_mse_comparison.png").exists())
            self.assertTrue((summary_dir / "parameter_comparison.png").exists())
            self.assertTrue((summary_dir / "epes_comparison.png").exists())
            self.assertTrue((summary_dir / "cpls_comparison.png").exists())
            self.assertTrue((summary_dir / "pairwise_cpl.csv").exists())
            self.assertTrue((summary_dir / "eval_model_predictions.csv").exists())
            self.assertFalse((summary_dir / "all_window_predictions.csv").exists())
            self.assertFalse((single_run.parent / "mae_mse_comparison.csv").exists())
            self.assertFalse((single_run.parent / "latest_run.json").exists())
            self.assertFalse((summary_dir / "model_comparison.csv").exists())
            self.assertFalse((summary_dir / "model_comparison.txt").exists())
            self.assertFalse((summary_dir / "mae_params_comparison.png").exists())
            self.assertFalse((summary_dir / "parameter_efficiency.png").exists())
            self.assertFalse((summary_dir / "epes_cpls_comparison.png").exists())

    def test_run_experiment_keeps_summary_artifacts_isolated_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            shared_results_dir = tmp_path / "shared"

            run_one = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(shared_results_dir),
                    models=("lstm_proposed",),
                    **self._common_kwargs(dataset_path, seed=123),
                )
            )
            run_two = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(shared_results_dir),
                    models=("lstm_proposed",),
                    **self._common_kwargs(dataset_path, seed=124),
                )
            )

            self.assertNotEqual(run_one, run_two)
            self.assertTrue((run_one / "summary" / "mae_mse_comparison.csv").exists())
            self.assertTrue((run_two / "summary" / "mae_mse_comparison.csv").exists())
            self.assertTrue((run_one / "summary" / "eval_model_predictions.csv").exists())
            self.assertTrue((run_two / "summary" / "eval_model_predictions.csv").exists())
            self.assertFalse((run_one.parent / "mae_mse_comparison.csv").exists())
            self.assertFalse((run_one.parent / "latest_run.json").exists())

    def test_run_experiment_exports_eval_predictions_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            run_dir = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "default_export"),
                    models=("lstm_proposed",),
                    **self._common_kwargs(dataset_path),
                )
            )

            export_df = pd.read_csv(run_dir / "summary" / "eval_model_predictions.csv")
            self.assertEqual(sorted(export_df["split"].unique().tolist()), ["test", "val"])
            self.assertFalse((run_dir / "summary" / "all_window_predictions.csv").exists())

    def test_run_experiment_can_disable_eval_prediction_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            run_dir = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "no_eval_export"),
                    models=("lstm_proposed",),
                    export_eval_predictions=False,
                    export_ours_diagnostics=False,
                    **self._common_kwargs(dataset_path),
                )
            )

            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "summary" / "training_history.csv").exists())
            self.assertFalse((run_dir / "summary" / "eval_model_predictions.csv").exists())
            self.assertFalse((run_dir / "summary" / "eval_model_predictions_scaled.csv").exists())

    def test_run_experiment_can_export_all_window_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            run_dir = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "all_export"),
                    models=("lstm_proposed",),
                    export_all_window_predictions=True,
                    **self._common_kwargs(dataset_path),
                )
            )

            all_df = pd.read_csv(run_dir / "summary" / "all_window_predictions.csv")
            self.assertIn("cross_split", all_df["split"].unique().tolist())
            self.assertIn("train", all_df["split"].unique().tolist())
            self.assertIn("val", all_df["split"].unique().tolist())
            self.assertIn("test", all_df["split"].unique().tolist())

    def test_run_experiment_records_dataset_and_efficiency_profile_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            run_dir = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "profile"),
                    dataset_name="toy_exchange",
                    models=("lstm_proposed",),
                    **self._common_kwargs(dataset_path),
                )
            )

            metrics_df = pd.read_csv(run_dir / "metrics.csv")
            self.assertEqual(metrics_df["dataset_name"].iat[0], "toy_exchange")
            self.assertIn("test_inference_ms_per_batch", metrics_df.columns)
            self.assertIn("test_inference_ms_per_sample", metrics_df.columns)
            self.assertIn("test_peak_memory_mb", metrics_df.columns)
            self.assertIn("device_type", metrics_df.columns)
            self.assertTrue((run_dir / "summary" / "latency_memory_comparison.csv").exists())
            self.assertTrue((run_dir / "summary" / "latency_memory_comparison.png").exists())

    def test_run_experiment_supports_dlinear_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            run_dir = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "dlinear"),
                    models=("dlinear",),
                    **self._common_kwargs(dataset_path),
                )
            )

            metrics_df = pd.read_csv(run_dir / "metrics.csv").set_index("model")
            self.assertIn("dlinear", metrics_df.index)
            self.assertGreater(float(metrics_df.loc["dlinear", "params"]), 0.0)

    def test_run_experiment_supports_new_standard_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            for model_name in ("tcn", "gru", "patchtst"):
                config_kwargs = self._common_kwargs(dataset_path)
                if model_name == "tcn":
                    config_kwargs["tcn_hidden_dim"] = 32
                    config_kwargs["tcn_dropout"] = 0.1
                run_dir = teo.run_experiment(
                    teo.ExperimentConfig(
                        results_dir=str(tmp_path / model_name),
                        models=(model_name,),
                        **config_kwargs,
                    )
                )
                metrics_df = pd.read_csv(run_dir / "metrics.csv").set_index("model")
                self.assertIn(model_name, metrics_df.index)
                self.assertGreater(float(metrics_df.loc[model_name, "params"]), 0.0)

    def test_run_experiment_supports_ours_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            for model_name in (
                "ours",
                "ours_no_router",
                "ours_fixed_bank",
                "ours_cluster_bank",
                "ours_cluster_bank_fixed",
                "ours_extended",
                "ours_direct_head",
            ):
                run_dir = teo.run_experiment(
                    teo.ExperimentConfig(
                        results_dir=str(tmp_path / model_name),
                        models=(model_name,),
                        **self._common_kwargs(dataset_path),
                    )
                )
                metrics_df = pd.read_csv(run_dir / "metrics.csv").set_index("model")
                self.assertIn(model_name, metrics_df.index)
                self.assertGreater(float(metrics_df.loc[model_name, "params"]), 0.0)
                self.assertTrue((run_dir / "summary" / "latency_memory_comparison.csv").exists())
                self.assertTrue((run_dir / "summary" / "epes_comparison.csv").exists())

    def test_run_experiment_emits_ours_analysis_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = self._make_toy_dataset(tmp_path, rows=120)
            run_dir = teo.run_experiment(
                teo.ExperimentConfig(
                    results_dir=str(tmp_path / "ours_analysis"),
                    models=("ours", "ours_extended", "ours_direct_head"),
                    **self._common_kwargs(dataset_path),
                )
            )

            summary_dir = run_dir / "summary"
            self.assertTrue((summary_dir / "pareto_accuracy_params.png").exists())
            self.assertTrue((summary_dir / "pareto_accuracy_latency.png").exists())
            self.assertTrue((summary_dir / "ours_branch_usage_raw.csv").exists())
            self.assertTrue((summary_dir / "ours_branch_usage_summary.csv").exists())
            self.assertTrue((summary_dir / "ours_branch_usage.png").exists())
            self.assertTrue((summary_dir / "ours_frequency_raw.csv").exists())
            self.assertTrue((summary_dir / "ours_frequency_summary.csv").exists())
            self.assertTrue((summary_dir / "ours_frequency.png").exists())

            branch_raw_df = pd.read_csv(summary_dir / "ours_branch_usage_raw.csv")
            branch_summary_df = pd.read_csv(summary_dir / "ours_branch_usage_summary.csv")
            frequency_raw_df = pd.read_csv(summary_dir / "ours_frequency_raw.csv")
            frequency_summary_df = pd.read_csv(summary_dir / "ours_frequency_summary.csv")

            self.assertEqual(set(branch_raw_df["model"]), {"ours", "ours_extended"})
            self.assertEqual(set(branch_summary_df["branch"]), {"trend", "seasonal", "transient"})
            self.assertTrue({"mean_router_weight", "mean_abs_coeff", "window_count"}.issubset(branch_summary_df.columns))
            self.assertEqual(set(frequency_raw_df["model"]), {"ours", "ours_extended"})
            self.assertTrue({"omega_mean", "effective_omega_mean", "delta_omega_mean_abs", "window_count"}.issubset(frequency_summary_df.columns))
            extended_delta = frequency_raw_df.loc[frequency_raw_df["model"] == "ours_extended", "delta_omega"].abs()
            self.assertGreater(float(extended_delta.max()), 0.0)
            base_rows = frequency_summary_df.loc[frequency_summary_df["model"] == "ours"]
            self.assertTrue(np.allclose(base_rows["omega_mean"], base_rows["effective_omega_mean"]))


if __name__ == "__main__":
    unittest.main()
