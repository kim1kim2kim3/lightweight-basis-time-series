import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str((ROOT / "scripts").resolve()))

from results_layout import build_run_dir, iter_run_dirs, load_run_metadata


class ResultsLayoutTests(unittest.TestCase):
    def test_build_run_dir_uses_nested_lookback_horizon_structure(self) -> None:
        run_dir = build_run_dir(Path("base"), lookback=56, horizon=14, seed=42, timestamp="20260326_120000")
        self.assertEqual(run_dir, Path("base") / "lb56" / "hz14" / "seed42_20260326_120000")

    def test_load_run_metadata_prefers_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "lb56" / "hz14" / "seed42_20260326_120000"
            run_dir.mkdir(parents=True)
            with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
                json.dump({"lookback": 56, "horizon": 14, "seed": 42, "target_col": "AUD", "dataset_name": "exchange"}, fp)

            metadata = load_run_metadata(run_dir)
            self.assertEqual(metadata["lookback"], 56)
            self.assertEqual(metadata["horizon"], 14)
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["target_col"], "AUD")
            self.assertEqual(metadata["dataset_name"], "exchange")

    def test_iter_run_dirs_discovers_nested_and_legacy_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested_run = root / "lb56" / "hz14" / "seed42_20260326_120000"
            nested_run.mkdir(parents=True)
            (nested_run / "config.json").write_text('{"lookback": 56, "horizon": 14, "seed": 42, "target_col": "AUD"}', encoding="utf-8")
            (nested_run / "metrics.csv").write_text("model,test_mae,test_rmse,params\nm,1,1,1\n", encoding="utf-8")

            legacy_run = root / "lb70_hz21_seed42_20260326_120000"
            legacy_run.mkdir(parents=True)
            (legacy_run / "metrics.csv").write_text("model,test_mae,test_rmse,params\nm,1,1,1\n", encoding="utf-8")

            run_dirs = iter_run_dirs(root)
            self.assertEqual(run_dirs, sorted([nested_run, legacy_run]))

    def test_iter_run_dirs_ignores_nested_artifact_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested_run = root / "lb56" / "hz14" / "seed42_20260326_120000"
            nested_run.mkdir(parents=True)
            (nested_run / "config.json").write_text('{"lookback": 56, "horizon": 14, "seed": 42, "target_col": "AUD"}', encoding="utf-8")
            (nested_run / "metrics.csv").write_text("model,test_mae,test_rmse,params\nm,1,1,1\n", encoding="utf-8")

            plots_dir = nested_run / "plots"
            plots_dir.mkdir()
            (plots_dir / "metrics.csv").write_text("model,test_mae,test_rmse,params\nm,1,1,1\n", encoding="utf-8")

            run_dirs = iter_run_dirs(root)
            self.assertEqual(run_dirs, [nested_run])


if __name__ == "__main__":
    unittest.main()
