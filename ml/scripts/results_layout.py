from __future__ import annotations

import json
from pathlib import Path


def build_run_dir(base_dir: Path, lookback: int, horizon: int, seed: int, timestamp: str) -> Path:
    return base_dir / f"lb{lookback}" / f"hz{horizon}" / f"seed{seed}_{timestamp}"


def parse_run_name(run_dir: Path) -> tuple[int, int] | None:
    parts = run_dir.name.split("_")
    if len(parts) < 2:
        return None
    if not parts[0].startswith("lb") or not parts[1].startswith("hz"):
        return None

    try:
        lookback = int(parts[0][2:])
        horizon = int(parts[1][2:])
    except ValueError:
        return None
    return lookback, horizon


def load_run_metadata(run_dir: Path) -> dict[str, int | str]:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {
            "lookback": int(payload["lookback"]),
            "horizon": int(payload["horizon"]),
            "seed": int(payload["seed"]),
            "target_col": str(payload["target_col"]),
            "dataset_name": str(payload.get("dataset_name", "")),
        }

    parsed = parse_run_name(run_dir)
    if parsed is None:
        raise ValueError(f"Could not infer run metadata from {run_dir}")
    lookback, horizon = parsed
    return {"lookback": lookback, "horizon": horizon}


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_metrics = (path / "metrics.csv").exists() or (path / "metrics_with_mse.csv").exists()
    if not has_metrics:
        return False
    return (path / "config.json").exists() or parse_run_name(path) is not None


def iter_run_dirs(results_dir: Path) -> list[Path]:
    run_dirs = sorted(path for path in results_dir.rglob("*") if is_run_dir(path))
    if run_dirs:
        return run_dirs

    # Backward-compatibility with older flat layouts.
    return sorted(path for path in results_dir.iterdir() if is_run_dir(path))
