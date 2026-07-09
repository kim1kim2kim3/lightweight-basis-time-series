from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from train_exchange_once import (
    DEFAULT_BENCHMARK_MODELS,
    OURS_MODEL_NAMES,
    ExperimentConfig,
    parse_models,
    run_experiment,
)


EXPERIMENT_FIELD_NAMES = {field.name for field in fields(ExperimentConfig)}
MODEL_PRESET_DIR = Path(__file__).resolve().parents[2] / "configs" / "model_presets"


def parse_seed_list(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def normalize_models(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return tuple(DEFAULT_BENCHMARK_MODELS)
    if isinstance(raw, str):
        return parse_models(raw)
    if isinstance(raw, tuple):
        return parse_models(",".join(str(item) for item in raw))
    if isinstance(raw, list):
        return parse_models(",".join(str(item) for item in raw))
    raise TypeError(f"Unsupported models value: {raw!r}")


def normalize_horizons(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Each dataset entry must define a non-empty horizons list.")
    horizons = tuple(int(item) for item in raw)
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be positive integers.")
    return horizons


def extract_experiment_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key in EXPERIMENT_FIELD_NAMES}


def model_preset_name(model_name: str) -> str | None:
    if model_name in OURS_MODEL_NAMES:
        return "ours"
    if model_name in {"dlinear", "patchtst"}:
        return model_name
    return None


def load_model_presets(models: tuple[str, ...], preset_dir: Path = MODEL_PRESET_DIR) -> dict[str, Any]:
    preset_kwargs: dict[str, Any] = {}
    loaded_presets: set[str] = set()
    for model_name in models:
        preset_name = model_preset_name(model_name)
        if preset_name is None or preset_name in loaded_presets:
            continue

        preset_path = preset_dir / f"{preset_name}.json"
        with preset_path.open("r", encoding="utf-8") as fp:
            raw_preset = json.load(fp)
        if not isinstance(raw_preset, dict):
            raise TypeError(f"Model preset must be a JSON object: {preset_path}")

        unknown_fields = sorted(set(raw_preset) - EXPERIMENT_FIELD_NAMES)
        if unknown_fields:
            raise ValueError(f"Unknown ExperimentConfig fields in {preset_path}: {unknown_fields}")
        preset_kwargs.update(raw_preset)
        loaded_presets.add(preset_name)
    return preset_kwargs


def merge_experiment_kwargs_for_models(models: tuple[str, ...], *overrides: dict[str, Any]) -> dict[str, Any]:
    merged = load_model_presets(models)
    for override in overrides:
        merged.update(override)
    return merged


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_results = script_dir.parent.parent / "runs" / "benchmark_manifest"

    parser = argparse.ArgumentParser(description="Run multi-seed benchmark sweeps from a JSON manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--seeds", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_root = args.manifest_path.resolve().parent
    with args.manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    defaults = extract_experiment_kwargs(manifest.get("defaults", {}))
    default_models = normalize_models(manifest.get("defaults", {}).get("models"))
    default_seeds = (
        parse_seed_list(args.seeds)
        if args.seeds is not None
        else tuple(int(seed) for seed in manifest.get("seeds", [defaults.get("seed", 42)]))
    )
    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("Manifest must contain a non-empty datasets list.")

    for dataset_entry in datasets:
        if "name" not in dataset_entry:
            raise ValueError("Each dataset entry must include a 'name'.")
        if "data_path" not in dataset_entry or "target_col" not in dataset_entry:
            raise ValueError("Each dataset entry must include 'data_path' and 'target_col'.")

        dataset_name = str(dataset_entry["name"])
        dataset_models = normalize_models(dataset_entry.get("models", default_models))
        dataset_defaults = merge_experiment_kwargs_for_models(
            dataset_models,
            defaults,
            extract_experiment_kwargs(dataset_entry),
        )
        data_path = Path(str(dataset_entry["data_path"]))
        if not data_path.is_absolute():
            data_path = (manifest_root / data_path).resolve()
        dataset_defaults["dataset_name"] = dataset_name
        dataset_defaults["data_path"] = str(data_path)
        dataset_defaults["target_col"] = str(dataset_entry["target_col"])
        dataset_defaults["results_dir"] = str(args.results_dir / dataset_name)
        dataset_defaults["models"] = dataset_models
        horizons = normalize_horizons(dataset_entry["horizons"])
        dataset_seeds = tuple(int(seed) for seed in dataset_entry.get("seeds", default_seeds))

        for seed in dataset_seeds:
            for horizon in horizons:
                config_kwargs = dict(dataset_defaults)
                config_kwargs["seed"] = int(seed)
                config_kwargs["horizon"] = int(horizon)
                config = ExperimentConfig(**config_kwargs)
                run_experiment(config)


if __name__ == "__main__":
    main()
