from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from run_benchmark_manifest import (
    extract_experiment_kwargs,
    merge_experiment_kwargs_for_models,
    normalize_horizons,
    normalize_models,
    parse_seed_list,
)
from train_exchange_once import ExperimentConfig, run_experiment


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-")
    return sanitized or "variant"


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_results = script_dir.parent.parent / "runs" / "ours_sweep"

    parser = argparse.ArgumentParser(description="Run Ours ablations and sweeps from a JSON manifest.")
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
    sweeps = manifest.get("sweeps", [])
    if not datasets:
        raise ValueError("Manifest must contain a non-empty datasets list.")
    if not sweeps:
        raise ValueError("Manifest must contain a non-empty sweeps list.")

    for dataset_entry in datasets:
        if "name" not in dataset_entry:
            raise ValueError("Each dataset entry must include a 'name'.")
        if "data_path" not in dataset_entry or "target_col" not in dataset_entry:
            raise ValueError("Each dataset entry must include 'data_path' and 'target_col'.")

        dataset_name = str(dataset_entry["name"])
        dataset_models = normalize_models(dataset_entry.get("models", default_models))
        dataset_overrides = extract_experiment_kwargs(dataset_entry)
        data_path = Path(str(dataset_entry["data_path"]))
        if not data_path.is_absolute():
            data_path = (manifest_root / data_path).resolve()
        horizons = normalize_horizons(dataset_entry["horizons"])
        dataset_seeds = tuple(int(seed) for seed in dataset_entry.get("seeds", default_seeds))

        for sweep_entry in sweeps:
            sweep_name = str(sweep_entry.get("name", "")).strip()
            if not sweep_name:
                raise ValueError("Each sweep entry must include a non-empty 'name'.")
            variants = sweep_entry.get("variants", [])
            if not variants:
                raise ValueError(f"Sweep '{sweep_name}' must include a non-empty variants list.")

            for variant_entry in variants:
                label = str(variant_entry.get("label", "")).strip()
                if not label:
                    raise ValueError(f"Sweep '{sweep_name}' contains a variant without a label.")
                overrides_raw = variant_entry.get("overrides", {})
                if not isinstance(overrides_raw, dict):
                    raise TypeError(f"Sweep '{sweep_name}' variant '{label}' overrides must be a JSON object.")
                overrides = extract_experiment_kwargs(overrides_raw)
                variant_models = normalize_models(overrides_raw.get("models", dataset_models))
                variant_defaults = merge_experiment_kwargs_for_models(
                    variant_models,
                    defaults,
                    dataset_overrides,
                )
                variant_defaults["dataset_name"] = dataset_name
                variant_defaults["data_path"] = str(data_path)
                variant_defaults["target_col"] = str(dataset_entry["target_col"])
                variant_base_dir = args.results_dir / dataset_name / sweep_name / sanitize_label(label)

                for seed in dataset_seeds:
                    for horizon in horizons:
                        config_kwargs = dict(variant_defaults)
                        config_kwargs.update(overrides)
                        config_kwargs["seed"] = int(seed)
                        config_kwargs["horizon"] = int(horizon)
                        config_kwargs["results_dir"] = str(variant_base_dir)
                        config_kwargs["models"] = variant_models
                        config = ExperimentConfig(**config_kwargs)
                        run_dir = run_experiment(config)
                        with (run_dir / "sweep_metadata.json").open("w", encoding="utf-8") as fp:
                            json.dump(
                                {
                                    "dataset_name": dataset_name,
                                    "sweep_name": sweep_name,
                                    "sweep_value": label,
                                    "variant_dir_name": sanitize_label(label),
                                    "variant_overrides": overrides_raw,
                                },
                                fp,
                                ensure_ascii=False,
                                indent=2,
                            )


if __name__ == "__main__":
    main()
