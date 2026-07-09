from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_test_timeline(main_run_dir: Path, patchtst_run_dir: Path | None = None):
    summary_dir = main_run_dir / "summary"
    main_csv_path = summary_dir / "eval_model_predictions.csv"
    
    if not main_csv_path.exists():
        print(f"Error: {main_csv_path} not found.")
        return

    df = pd.read_csv(main_csv_path)
    df = df[df["split"] == "test"].copy()
    
    if patchtst_run_dir:
        patchtst_csv_path = patchtst_run_dir / "summary" / "eval_model_predictions.csv"
        if patchtst_csv_path.exists():
            patch_df = pd.read_csv(patchtst_csv_path)
            patch_df = patch_df[(patch_df["split"] == "test") & (patch_df["model"] == "patchtst")].copy()
            if not patch_df.empty:
                df = pd.concat([df, patch_df], ignore_index=True)
            else:
                print("Warning: PatchTST data not found in secondary CSV.")
        else:
            print(f"Warning: {patchtst_csv_path} not found.")

    if df.empty:
        print("Error: No test split data found.")
        return

    df = df.sort_values(["model", "window_end_index"])
    
    models_to_plot = ["ours", "dlinear", "patchtst"]
    available_models = df["model"].unique()
    models = [m for m in models_to_plot if m in available_models]
    
    plt.figure(figsize=(20, 8))
    
    # Plot actual values from any available model
    first_model = models[0]
    model_df = df[df["model"] == first_model]
    indices = model_df["window_end_index"].values
    actual = model_df["actual_t+1"].values
    
    plt.plot(indices, actual, label="Actual", color="black", alpha=0.4, linewidth=1)
    
    colors = {"ours": "#c8553d", "dlinear": "#355070", "patchtst": "#2a9d8f"}
    
    for model in models:
        model_df = df[df["model"] == model]
        if len(model_df) != len(indices):
            print(f"Warning: Model {model} has {len(model_df)} points, but Actual has {len(indices)}. Alignment might be off.")
            # Simple alignment by window_end_index
            merged = pd.DataFrame({"window_end_index": indices}).merge(model_df, on="window_end_index", how="left")
            pred = merged["pred_t+1"].values
        else:
            pred = model_df["pred_t+1"].values
            
        plt.plot(indices, pred, label=model.upper(), color=colors.get(model), alpha=0.8, linewidth=1.5)
    
    plt.title(f"Test Timeline Reconstruction (t+1) - ETTh1 hz720")
    plt.xlabel("Time Index")
    plt.ylabel("Target Value") # Added missing y-axis label
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = summary_dir / "test_timeline_reconstructed.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved combined plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run-dir", type=Path, required=True)
    parser.add_argument("--patchtst-run-dir", type=Path, default=None)
    args = parser.parse_args()
    plot_test_timeline(args.main_run_dir, args.patchtst_run_dir)
