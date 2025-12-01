from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"


def main(args: List[str]) -> None:
    if len(args) != 1:
        print("Usage: python demo_agglomerative.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV and pick numeric columns
    df = pd.read_csv(input_path)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]

    base = os.path.splitext(os.path.basename(input_path))[0]

    # Run agglomerative clustering for k = 2,3,4,5
    metrics_summary = []

    for k in (2, 3, 4, 5):
        print(f"\n=== Running agglomerative clustering with k = {k} ===")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="agglomerative",  # New algorithm
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_agglo_k{k}.csv"),
            random_state=42,
            compute_elbow=False,
        )

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_agglo_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # Summarise metrics across k
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_agglo_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Plot silhouette scores
    if "silhouette" in metrics_df.columns and metrics_df["silhouette"].notnull().any():
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"])
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different k (Agglomerative)")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_agglo_silhouette.png")
        plt.savefig(stats_path, dpi=150)
        plt.close()

    print("\nDemo completed.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
