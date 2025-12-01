from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"


def main(args: List[str]) -> None:
    if len(args) != 1:
        print("Usage: python simulated_clustering.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data and pick numeric columns
    df = pd.read_csv(input_path)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        print("Error: input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]

    base = os.path.splitext(os.path.basename(input_path))[0]

    metrics_summary = []
    cluster_results = {}

    # Run K-means for multiple k values
    for k in (2, 3, 4, 5):
        print(f"\n=== Running K-means with k = {k} ===")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=42,
            compute_elbow=False,
        )

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)
        cluster_results[k] = result

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # Summarise metrics
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Determine the best k by silhouette score
    if "silhouette" in metrics_df.columns and metrics_df["silhouette"].notnull().any():
        best_row = metrics_df.loc[metrics_df["silhouette"].idxmax()]
        best_k = best_row["k"]
        print(f"\nBest k based on silhouette score: {best_k}")
    else:
        best_k = None

    # Create a single figure with subplots for all k
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, k in enumerate((2, 3, 4, 5)):
        result = cluster_results[k]
        X_plot = result["data"][feature_cols].to_numpy(dtype=float)
        labels = result["labels"]
        ax = axes[i]
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap="tab10", alpha=0.8)
        ax.set_title(f"k = {k}" + (" (best)" if k == best_k else ""))
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
    fig.colorbar(scatter, ax=axes.tolist(), label="Cluster label")
    fig.suptitle("Clusterings for different k", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined_plot_path = os.path.join(OUTPUT_DIR, f"{base}_all_k_comparison.png")
    fig.savefig(combined_plot_path, dpi=150)
    plt.close(fig)

    print("\nAnalysis completed.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
