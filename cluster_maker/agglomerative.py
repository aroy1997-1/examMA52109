from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical agglomerative clustering.

    Returns labels and centroids so that interface.py can treat it like kmeans.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : str, default 'ward'
        Linkage criterion ('ward', 'complete', 'average', 'single').

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    centroids : ndarray of shape (k, n_features)
        Approximate centroids of each cluster (mean of points in cluster)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)

    # Compute centroids as mean of points in each cluster
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    for cluster_id in range(k):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) > 0:
            centroids[cluster_id] = cluster_points.mean(axis=0)
        else:
            centroids[cluster_id] = np.nan  # unlikely, but safe

    return labels, centroids
