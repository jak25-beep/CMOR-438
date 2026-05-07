"""
dbscan.py
---------
DBSCAN — Density-Based Spatial Clustering of Applications with Noise.

Labels:
  >= 0  →  cluster id
  -1    →  noise

Usage
-----
from unsupervised_learning.dbscan import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X)
print(db.labels_)         # cluster assignments (-1 = noise)
print(db.n_clusters_)     # number of clusters found
print(db.core_samples_)   # indices of core points
"""

import numpy as np
from collections import deque


class DBSCAN:
    """
    DBSCAN Clustering.

    Parameters
    ----------
    eps : float — neighbourhood radius
    min_samples : int — min neighbours to be a core point
    metric : 'euclidean' | 'manhattan'
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        assert metric in ("euclidean", "manhattan")
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: np.ndarray | None = None
        self.core_samples_: np.ndarray | None = None
        self.n_clusters_: int = 0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=float)
        n = len(X)
        dist_matrix = self._pairwise_distances(X)

        neighbors = [
            np.where(dist_matrix[i] <= self.eps)[0].tolist()
            for i in range(n)
        ]
        is_core = np.array([len(nb) >= self.min_samples for nb in neighbors])

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue

            # BFS to expand cluster
            queue = deque([i])
            labels[i] = cluster_id

            while queue:
                point = queue.popleft()
                for nb in neighbors[point]:
                    if labels[nb] == -1:
                        labels[nb] = cluster_id
                        if is_core[nb]:
                            queue.append(nb)

            cluster_id += 1

        self.labels_ = labels
        self.core_samples_ = np.where(is_core)[0]
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    # ── Distance ─────────────────────────────────────────────────────────────

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            diff = X[:, None, :] - X[None, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))
        else:
            diff = X[:, None, :] - X[None, :, :]
            return np.sum(np.abs(diff), axis=-1)

    def __repr__(self):
        return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples})"
