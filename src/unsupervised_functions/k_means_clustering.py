"""
k_means_clustering.py
---------------------
K-Means Clustering with 'random' and 'k-means++' initialisation.

Usage
-----
from unsupervised_learning.k_means_clustering import KMeans

km = KMeans(k=3, init="k-means++", n_init=10)
km.fit(X)
print(km.labels_)
print(km.inertia_)
print(km.predict(X_new))
"""

import numpy as np


class KMeans:
    """
    K-Means Clustering.

    Parameters
    ----------
    k : int — number of clusters
    init : 'random' | 'k-means++'
    n_init : number of random restarts (best inertia is kept)
    max_iter : max iterations per run
    tol : convergence tolerance (centroid shift)
    random_state : int | None
    """

    def __init__(
        self,
        k: int = 3,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        assert init in ("random", "k-means++"), "init must be 'random' or 'k-means++'"
        self.k = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        best_centroids, best_labels, best_inertia = None, None, np.inf

        for trial in range(self.n_init):
            seed = rng.randint(0, 2**31)
            centroids = self._init_centroids(X, np.random.RandomState(seed))
            labels, centroids, inertia, n_iter = self._run(X, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                self.n_iter_ = n_iter

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        n_samples = len(X)
        if self.init == "random":
            idx = rng.choice(n_samples, self.k, replace=False)
            return X[idx].copy()

        # k-means++ initialisation
        centroids = [X[rng.randint(0, n_samples)]]
        for _ in range(1, self.k):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
            probs = dists / dists.sum()
            cumprobs = np.cumsum(probs)
            r = rng.rand()
            idx = np.searchsorted(cumprobs, r)
            centroids.append(X[idx])
        return np.array(centroids)

    def _run(self, X: np.ndarray, centroids: np.ndarray):
        for i in range(self.max_iter):
            labels = self._assign(X, centroids)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
                for j in range(self.k)
            ])
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift < self.tol:
                break

        labels = self._assign(X, centroids)
        inertia = self._inertia(X, labels, centroids)
        return labels, centroids, inertia, i + 1

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def _inertia(self, X, labels, centroids) -> float:
        return float(sum(
            np.sum((X[labels == j] - centroids[j]) ** 2)
            for j in range(self.k)
            if (labels == j).any()
        ))

    # ── Predict / Transform ───────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(np.asarray(X, dtype=float), self.centroids_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances from each sample to each centroid."""
        X = np.asarray(X, dtype=float)
        return np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def silhouette_score(self, X: np.ndarray) -> float:
        """Compute mean silhouette coefficient (slow but exact)."""
        X = np.asarray(X, dtype=float)
        labels = self.labels_
        n = len(X)
        scores = np.zeros(n)
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            if same.sum() == 0:
                scores[i] = 0.0
                continue
            a = np.mean(np.linalg.norm(X[same] - X[i], axis=1))
            b = min(
                np.mean(np.linalg.norm(X[labels == j] - X[i], axis=1))
                for j in range(self.k)
                if j != labels[i] and (labels == j).any()
            )
            scores[i] = (b - a) / max(a, b)
        return float(np.mean(scores))

    def __repr__(self):
        return f"KMeans(k={self.k}, init='{self.init}', n_init={self.n_init})"
