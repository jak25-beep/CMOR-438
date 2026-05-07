"""
k_nearest_neighbors.py
-----------------------
Implements KNNClassifier and KNNRegressor (Lazy Learning).

Both classes use the same distance-based core; they only differ
in how the k neighbours' labels are aggregated.

Usage
-----
from supervised_learning.k_nearest_neighbors import KNNClassifier, KNNRegressor

clf = KNNClassifier(k=5, metric="euclidean")
clf.fit(X_train, y_train)
print(clf.predict(X_test))

reg = KNNRegressor(k=5)
reg.fit(X_train, y_train)
print(reg.predict(X_test))
"""

import numpy as np
from collections import Counter


# ── Distance Functions ────────────────────────────────────────────────────────

def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised Euclidean distance from row-vector a to matrix b."""
    return np.sqrt(np.sum((b - a) ** 2, axis=1))


def _manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(b - a), axis=1)


def _minkowski(a: np.ndarray, b: np.ndarray, p: int = 3) -> np.ndarray:
    return np.sum(np.abs(b - a) ** p, axis=1) ** (1.0 / p)


_METRICS = {
    "euclidean": _euclidean,
    "manhattan": _manhattan,
    "minkowski": _minkowski,
}


# ── Shared Base ───────────────────────────────────────────────────────────────

class _KNNBase:
    def __init__(self, k: int = 5, metric: str = "euclidean"):
        assert k >= 1, "k must be >= 1"
        assert metric in _METRICS, f"metric must be one of {list(_METRICS)}"
        self.k = k
        self.metric = metric
        self._dist_fn = _METRICS[metric]
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_KNNBase":
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def _get_neighbors(self, x: np.ndarray) -> np.ndarray:
        """Return indices of the k nearest neighbours for a single sample."""
        distances = self._dist_fn(x, self.X_train_)
        return np.argsort(distances)[: self.k]


# ── Classifier ────────────────────────────────────────────────────────────────

class KNNClassifier(_KNNBase):
    """
    K-Nearest Neighbours Classifier.

    Parameters
    ----------
    k : int — number of neighbours
    metric : 'euclidean' | 'manhattan' | 'minkowski'
    weights : 'uniform' | 'distance'  — how to weight neighbour votes
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        super().__init__(k, metric)
        assert weights in ("uniform", "distance"), "weights must be 'uniform' or 'distance'"
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray):
        neighbor_idx = self._get_neighbors(x)
        neighbor_labels = self.y_train_[neighbor_idx]

        if self.weights == "uniform":
            return Counter(neighbor_labels).most_common(1)[0][0]
        else:
            distances = self._dist_fn(x, self.X_train_[neighbor_idx])
            # Avoid division by zero
            inv_dist = 1.0 / (distances + 1e-10)
            label_weights: dict = {}
            for label, w in zip(neighbor_labels, inv_dist):
                label_weights[label] = label_weights.get(label, 0.0) + w
            return max(label_weights, key=label_weights.get)

    def predict_proba(self, X: np.ndarray) -> dict:
        """Return class probability dict for each sample."""
        X = np.asarray(X, dtype=float)
        classes = np.unique(self.y_train_)
        proba = []
        for x in X:
            neighbor_labels = self.y_train_[self._get_neighbors(x)]
            counts = Counter(neighbor_labels)
            proba.append({c: counts.get(c, 0) / self.k for c in classes})
        return proba

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self):
        return f"KNNClassifier(k={self.k}, metric='{self.metric}', weights='{self.weights}')"


# ── Regressor ─────────────────────────────────────────────────────────────────

class KNNRegressor(_KNNBase):
    """
    K-Nearest Neighbours Regressor.

    Parameters
    ----------
    k : int
    metric : 'euclidean' | 'manhattan' | 'minkowski'
    weights : 'uniform' | 'distance'
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "uniform"):
        super().__init__(k, metric)
        assert weights in ("uniform", "distance")
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> float:
        neighbor_idx = self._get_neighbors(x)
        neighbor_vals = self.y_train_[neighbor_idx].astype(float)

        if self.weights == "uniform":
            return float(np.mean(neighbor_vals))
        else:
            distances = self._dist_fn(x, self.X_train_[neighbor_idx])
            inv_dist = 1.0 / (distances + 1e-10)
            return float(np.dot(inv_dist, neighbor_vals) / inv_dist.sum())

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def __repr__(self):
        return f"KNNRegressor(k={self.k}, metric='{self.metric}', weights='{self.weights}')"
