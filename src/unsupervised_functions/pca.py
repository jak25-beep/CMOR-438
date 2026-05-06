"""
pca.py
------
Principal Component Analysis (PCA) via eigenvalue decomposition of the
covariance matrix.

Usage
-----
from unsupervised_learning.pca import PCA

pca = PCA(n_components=2)
pca.fit(X_train)
X_reduced = pca.transform(X_train)
X_reconstructed = pca.inverse_transform(X_reduced)
print(pca.explained_variance_ratio_)
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis.

    Finds the directions of maximum variance via eigendecomposition of
    the sample covariance matrix.

    Parameters
    ----------
    n_components : int | float | None
        int   → keep this many components
        float → keep enough components to explain this fraction of variance
        None  → keep all components
    """

    def __init__(self, n_components: int | float | None = None):
        self.n_components = n_components

        self.components_: np.ndarray | None = None          # (k, n_features)
        self.explained_variance_: np.ndarray | None = None  # eigenvalues
        self.explained_variance_ratio_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_components_: int = 0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # Covariance matrix (unbiased)
        cov = np.cov(X_c, rowvar=False)

        # Eigendecomposition (eigh for symmetric matrices — numerically stable)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]   # columns are eigenvectors

        # Determine k
        k = self._resolve_n_components(eigenvalues)
        self.n_components_ = k

        self.components_ = eigenvectors[:, :k].T   # (k, n_features)
        self.explained_variance_ = eigenvalues[:k]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:k] / (total_var + 1e-10)

        return self

    def _resolve_n_components(self, eigenvalues: np.ndarray) -> int:
        nc = self.n_components
        if nc is None:
            return len(eigenvalues)
        if isinstance(nc, float):
            # Find minimum k such that cumulative explained variance >= nc
            cumvar = np.cumsum(eigenvalues) / (eigenvalues.sum() + 1e-10)
            return int(np.searchsorted(cumvar, nc) + 1)
        return int(nc)

    # ── Transform / Inverse ───────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto principal components. Returns (n_samples, k)."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct approximate original X from reduced representation."""
        return np.asarray(X_reduced, dtype=float) @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error."""
        X = np.asarray(X, dtype=float)
        X_rec = self.inverse_transform(self.transform(X))
        return float(np.mean((X - X_rec) ** 2))

    def __repr__(self):
        return f"PCA(n_components={self.n_components})"
