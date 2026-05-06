"""
perceptron.py
-------------
Classic single-layer Perceptron for binary classification.

The Perceptron update rule:
    if y_pred != y_true:
        w += lr * y_true * x
        b += lr * y_true

Usage
-----
from supervised_learning.perceptron import Perceptron

model = Perceptron(learning_rate=0.1, n_iterations=200)
model.fit(X_train, y_train)
print(model.predict(X_test))          # {-1, 1}
print(model.accuracy(X_test, y_test))
"""

import numpy as np


class Perceptron:
    """
    Single-layer Perceptron (Rosenblatt 1958).

    Labels must be in {-1, +1} or {0, 1} — automatically mapped internally.

    Parameters
    ----------
    learning_rate : float
    n_iterations : int
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.errors_per_epoch_: list[int] = []

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Map {0,1} → {-1,+1} for classic Perceptron rule
        y_mapped = np.where(y <= 0, -1.0, 1.0)

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.errors_per_epoch_ = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y_mapped):
                y_hat = self._activation(np.dot(xi, self.weights_) + self.bias_)
                if yi != y_hat:
                    update = self.learning_rate * yi
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors += 1
            self.errors_per_epoch_.append(errors)
            if errors == 0:          # converged
                break

        return self

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _activation(z: float) -> float:
        return 1.0 if z >= 0.0 else -1.0

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        raw = X @ self.weights_ + self.bias_
        return np.where(raw >= 0.0, 1, -1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        y_mapped = np.where(y <= 0, -1.0, 1.0)
        return float(np.mean(self.predict(X) == y_mapped))

    def __repr__(self):
        return f"Perceptron(lr={self.learning_rate}, iters={self.n_iterations})"
