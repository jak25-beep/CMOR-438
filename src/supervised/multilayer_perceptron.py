"""
multilayer_perceptron.py
------------------------
MLP (Neural Network) for binary classification using backpropagation.

Supports:
  - Arbitrary hidden layer configuration
  - ReLU hidden activations, Sigmoid output
  - L2 regularisation
  - Mini-batch gradient descent

Usage
-----
from supervised_learning.multilayer_perceptron import MLPClassifier

model = MLPClassifier(hidden_layers=(64, 32), learning_rate=0.01, n_iterations=500)
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.accuracy(X_test, y_test))
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
from _linear_helpers import sigmoid, binary_cross_entropy


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)


class MLPClassifier:
    """
    Multi-Layer Perceptron for binary classification.

    Parameters
    ----------
    hidden_layers : tuple of ints — number of neurons per hidden layer
    learning_rate : float
    n_iterations : int
    batch_size : int | None
    l2 : L2 regularisation coefficient
    verbose : bool
    """

    def __init__(
        self,
        hidden_layers: tuple = (64,),
        learning_rate: float = 0.01,
        n_iterations: int = 500,
        batch_size: int | None = None,
        l2: float = 0.0,
        verbose: bool = False,
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.l2 = l2
        self.verbose = verbose

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.loss_history_: list[float] = []

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_params(self, n_features: int):
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self.weights_ = []
        self.biases_ = []
        for i in range(len(layer_sizes) - 1):
            # He initialisation for ReLU layers
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights_.append(W)
            self.biases_.append(b)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray):
        """Returns list of activations (one per layer including input)."""
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ W + b
            pre_activations.append(z)
            # Hidden layers → ReLU; output layer → Sigmoid
            if i < len(self.weights_) - 1:
                a = _relu(z)
            else:
                a = sigmoid(z)
            activations.append(a)

        return activations, pre_activations

    # ── Backpropagation ───────────────────────────────────────────────────────

    def _backward(self, activations, pre_activations, y_batch):
        n = len(y_batch)
        grad_W = [None] * len(self.weights_)
        grad_b = [None] * len(self.biases_)

        # Output layer delta
        y_col = y_batch.reshape(-1, 1)
        delta = activations[-1] - y_col          # dL/dz for sigmoid + BCE

        for i in reversed(range(len(self.weights_))):
            grad_W[i] = (activations[i].T @ delta) / n
            grad_b[i] = delta.mean(axis=0, keepdims=True)

            if self.l2 > 0:
                grad_W[i] += (self.l2 / n) * self.weights_[i]

            if i > 0:
                delta = (delta @ self.weights_[i].T) * _relu_deriv(pre_activations[i - 1])

        return grad_W, grad_b

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self._init_params(n_features)

        bs = self.batch_size or n_samples

        for epoch in range(self.n_iterations):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, n_samples, bs):
                X_b = X_shuf[start: start + bs]
                y_b = y_shuf[start: start + bs]

                activations, pre_activations = self._forward(X_b)
                grad_W, grad_b = self._backward(activations, pre_activations, y_b)

                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.learning_rate * grad_W[i]
                    self.biases_[i] -= self.learning_rate * grad_b[i]

            # Epoch loss
            _, loss = self._forward_loss(X, y)
            self.loss_history_.append(loss)
            if self.verbose and epoch % 100 == 0:
                print(f"  epoch {epoch:>5d} | loss = {loss:.6f}")

        return self

    def _forward_loss(self, X, y):
        activations, _ = self._forward(X)
        y_prob = activations[-1].ravel()
        loss = binary_cross_entropy(y, y_prob)
        return y_prob, loss

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(np.asarray(X, dtype=float))
        return activations[-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=float)))

    def __repr__(self):
        return (f"MLPClassifier(hidden_layers={self.hidden_layers}, "
                f"lr={self.learning_rate}, iters={self.n_iterations})")
