"""
gradient_descent.py
-------------------
Utility functions and base classes for Gradient Descent optimisation.
Shared by LinearRegression and LogisticRegression.
"""

import numpy as np


class GradientDescentMixin:
    """
    Mixin that adds batch / stochastic / mini-batch gradient descent.

    Subclasses must implement:
        _compute_loss(X, y, weights, bias) -> float
        _compute_gradients(X, y, weights, bias) -> (dw, db)
    """

    def _gd_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> "GradientDescentMixin":
        """
        Run gradient descent to learn weights_ and bias_.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        learning_rate : step size
        n_iterations : number of passes
        batch_size : None → full batch; 1 → SGD; k → mini-batch
        verbose : print loss every 100 steps
        """
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_: list[float] = []

        for i in range(n_iterations):
            # --- select mini-batch ---
            if batch_size is None or batch_size >= n_samples:
                X_batch, y_batch = X, y
            else:
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch, y_batch = X[idx], y[idx]

            dw, db = self._compute_gradients(X_batch, y_batch, self.weights_, self.bias_)
            self.weights_ -= learning_rate * dw
            self.bias_ -= learning_rate * db

            loss = self._compute_loss(X, y, self.weights_, self.bias_)
            self.loss_history_.append(loss)

            if verbose and i % 100 == 0:
                print(f"  iter {i:>5d} | loss = {loss:.6f}")

        return self


def learning_rate_schedule(initial_lr: float, iteration: int, decay: float = 1e-4) -> float:
    """Simple time-based learning rate decay: lr / (1 + decay * t)."""
    return initial_lr / (1.0 + decay * iteration)
