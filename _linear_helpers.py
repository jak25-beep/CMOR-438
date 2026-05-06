"""
_linear_helpers.py
------------------
Internal helper functions for linear models:
  - cost / loss functions
  - OLS normal equation solver
  - Ridge penalty
  - evaluation metrics
"""

import numpy as np


# ── Loss / Cost Functions ─────────────────────────────────────────────────────

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE = (1/n) * SUM(y - yhat)^2"""
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-9) -> float:
    """Log-loss for binary classification."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


# ── OLS Normal Equation ───────────────────────────────────────────────────────

def ols_normal_equation(X: np.ndarray, y: np.ndarray):
    """
    Solve w = (X^T)X)^(-1) (X^T)y via pseudo-inverse (numerically stable).

    Returns
    -------
    weights : (n_features,)
    bias    : float
    """
    # Add bias column
    X_b = np.column_stack([np.ones(len(X)), X])
    params = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return params[1:], params[0]


def ridge_normal_equation(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """
    Ridge regression closed form: w = (X^(T)X + alpha*I)^(-1) X^(T)y
    (bias term excluded from regularisation).
    """
    X_b = np.column_stack([np.ones(len(X)), X])
    n_features = X_b.shape[1]
    reg = alpha * np.eye(n_features)
    reg[0, 0] = 0.0  # don't regularise bias
    params = np.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)
    return params[1:], params[0]


# ── Activations ───────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def linear(z: np.ndarray) -> np.ndarray:
    return z
