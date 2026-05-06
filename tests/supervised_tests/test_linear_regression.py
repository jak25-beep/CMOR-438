"""
test_linear_regression.py
=========================
Unit tests for the LinearRegression class, which supports three fitting
strategies: Ordinary Least Squares (OLS), Ridge regression, and Gradient
Descent (GD).

Run all tests:
    pytest test_linear_regression.py -v

Each test is self-contained and generates its own synthetic data via
make_data(), so no external files are required.
"""

import numpy as np
import sys, os

# Make the algorithm and shared helpers importable from this file's location
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from linear_regression import LinearRegression


# ── Shared synthetic data factory ─────────────────────────────────────────────

def make_data(n=100, noise=0.0, seed=42):
    """
    Generate a simple linear regression dataset.

    The true relationship is:
        y = 2*x0 - 1*x1 + 0.5*x2 + 1.0 + noise

    Parameters
    ----------
    n     : number of samples
    noise : standard deviation of Gaussian noise added to y
    seed  : random seed for reproducibility

    Returns
    -------
    X : (n, 3) float array of features
    y : (n,)   float array of targets
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + 1.0 + rng.randn(n) * noise
    return X, y


# ── OLS tests ─────────────────────────────────────────────────────────────────

def test_ols_perfect_fit():
    """
    On noise-free data OLS should find an exact solution.
    R² must be essentially 1.0 because the data lies exactly on a hyperplane.
    """
    X, y = make_data(noise=0.0)
    m = LinearRegression(method="ols").fit(X, y)
    assert m.score(X, y) > 0.9999, "OLS should achieve near-perfect R² on noiseless data"

def test_ols_recovers_weights():
    """
    The closed-form normal equation is exact, so the recovered weights must
    match the true weights [2.0, -1.0, 0.5] and bias 1.0 to floating-point
    precision.
    """
    X, y = make_data(noise=0.0)
    m = LinearRegression(method="ols").fit(X, y)
    assert np.allclose(m.weights_, [2.0, -1.0, 0.5], atol=1e-6),         "OLS weights should match the ground-truth coefficients on noiseless data"
    assert abs(m.bias_ - 1.0) < 1e-6,         "OLS bias should match the ground-truth intercept on noiseless data"

def test_ols_noisy_r2_positive():
    """
    Even on noisy data the model should explain more variance than a trivial
    mean predictor (R² > 0), and given the strong signal here, comfortably
    above 0.5.
    """
    X, y = make_data(noise=1.0)
    m = LinearRegression(method="ols").fit(X, y)
    assert m.score(X, y) > 0.5,         "OLS should achieve R² > 0.5 even with moderate noise given a strong signal"

def test_ols_predict_shape():
    """
    predict() must return an array with the same number of elements as there
    are input samples — shape (n,).
    """
    X, y = make_data()
    m = LinearRegression(method="ols").fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output shape must match the number of input samples"


# ── Ridge tests ───────────────────────────────────────────────────────────────

def test_ridge_perfect_fit():
    """
    With a very small regularisation constant (alpha≈0) Ridge approaches OLS.
    On noiseless data the fit should still be near-perfect.
    """
    X, y = make_data(noise=0.0)
    m = LinearRegression(method="ridge", alpha=0.0001).fit(X, y)
    assert m.score(X, y) > 0.999,         "Ridge with near-zero alpha should approach OLS quality on noiseless data"

def test_ridge_high_alpha_shrinks_weights():
    """
    The fundamental property of Ridge regularisation: higher alpha forces
    weights closer to zero. Comparing L2 norms confirms the penalty is active.
    """
    X, y = make_data(noise=0.5)
    m_low  = LinearRegression(method="ridge", alpha=0.001).fit(X, y)
    m_high = LinearRegression(method="ridge", alpha=1000.0).fit(X, y)
    assert np.linalg.norm(m_high.weights_) < np.linalg.norm(m_low.weights_),         "Higher alpha should produce smaller weight norms (L2 shrinkage)"

def test_ridge_mse_finite():
    """
    The MSE must always be a finite number. This guards against numerical
    problems such as matrix inversion instability.
    """
    X, y = make_data(noise=1.0)
    m = LinearRegression(method="ridge").fit(X, y)
    assert np.isfinite(m.mse(X, y)), "Ridge MSE must be a finite number"


# ── Gradient Descent tests ────────────────────────────────────────────────────

def test_gd_loss_decreases():
    """
    Gradient descent must reduce the training loss over time.
    Comparing the first and last recorded loss values confirms the optimiser
    is moving in the right direction.
    """
    X, y = make_data(noise=0.5)
    m = LinearRegression(method="gd", learning_rate=0.01, n_iterations=500).fit(X, y)
    assert m.loss_history_[0] > m.loss_history_[-1],         "GD loss must be strictly lower at the end than at the start of training"

def test_gd_reasonable_r2():
    """
    Given enough iterations and a suitable learning rate, GD should converge
    to a solution whose quality is comparable to OLS on the same data.
    """
    X, y = make_data(noise=0.5)
    m = LinearRegression(method="gd", learning_rate=0.05, n_iterations=2000).fit(X, y)
    assert m.score(X, y) > 0.5,         "GD should achieve R² > 0.5 after sufficient iterations with a reasonable learning rate"

def test_gd_minibatch_runs():
    """
    Mini-batch GD (batch_size=32) must complete without error and produce a
    fitted weights array — confirming the batching logic doesn't break the
    update loop.
    """
    X, y = make_data(n=200, noise=0.5)
    m = LinearRegression(method="gd", learning_rate=0.01,
                         n_iterations=300, batch_size=32).fit(X, y)
    assert m.weights_ is not None,         "Mini-batch GD must populate weights_ after training"

def test_gd_loss_history_length():
    """
    One loss value is recorded per iteration, so loss_history_ must have
    exactly n_iterations entries.
    """
    X, y = make_data()
    m = LinearRegression(method="gd", n_iterations=100).fit(X, y)
    assert len(m.loss_history_) == 100,         "loss_history_ must contain exactly one entry per training iteration"


# ── Edge case tests ───────────────────────────────────────────────────────────

def test_single_feature():
    """
    Verify that the model works correctly with a single input feature.
    The data is y = 2x, which is exactly linear, so R² should be essentially 1.
    """
    X = np.array([[1],[2],[3],[4],[5]], dtype=float)
    y = np.array([2,4,6,8,10], dtype=float)
    m = LinearRegression(method="ols").fit(X, y)
    assert m.score(X, y) > 0.999,         "OLS must fit a perfectly linear single-feature dataset with R² ≈ 1"

def test_repr():
    """
    repr() must include the chosen method name so the model can be identified
    in logs and interactive sessions.
    """
    m = LinearRegression(method="ridge")
    assert "ridge" in repr(m), "repr() must mention the fitting method"
