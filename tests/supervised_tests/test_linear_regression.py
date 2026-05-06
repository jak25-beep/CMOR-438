"""
Unit tests for LinearRegression.
Run with: pytest test_linear_regression.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from linear_regression import LinearRegression


def make_data(n=100, noise=0.0, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + 1.0 + rng.randn(n) * noise
    return X, y


# ── OLS ───────────────────────────────────────────────────────────────────────

def test_ols_perfect_fit():
    X, y = make_data(noise=0.0)
    m = LinearRegression(method="ols").fit(X, y)
    assert m.score(X, y) > 0.9999

def test_ols_recovers_weights():
    X, y = make_data(noise=0.0)
    m = LinearRegression(method="ols").fit(X, y)
    assert np.allclose(m.weights_, [2.0, -1.0, 0.5], atol=1e-6)
    assert abs(m.bias_ - 1.0) < 1e-6

def test_ols_noisy_r2_positive():
    X, y = make_data(noise=1.0)
    m = LinearRegression(method="ols").fit(X, y)
    assert m.score(X, y) > 0.5

def test_ols_predict_shape():
    X, y = make_data()
    m = LinearRegression(method="ols").fit(X, y)
    preds = m.predict(X)
    assert preds.shape == y.shape


# ── Ridge ─────────────────────────────────────────────────────────────────────

def test_ridge_perfect_fit():
    X, y = make_data(noise=0.0)
    m = LinearRegression(method="ridge", alpha=0.0001).fit(X, y)
    assert m.score(X, y) > 0.999

def test_ridge_high_alpha_shrinks_weights():
    X, y = make_data(noise=0.5)
    m_low  = LinearRegression(method="ridge", alpha=0.001).fit(X, y)
    m_high = LinearRegression(method="ridge", alpha=1000.0).fit(X, y)
    assert np.linalg.norm(m_high.weights_) < np.linalg.norm(m_low.weights_)

def test_ridge_mse_finite():
    X, y = make_data(noise=1.0)
    m = LinearRegression(method="ridge").fit(X, y)
    assert np.isfinite(m.mse(X, y))


# ── Gradient Descent ──────────────────────────────────────────────────────────

def test_gd_loss_decreases():
    X, y = make_data(noise=0.5)
    m = LinearRegression(method="gd", learning_rate=0.01, n_iterations=500).fit(X, y)
    assert m.loss_history_[0] > m.loss_history_[-1]

def test_gd_reasonable_r2():
    X, y = make_data(noise=0.5)
    m = LinearRegression(method="gd", learning_rate=0.05, n_iterations=2000).fit(X, y)
    assert m.score(X, y) > 0.5

def test_gd_minibatch_runs():
    X, y = make_data(n=200, noise=0.5)
    m = LinearRegression(method="gd", learning_rate=0.01, n_iterations=300, batch_size=32).fit(X, y)
    assert m.weights_ is not None

def test_gd_loss_history_length():
    X, y = make_data()
    m = LinearRegression(method="gd", n_iterations=100).fit(X, y)
    assert len(m.loss_history_) == 100


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_single_feature():
    X = np.array([[1],[2],[3],[4],[5]], dtype=float)
    y = np.array([2,4,6,8,10], dtype=float)
    m = LinearRegression(method="ols").fit(X, y)
    assert m.score(X, y) > 0.999

def test_repr():
    m = LinearRegression(method="ridge")
    assert "ridge" in repr(m)
