"""
Unit tests for LogisticRegression.
Run with: pytest test_logistic_regression.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from logistic_regression import LogisticRegression


def make_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:,0] + X[:,1] > 0).astype(float)
    return X, y


def test_fits_linearly_separable():
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(X, y)
    assert m.accuracy(X, y) > 0.90

def test_predict_proba_range():
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    probs = m.predict_proba(X)
    assert probs.min() >= 0.0 and probs.max() <= 1.0

def test_predict_binary_labels():
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})

def test_loss_decreases():
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    assert m.loss_history_[0] > m.loss_history_[-1]

def test_l2_shrinks_weights():
    X, y = make_data()
    m0 = LogisticRegression(learning_rate=0.1, n_iterations=300, l2=0.0).fit(X, y)
    m1 = LogisticRegression(learning_rate=0.1, n_iterations=300, l2=10.0).fit(X, y)
    assert np.linalg.norm(m1.weights_) < np.linalg.norm(m0.weights_)

def test_custom_threshold():
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300, threshold=0.3).fit(X, y)
    preds = m.predict(X)
    assert preds.sum() >= LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y).predict(X).sum()

def test_minibatch_runs():
    X, y = make_data(n=300)
    m = LogisticRegression(learning_rate=0.05, n_iterations=200, batch_size=32).fit(X, y)
    assert m.weights_ is not None

def test_repr():
    assert "LogisticRegression" in repr(LogisticRegression())
