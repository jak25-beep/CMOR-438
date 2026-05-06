"""
Unit tests for MLPClassifier.
Run with: pytest test_multilayer_perceptron.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from multilayer_perceptron import MLPClassifier


def make_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:,0]**2 + X[:,1]**2 > 1.5).astype(float)
    return X, y


def test_accuracy_above_chance():
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), learning_rate=0.05, n_iterations=200).fit(X, y)
    assert m.accuracy(X, y) > 0.6

def test_loss_decreases():
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), learning_rate=0.05, n_iterations=200).fit(X, y)
    assert m.loss_history_[0] > m.loss_history_[-1]

def test_predict_proba_range():
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), n_iterations=100).fit(X, y)
    probs = m.predict_proba(X)
    assert probs.min() >= 0.0 and probs.max() <= 1.0

def test_predict_binary():
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), n_iterations=100).fit(X, y)
    assert set(m.predict(X)).issubset({0, 1})

def test_deep_architecture_runs():
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(32, 16, 8), n_iterations=50).fit(X, y)
    assert m.weights_ is not None and len(m.weights_) == 4

def test_loss_history_length():
    X, y = make_data()
    m = MLPClassifier(n_iterations=75).fit(X, y)
    assert len(m.loss_history_) == 75

def test_l2_runs_without_error():
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), l2=0.01, n_iterations=50).fit(X, y)
    assert m.accuracy(X, y) > 0.4

def test_repr():
    assert "MLP" in repr(MLPClassifier())
