"""
Unit tests for Perceptron.
Run with: pytest test_perceptron.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from perceptron import Perceptron


def make_separable(n=100, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = (X[:,0] + X[:,1] > 0).astype(float)
    return X, y


def test_converges_on_separable():
    X, y = make_separable()
    p = Perceptron(learning_rate=0.1, n_iterations=500).fit(X, y)
    assert p.accuracy(X, y) > 0.95

def test_errors_decrease():
    X, y = make_separable()
    p = Perceptron(learning_rate=0.1, n_iterations=200).fit(X, y)
    assert p.errors_per_epoch_[-1] <= p.errors_per_epoch_[0]

def test_predict_labels_are_pm1():
    X, y = make_separable()
    p = Perceptron().fit(X, y)
    preds = p.predict(X)
    assert set(preds).issubset({-1, 1})

def test_weights_shape():
    X, y = make_separable()
    p = Perceptron().fit(X, y)
    assert p.weights_.shape == (2,)

def test_errors_per_epoch_length():
    X, y = make_separable()
    p = Perceptron(n_iterations=50).fit(X, y)
    assert len(p.errors_per_epoch_) <= 50

def test_repr():
    assert "Perceptron" in repr(Perceptron())
