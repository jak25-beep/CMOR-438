"""
test_perceptron.py
==================
Unit tests for the Perceptron, the classic single-layer binary classifier
that updates weights only when it makes a mistake.

Labels are mapped internally to {-1, +1} and the model outputs the same set.

Run all tests:
    pytest test_perceptron.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from perceptron import Perceptron


# ── Shared data factory ───────────────────────────────────────────────────────

def make_separable(n=100, seed=42):
    """
    Generate a linearly separable binary dataset in 2D.

    The boundary is the line x0 + x1 = 0. Points above it are class 1,
    below are class 0 (mapped to -1 internally by the Perceptron).

    Returns
    -------
    X : (n, 2) float feature array
    y : (n,)   binary label array in {0, 1}
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = (X[:,0] + X[:,1] > 0).astype(float)
    return X, y


# ── Convergence tests ─────────────────────────────────────────────────────────

def test_converges_on_separable():
    """
    The Perceptron Convergence Theorem guarantees that on linearly separable
    data the algorithm will find a separating hyperplane. We expect >95%
    training accuracy with 500 iterations and lr=0.1.
    """
    X, y = make_separable()
    p = Perceptron(learning_rate=0.1, n_iterations=500).fit(X, y)
    assert p.accuracy(X, y) > 0.95,         "Perceptron must converge to >95% accuracy on linearly separable data"

def test_errors_decrease():
    """
    Even if the Perceptron hasn't fully converged, the number of errors in
    the final epoch should not be larger than in the first epoch. This checks
    that the weight updates are improving the model, not degrading it.
    """
    X, y = make_separable()
    p = Perceptron(learning_rate=0.1, n_iterations=200).fit(X, y)
    assert p.errors_per_epoch_[-1] <= p.errors_per_epoch_[0],         "Errors in the last epoch must not exceed errors in the first epoch"


# ── Output format tests ───────────────────────────────────────────────────────

def test_predict_labels_are_pm1():
    """
    The Perceptron uses a step activation and outputs {-1, +1} — not {0, 1}.
    This is consistent with the internal label mapping and the classic
    Perceptron formulation.
    """
    X, y = make_separable()
    p = Perceptron().fit(X, y)
    preds = p.predict(X)
    assert set(preds).issubset({-1, 1}),         "Perceptron must only predict labels in {-1, +1}"

def test_weights_shape():
    """
    After fitting, the weight vector must have one entry per input feature.
    With 2 input features, weights_ must have shape (2,).
    """
    X, y = make_separable()
    p = Perceptron().fit(X, y)
    assert p.weights_.shape == (2,),         "weights_ must have one element per input feature"


# ── Training state tests ──────────────────────────────────────────────────────

def test_errors_per_epoch_length():
    """
    One error count is recorded per epoch. If the model converges early the
    loop stops, so the length must be at most n_iterations — never more.
    """
    X, y = make_separable()
    p = Perceptron(n_iterations=50).fit(X, y)
    assert len(p.errors_per_epoch_) <= 50,         "errors_per_epoch_ must have at most n_iterations entries (fewer if early convergence)"

def test_repr():
    """
    repr() must identify the class so models can be recognised in logs.
    """
    assert "Perceptron" in repr(Perceptron()),         "repr() must contain the class name"
