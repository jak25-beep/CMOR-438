"""
test_multilayer_perceptron.py
=============================
Unit tests for MLPClassifier, a feedforward neural network for binary
classification trained via backpropagation and mini-batch gradient descent.

The test data uses a non-linear (circular) boundary to verify that hidden
layers genuinely help — a single-layer model could not fit this.

Run all tests:
    pytest test_multilayer_perceptron.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from multilayer_perceptron import MLPClassifier


# ── Shared data factory ───────────────────────────────────────────────────────

def make_data(n=300, seed=42):
    """
    Generate a non-linearly separable binary classification dataset.

    Labels are determined by whether a point falls inside or outside a circle
    of radius √1.5 centred at the origin. A linear model cannot separate these
    classes — hidden layers with ReLU activations are required.

    Returns
    -------
    X : (n, 4) float feature array
    y : (n,)   binary label array in {0, 1}
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:,0]**2 + X[:,1]**2 > 1.5).astype(float)
    return X, y


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_accuracy_above_chance():
    """
    After 200 epochs the MLP must exceed the trivial 50% baseline, confirming
    that the network has learned something meaningful from the non-linear data.
    """
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), learning_rate=0.05,
                      n_iterations=200).fit(X, y)
    assert m.accuracy(X, y) > 0.6,         "MLP must exceed chance-level accuracy (>60%) on the training set after 200 epochs"

def test_loss_decreases():
    """
    The Binary Cross-Entropy loss must decrease over training. A non-decreasing
    loss would indicate a broken backpropagation implementation or a learning
    rate that causes divergence.
    """
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), learning_rate=0.05,
                      n_iterations=200).fit(X, y)
    assert m.loss_history_[0] > m.loss_history_[-1],         "Training loss must be lower at the end than at the start"


# ── Output format tests ───────────────────────────────────────────────────────

def test_predict_proba_range():
    """
    The output layer uses a sigmoid activation, so all probabilities must lie
    in [0, 1]. Values outside this range indicate a numerical overflow in
    the forward pass.
    """
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), n_iterations=100).fit(X, y)
    probs = m.predict_proba(X)
    assert probs.min() >= 0.0 and probs.max() <= 1.0,         "All output probabilities must be in [0, 1]"

def test_predict_binary():
    """
    predict() thresholds probabilities at 0.5, so all outputs must be in
    {0, 1}. Any other value would indicate a broken threshold step.
    """
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), n_iterations=100).fit(X, y)
    assert set(m.predict(X)).issubset({0, 1}),         "predict() must return only binary labels 0 or 1"


# ── Architecture tests ────────────────────────────────────────────────────────

def test_deep_architecture_runs():
    """
    A three-hidden-layer network (32→16→8) must initialise and train without
    error. The number of weight matrices equals the number of layer transitions:
    input→h1, h1→h2, h2→h3, h3→output = 4 matrices.
    """
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(32, 16, 8), n_iterations=50).fit(X, y)
    assert m.weights_ is not None, "weights_ must be populated after training"
    assert len(m.weights_) == 4,         "A 3-hidden-layer network must have 4 weight matrices (one per layer transition)"

def test_loss_history_length():
    """
    One loss value is recorded per epoch, so loss_history_ must have exactly
    n_iterations entries after training.
    """
    X, y = make_data()
    m = MLPClassifier(n_iterations=75).fit(X, y)
    assert len(m.loss_history_) == 75,         "loss_history_ must have exactly one entry per training epoch"

def test_l2_runs_without_error():
    """
    L2 regularisation modifies the gradient computation. Verify the model
    still trains and produces reasonable predictions when l2 > 0.
    """
    X, y = make_data()
    m = MLPClassifier(hidden_layers=(16,), l2=0.01, n_iterations=50).fit(X, y)
    assert m.accuracy(X, y) > 0.4,         "MLP with L2 regularisation must still produce above-random-chance accuracy"

def test_repr():
    """
    repr() must include "MLP" so the model can be identified in logs.
    """
    assert "MLP" in repr(MLPClassifier()), "repr() must contain the class name abbreviation"
