"""
test_logistic_regression.py
===========================
Unit tests for LogisticRegression, a binary classifier trained by minimising
Binary Cross-Entropy via Gradient Descent with an optional L2 penalty.

Run all tests:
    pytest test_logistic_regression.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from logistic_regression import LogisticRegression


# ── Shared data factory ───────────────────────────────────────────────────────

def make_data(n=200, seed=42):
    """
    Generate a linearly separable binary classification dataset.

    Labels are determined by the sign of (x0 + x1), giving a clear linear
    boundary that Logistic Regression can fit without difficulty.

    Returns
    -------
    X : (n, 4) float feature array
    y : (n,)   binary label array in {0, 1}
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:,0] + X[:,1] > 0).astype(float)
    return X, y


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_fits_linearly_separable():
    """
    On linearly separable data the model should comfortably exceed 90% training
    accuracy after 500 iterations with a learning rate of 0.1.
    """
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(X, y)
    assert m.accuracy(X, y) > 0.90,         "Logistic Regression must achieve >90% accuracy on linearly separable data"

def test_predict_proba_range():
    """
    predict_proba() applies a sigmoid, so all outputs must lie in [0, 1].
    Values outside this range would indicate a numerical problem.
    """
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    probs = m.predict_proba(X)
    assert probs.min() >= 0.0 and probs.max() <= 1.0,         "All predicted probabilities must be in the valid range [0, 1]"

def test_predict_binary_labels():
    """
    predict() applies a threshold to probabilities, so outputs must be
    exclusively 0 or 1 — no other values are valid class labels.
    """
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1}),         "predict() must return only binary labels 0 or 1"

def test_loss_decreases():
    """
    Gradient descent must reduce the Binary Cross-Entropy loss monotonically
    (or close to it). Comparing first and last loss values confirms the
    optimiser is working correctly.
    """
    X, y = make_data()
    m = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    assert m.loss_history_[0] > m.loss_history_[-1],         "Training loss must decrease over the course of gradient descent"


# ── Regularisation tests ──────────────────────────────────────────────────────

def test_l2_shrinks_weights():
    """
    The core property of L2 regularisation: a larger penalty coefficient
    forces the weight vector closer to zero. Measured via the L2 (Euclidean)
    norm of the weight vector.
    """
    X, y = make_data()
    m_no_reg = LogisticRegression(learning_rate=0.1, n_iterations=300, l2=0.0).fit(X, y)
    m_strong = LogisticRegression(learning_rate=0.1, n_iterations=300, l2=10.0).fit(X, y)
    assert np.linalg.norm(m_strong.weights_) < np.linalg.norm(m_no_reg.weights_),         "Strong L2 regularisation must produce a smaller-norm weight vector"

def test_custom_threshold():
    """
    Lowering the decision threshold from 0.5 to 0.3 means the model predicts
    class 1 for a wider range of probabilities. The total number of positive
    predictions must therefore be at least as large as with the default threshold.
    """
    X, y = make_data()
    m_default = LogisticRegression(learning_rate=0.1, n_iterations=300).fit(X, y)
    m_low_thr = LogisticRegression(learning_rate=0.1, n_iterations=300,
                                   threshold=0.3).fit(X, y)
    assert m_low_thr.predict(X).sum() >= m_default.predict(X).sum(),         "A lower threshold must produce at least as many positive predictions"


# ── Training configuration tests ─────────────────────────────────────────────

def test_minibatch_runs():
    """
    Mini-batch gradient descent (batch_size=32) must complete without error
    and produce a fitted model, confirming the batching code path works.
    """
    X, y = make_data(n=300)
    m = LogisticRegression(learning_rate=0.05, n_iterations=200,
                           batch_size=32).fit(X, y)
    assert m.weights_ is not None,         "Mini-batch training must populate weights_ after fitting"

def test_repr():
    """
    repr() must include the class name so models can be identified in logs
    and interactive sessions.
    """
    assert "LogisticRegression" in repr(LogisticRegression()),         "repr() must contain the class name"
