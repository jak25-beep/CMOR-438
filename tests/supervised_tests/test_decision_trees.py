"""
test_decision_trees.py
======================
Unit tests for DecisionTreeClassifier, which builds a recursive binary tree
by choosing splits that maximise Information Gain (Gini or Entropy).

Run all tests:
    pytest test_decision_trees.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from decision_trees import DecisionTreeClassifier


# ── Shared data factory ───────────────────────────────────────────────────────

def make_data(n=200, seed=42):
    """
    Generate a linearly separable binary classification dataset in 4D.
    The boundary is the hyperplane x0 + x1 = 0.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_perfect_fit_deep():
    """
    With no depth limit the tree can grow until every leaf is pure (contains
    samples from only one class). Training accuracy must be exactly 100%.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert m.accuracy(X, y) == 1.0,         "An unlimited-depth tree must perfectly memorise the training set"

def test_depth_limits_tree():
    """
    Setting max_depth=3 must restrict the actual tree depth to at most 3.
    This verifies that the stopping condition is enforced during tree building.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert m.get_depth() <= 3,         "Actual tree depth must not exceed the specified max_depth"

def test_reasonable_accuracy():
    """
    With max_depth=6, the tree should be deep enough to capture the boundary
    well, but not so deep that it only memorises. We expect >85% training
    accuracy on this easily separable dataset.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=6).fit(X, y)
    assert m.accuracy(X, y) > 0.85,         "DecisionTreeClassifier (depth=6) must achieve >85% training accuracy"

def test_entropy_criterion():
    """
    Both Gini and Entropy are valid splitting criteria that should give
    comparable accuracy on the same data. Verifies the criterion parameter
    is wired up correctly.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(criterion='entropy', max_depth=6).fit(X, y)
    assert m.accuracy(X, y) > 0.85,         "DecisionTreeClassifier with Entropy criterion must achieve >85% training accuracy"


# ── Feature importance tests ──────────────────────────────────────────────────

def test_feature_importances_sum_to_one():
    """
    Feature importances are normalised so they sum to exactly 1.0, making
    them interpretable as relative contributions.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6,         "Feature importances must sum to 1.0 after normalisation"

def test_feature_importances_nonnegative():
    """
    Information gain is always non-negative (a split cannot increase impurity),
    so all feature importance scores must be >= 0.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert (m.feature_importances_ >= 0).all(),         "All feature importances must be non-negative"


# ── Output and parameter tests ────────────────────────────────────────────────

def test_predict_shape():
    """
    predict() must return exactly one label per input sample.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output must have the same shape as the input labels"

def test_min_samples_leaf():
    """
    Setting min_samples_leaf=10 prevents very small leaf nodes, acting as a
    form of regularisation. The tree must still achieve reasonable accuracy
    (>70%) despite being constrained.
    """
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10).fit(X, y)
    assert m.accuracy(X, y) > 0.70,         "A leaf-constrained tree must still achieve >70% training accuracy"

def test_repr():
    """
    repr() must include the class name for identification in logs.
    """
    assert "DecisionTreeClassifier" in repr(DecisionTreeClassifier()),         "repr() must contain the class name"
