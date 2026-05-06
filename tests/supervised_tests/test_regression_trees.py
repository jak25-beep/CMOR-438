"""
test_regression_trees.py
========================
Unit tests for DecisionTreeRegressor, which builds a recursive binary tree
using variance reduction as the splitting criterion and predicts the mean
target value in each leaf.

Run all tests:
    pytest test_regression_trees.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from regression_trees import DecisionTreeRegressor


# ── Shared data factory ───────────────────────────────────────────────────────

def make_data(n=200, noise=0.3, seed=42):
    """
    Generate a simple linear regression dataset in 3D.

    True relationship: y = 2*x0 - x1 + 0.5*x2 + noise

    Parameters
    ----------
    noise : std of Gaussian noise (0.0 for a perfectly predictable signal)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = X[:,0]*2 - X[:,1] + 0.5*X[:,2] + rng.randn(n)*noise
    return X, y


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_deep_tree_memorises():
    """
    With no depth limit and no noise, each leaf can contain a single training
    point, giving a prediction exactly equal to the true target. R² must be
    essentially 1.0.
    """
    X, y = make_data(noise=0.0)
    m = DecisionTreeRegressor(max_depth=None).fit(X, y)
    assert m.score(X, y) > 0.99,         "An unlimited-depth tree must achieve near-perfect R² on noiseless data"

def test_depth_limits_tree():
    """
    max_depth=3 must be enforced during tree building. The actual depth
    returned by get_depth() must not exceed 3.
    """
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert m.get_depth() <= 3,         "Actual tree depth must not exceed the specified max_depth"

def test_reasonable_r2():
    """
    With max_depth=6 on low-noise data, the tree should explain most of the
    variance. We require R² > 0.7 to confirm the algorithm is working.
    """
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=6).fit(X, y)
    assert m.score(X, y) > 0.7,         "DecisionTreeRegressor (depth=6) must achieve R² > 0.7 on low-noise data"

def test_mse_positive():
    """
    MSE is a sum of squared errors, so it must always be >= 0.
    A negative MSE would indicate a computation error.
    """
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=5).fit(X, y)
    assert m.mse(X, y) >= 0.0, "MSE must be non-negative"


# ── Output and parameter tests ────────────────────────────────────────────────

def test_predict_shape():
    """
    predict() must return exactly one value per input sample.
    """
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=4).fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output must have the same shape as the target array"

def test_feature_importances_sum_to_one():
    """
    Variance reduction importances are normalised to sum to 1.0, analogous
    to the classifier. All features that contributed to any split should have
    a non-zero importance.
    """
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=5).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6,         "Feature importances must sum to 1.0 after normalisation"

def test_min_samples_leaf_reduces_overfit():
    """
    A tree with min_samples_leaf=1 (default) can create leaves containing a
    single point, giving perfect training R². A tree with min_samples_leaf=10
    is more constrained and will have lower (or equal) training R² — confirming
    that the leaf constraint genuinely restricts the model.
    """
    X, y = make_data()
    m_leaf1  = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1).fit(X, y)
    m_leaf10 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10).fit(X, y)
    assert m_leaf1.score(X, y) >= m_leaf10.score(X, y),         "A less constrained tree (min_samples_leaf=1) must have R² >= a more constrained one"

def test_repr():
    """
    repr() must include the class name for identification in logs.
    """
    assert "DecisionTreeRegressor" in repr(DecisionTreeRegressor()),         "repr() must contain the class name"
