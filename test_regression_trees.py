"""
Unit tests for DecisionTreeRegressor.
Run with: pytest test_regression_trees.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from regression_trees import DecisionTreeRegressor


def make_data(n=200, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = X[:,0]*2 - X[:,1] + 0.5*X[:,2] + rng.randn(n)*noise
    return X, y


def test_deep_tree_memorises():
    X, y = make_data(noise=0.0)
    m = DecisionTreeRegressor(max_depth=None).fit(X, y)
    assert m.score(X, y) > 0.99

def test_depth_limits_tree():
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert m.get_depth() <= 3

def test_reasonable_r2():
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=6).fit(X, y)
    assert m.score(X, y) > 0.7

def test_mse_positive():
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=5).fit(X, y)
    assert m.mse(X, y) >= 0.0

def test_predict_shape():
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=4).fit(X, y)
    assert m.predict(X).shape == y.shape

def test_feature_importances_sum_to_one():
    X, y = make_data()
    m = DecisionTreeRegressor(max_depth=5).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6

def test_min_samples_leaf_reduces_overfit():
    X, y = make_data()
    m_leaf1 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1).fit(X, y)
    m_leaf10 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10).fit(X, y)
    assert m_leaf1.score(X, y) >= m_leaf10.score(X, y)

def test_repr():
    assert "DecisionTreeRegressor" in repr(DecisionTreeRegressor())
