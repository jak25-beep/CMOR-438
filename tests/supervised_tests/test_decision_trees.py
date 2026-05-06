"""
Unit tests for DecisionTreeClassifier.
Run with: pytest test_decision_trees.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from decision_trees import DecisionTreeClassifier


def make_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y


def test_perfect_fit_deep():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert m.accuracy(X, y) == 1.0

def test_depth_limits_tree():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert m.get_depth() <= 3

def test_reasonable_accuracy():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=6).fit(X, y)
    assert m.accuracy(X, y) > 0.85

def test_entropy_criterion():
    X, y = make_data()
    m = DecisionTreeClassifier(criterion='entropy', max_depth=6).fit(X, y)
    assert m.accuracy(X, y) > 0.85

def test_feature_importances_sum_to_one():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6

def test_feature_importances_nonnegative():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert (m.feature_importances_ >= 0).all()

def test_predict_shape():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert m.predict(X).shape == y.shape

def test_min_samples_leaf():
    X, y = make_data()
    m = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10).fit(X, y)
    assert m.accuracy(X, y) > 0.70

def test_repr():
    assert "DecisionTreeClassifier" in repr(DecisionTreeClassifier())
