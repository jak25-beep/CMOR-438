"""
Unit tests for KNNClassifier and KNNRegressor.
Run with: pytest test_k_nearest_neighbors.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from k_nearest_neighbors import KNNClassifier, KNNRegressor


def make_clf_data(n=150, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y

def make_reg_data(n=150, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = X[:,0]*2 - X[:,1] + 1 + rng.randn(n)*0.3
    return X, y


# ── Classifier ────────────────────────────────────────────────────────────────
def test_clf_k1_memorises():
    X, y = make_clf_data()
    m = KNNClassifier(k=1).fit(X, y)
    assert m.accuracy(X, y) == 1.0

def test_clf_reasonable_accuracy():
    X, y = make_clf_data()
    m = KNNClassifier(k=5).fit(X, y)
    assert m.accuracy(X, y) > 0.80

def test_clf_distance_weights():
    X, y = make_clf_data()
    m = KNNClassifier(k=5, weights='distance').fit(X, y)
    assert m.accuracy(X, y) > 0.75

def test_clf_predict_shape():
    X, y = make_clf_data()
    m = KNNClassifier(k=3).fit(X, y)
    assert m.predict(X).shape == y.shape

def test_clf_manhattan():
    X, y = make_clf_data()
    m = KNNClassifier(k=5, metric='manhattan').fit(X, y)
    assert m.accuracy(X, y) > 0.75


# ── Regressor ─────────────────────────────────────────────────────────────────
def test_reg_k1_memorises():
    X, y = make_reg_data()
    m = KNNRegressor(k=1).fit(X, y)
    assert m.score(X, y) > 0.99

def test_reg_reasonable_r2():
    X, y = make_reg_data()
    m = KNNRegressor(k=5).fit(X, y)
    assert m.score(X, y) > 0.60

def test_reg_predict_shape():
    X, y = make_reg_data()
    m = KNNRegressor(k=3).fit(X, y)
    assert m.predict(X).shape == y.shape

def test_reg_distance_weights():
    X, y = make_reg_data()
    m = KNNRegressor(k=5, weights='distance').fit(X, y)
    assert m.score(X, y) > 0.60

def test_repr():
    assert "KNNClassifier" in repr(KNNClassifier())
    assert "KNNRegressor"  in repr(KNNRegressor())
