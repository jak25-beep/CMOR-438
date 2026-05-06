"""
test_k_nearest_neighbors.py
===========================
Unit tests for KNNClassifier and KNNRegressor — lazy learning models that
make predictions by finding the k nearest training points.

Run all tests:
    pytest test_k_nearest_neighbors.py -v
"""

import numpy as np
import sys, os

# KNN has no shared helpers — just add the algorithm folder
sys.path.insert(0, os.path.dirname(__file__))
from k_nearest_neighbors import KNNClassifier, KNNRegressor


# ── Shared data factories ─────────────────────────────────────────────────────

def make_clf_data(n=150, seed=42):
    """
    Generate a linearly separable binary classification dataset in 3D.
    The decision boundary is the plane x0 + x1 = 0.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = (X[:,0] + X[:,1] > 0).astype(int)
    return X, y

def make_reg_data(n=150, seed=42):
    """
    Generate a simple linear regression dataset in 3D.
    True relationship: y = 2*x0 - x1 + 1 + small_noise
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = X[:,0]*2 - X[:,1] + 1 + rng.randn(n)*0.3
    return X, y


# ── KNNClassifier tests ───────────────────────────────────────────────────────

def test_clf_k1_memorises():
    """
    With k=1 every training point is its own nearest neighbour, so the
    classifier perfectly memorises the training set — 100% training accuracy.
    This is expected behaviour (overfitting), not a bug.
    """
    X, y = make_clf_data()
    m = KNNClassifier(k=1).fit(X, y)
    assert m.accuracy(X, y) == 1.0,         "k=1 KNN must achieve 100% training accuracy by memorising each sample"

def test_clf_reasonable_accuracy():
    """
    With k=5 the classifier averages over five neighbours, smoothing out noise.
    On this linearly separable data we expect at least 80% training accuracy.
    """
    X, y = make_clf_data()
    m = KNNClassifier(k=5).fit(X, y)
    assert m.accuracy(X, y) > 0.80,         "KNNClassifier with k=5 should achieve >80% training accuracy"

def test_clf_distance_weights():
    """
    Distance-weighted voting gives closer neighbours more influence.
    The model must still produce meaningful accuracy (>75%) with this weighting.
    """
    X, y = make_clf_data()
    m = KNNClassifier(k=5, weights='distance').fit(X, y)
    assert m.accuracy(X, y) > 0.75,         "Distance-weighted KNNClassifier must achieve >75% training accuracy"

def test_clf_predict_shape():
    """
    predict() must return one label per input sample.
    Output shape must match the number of rows in X.
    """
    X, y = make_clf_data()
    m = KNNClassifier(k=3).fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output shape must match the number of input samples"

def test_clf_manhattan():
    """
    The Manhattan distance metric should produce comparable accuracy to
    Euclidean distance on this dataset, confirming the metric switching logic
    works correctly.
    """
    X, y = make_clf_data()
    m = KNNClassifier(k=5, metric='manhattan').fit(X, y)
    assert m.accuracy(X, y) > 0.75,         "KNNClassifier with Manhattan distance must achieve >75% training accuracy"


# ── KNNRegressor tests ────────────────────────────────────────────────────────

def test_reg_k1_memorises():
    """
    With k=1, the regressor returns the exact training target for each point,
    giving near-perfect R² on the training set.
    """
    X, y = make_reg_data()
    m = KNNRegressor(k=1).fit(X, y)
    assert m.score(X, y) > 0.99,         "k=1 KNNRegressor must achieve near-perfect R² by returning training targets exactly"

def test_reg_reasonable_r2():
    """
    With k=5, averaging over five neighbours smooths predictions. We expect
    at least R²=0.60 on this low-noise dataset.
    """
    X, y = make_reg_data()
    m = KNNRegressor(k=5).fit(X, y)
    assert m.score(X, y) > 0.60,         "KNNRegressor with k=5 should achieve R² > 0.60 on this dataset"

def test_reg_predict_shape():
    """
    predict() must return one value per input sample.
    """
    X, y = make_reg_data()
    m = KNNRegressor(k=3).fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output shape must match the number of input samples"

def test_reg_distance_weights():
    """
    Distance weighting should work for the regressor as well, giving
    at least R²=0.60.
    """
    X, y = make_reg_data()
    m = KNNRegressor(k=5, weights='distance').fit(X, y)
    assert m.score(X, y) > 0.60,         "Distance-weighted KNNRegressor must achieve R² > 0.60"

def test_repr():
    """
    Both classes must include their class names in repr() output.
    """
    assert "KNNClassifier" in repr(KNNClassifier()),         "KNNClassifier repr() must include the class name"
    assert "KNNRegressor" in repr(KNNRegressor()),         "KNNRegressor repr() must include the class name"
