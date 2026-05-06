"""
Unit tests for RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor.
Run with: pytest test_ensemble_methods.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from ensemble_methods import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor


def make_clf(n=200, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:,0]+X[:,1] > 0).astype(int)
    return X, y

def make_reg(n=200, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = X[:,0]*2 - X[:,1] + rng.randn(n)*0.5
    return X, y


# ── Random Forest Classifier ──────────────────────────────────────────────────
def test_rfc_accuracy():
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
    assert m.accuracy(X, y) > 0.85

def test_rfc_feature_importances_sum():
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6

def test_rfc_n_estimators():
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=15, random_state=42).fit(X, y)
    assert len(m.estimators_) == 15

def test_rfc_predict_shape():
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    assert m.predict(X).shape == y.shape


# ── Random Forest Regressor ───────────────────────────────────────────────────
def test_rfr_r2():
    X, y = make_reg()
    m = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
    assert m.score(X, y) > 0.80

def test_rfr_predict_shape():
    X, y = make_reg()
    m = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    assert m.predict(X).shape == y.shape


# ── Gradient Boosting ─────────────────────────────────────────────────────────
def test_gbr_r2():
    X, y = make_reg()
    m = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X, y)
    assert m.score(X, y) > 0.80

def test_gbr_loss_decreases():
    X, y = make_reg()
    m = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42).fit(X, y)
    assert m.train_loss_[0] > m.train_loss_[-1]

def test_gbr_n_estimators():
    X, y = make_reg()
    m = GradientBoostingRegressor(n_estimators=30, random_state=42).fit(X, y)
    assert len(m.estimators_) == 30

def test_repr():
    assert "RandomForest" in repr(RandomForestClassifier())
    assert "GradientBoosting" in repr(GradientBoostingRegressor())
