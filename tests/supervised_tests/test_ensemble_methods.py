"""
test_ensemble_methods.py
========================
Unit tests for the three ensemble learning classes:
  - RandomForestClassifier  (bagging + random subspace, majority voting)
  - RandomForestRegressor   (bagging + random subspace, averaged predictions)
  - GradientBoostingRegressor (sequential residual fitting with shrinkage)

All three are built on top of the decision/regression tree implementations.

Run all tests:
    pytest test_ensemble_methods.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))
from ensemble_methods import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)


# ── Shared data factories ─────────────────────────────────────────────────────

def make_clf(n=200, seed=42):
    """
    Linearly separable binary classification dataset with 5 features.
    Decision boundary: x0 + x1 = 0.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:,0]+X[:,1] > 0).astype(int)
    return X, y

def make_reg(n=200, seed=42):
    """
    Simple linear regression dataset with 5 features and low noise.
    True relationship: y = 2*x0 - x1 + noise(std=0.5)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = X[:,0]*2 - X[:,1] + rng.randn(n)*0.5
    return X, y


# ── RandomForestClassifier tests ──────────────────────────────────────────────

def test_rfc_accuracy():
    """
    With 20 trees and max_depth=5, the ensemble should clearly outperform a
    single tree on this data. We require >85% training accuracy.
    """
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=20, max_depth=5,
                               random_state=42).fit(X, y)
    assert m.accuracy(X, y) > 0.85,         "RandomForestClassifier (20 trees) must achieve >85% training accuracy"

def test_rfc_feature_importances_sum():
    """
    Feature importances are averaged across all trees and then normalised.
    They must sum to 1.0 to be interpretable as relative contributions.
    """
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=10, max_depth=4,
                               random_state=42).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6,         "Averaged feature importances must sum to 1.0"

def test_rfc_n_estimators():
    """
    The estimators_ list must contain exactly n_estimators fitted trees after
    training — confirming the training loop ran the correct number of times.
    """
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=15, random_state=42).fit(X, y)
    assert len(m.estimators_) == 15,         "estimators_ must contain exactly n_estimators fitted trees"

def test_rfc_predict_shape():
    """
    The majority-vote prediction must return exactly one label per input sample.
    """
    X, y = make_clf()
    m = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output shape must match the number of input samples"


# ── RandomForestRegressor tests ───────────────────────────────────────────────

def test_rfr_r2():
    """
    With 20 trees the averaged predictions should achieve R² > 0.80 on this
    low-noise dataset — demonstrating that averaging reduces the high variance
    of individual trees.
    """
    X, y = make_reg()
    m = RandomForestRegressor(n_estimators=20, max_depth=5,
                              random_state=42).fit(X, y)
    assert m.score(X, y) > 0.80,         "RandomForestRegressor (20 trees) must achieve R² > 0.80"

def test_rfr_predict_shape():
    """
    The averaged prediction must return exactly one value per input sample.
    """
    X, y = make_reg()
    m = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    assert m.predict(X).shape == y.shape,         "predict() output shape must match the number of input samples"


# ── GradientBoostingRegressor tests ──────────────────────────────────────────

def test_gbr_r2():
    """
    After 50 boosting rounds the ensemble must achieve R² > 0.80, demonstrating
    that sequential residual fitting effectively improves predictions.
    """
    X, y = make_reg()
    m = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                  max_depth=3, random_state=42).fit(X, y)
    assert m.score(X, y) > 0.80,         "GradientBoostingRegressor (50 rounds) must achieve R² > 0.80"

def test_gbr_loss_decreases():
    """
    The training loss (residual MSE) must decrease over boosting rounds.
    A flat or increasing loss would indicate the shrinkage or residual
    computation is broken.
    """
    X, y = make_reg()
    m = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                  random_state=42).fit(X, y)
    assert m.train_loss_[0] > m.train_loss_[-1],         "Gradient Boosting training loss must decrease over boosting rounds"

def test_gbr_n_estimators():
    """
    One tree is added per boosting round, so estimators_ must have exactly
    n_estimators entries after training.
    """
    X, y = make_reg()
    m = GradientBoostingRegressor(n_estimators=30, random_state=42).fit(X, y)
    assert len(m.estimators_) == 30,         "estimators_ must contain exactly n_estimators trees after training"

def test_repr():
    """
    repr() for both classes must include their class names.
    """
    assert "RandomForest" in repr(RandomForestClassifier()),         "RandomForestClassifier repr() must contain the class name"
    assert "GradientBoosting" in repr(GradientBoostingRegressor()),         "GradientBoostingRegressor repr() must contain the class name"
