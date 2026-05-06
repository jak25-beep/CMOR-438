"""
Unit tests for DBSCAN.
Run with: pytest test_dbscan.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dbscan import DBSCAN


def make_blobs(n=100, k=2, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, 2) * 6
    X = np.vstack([rng.randn(n//k, 2)*0.4 + c for c in centers])
    return X


def test_finds_two_blobs():
    X = make_blobs(k=2)
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert db.n_clusters_ == 2

def test_labels_shape():
    X = make_blobs()
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert db.labels_.shape == (len(X),)

def test_noise_label_is_minus_one():
    X = make_blobs()
    X_with_noise = np.vstack([X, [[100, 100]]])
    db = DBSCAN(eps=1.0, min_samples=3).fit(X_with_noise)
    assert db.labels_[-1] == -1

def test_core_samples_are_subset():
    X = make_blobs()
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert set(db.core_samples_).issubset(set(range(len(X))))

def test_tight_eps_creates_noise():
    X = make_blobs()
    db = DBSCAN(eps=0.01, min_samples=5).fit(X)
    assert (db.labels_ == -1).sum() > 0

def test_large_eps_one_cluster():
    X = make_blobs()
    db = DBSCAN(eps=100.0, min_samples=2).fit(X)
    assert db.n_clusters_ == 1

def test_manhattan_metric():
    X = make_blobs()
    db = DBSCAN(eps=1.5, min_samples=3, metric='manhattan').fit(X)
    assert db.n_clusters_ >= 1

def test_fit_predict():
    X = make_blobs()
    db = DBSCAN(eps=1.0, min_samples=3)
    labels = db.fit_predict(X)
    assert labels.shape == (len(X),)

def test_repr():
    assert "DBSCAN" in repr(DBSCAN())
