"""
test_dbscan.py
==============
Unit tests for DBSCAN (Density-Based Spatial Clustering of Applications
with Noise), which discovers clusters of arbitrary shape and labels sparse
points as noise (-1) without requiring k to be specified.

Tests use tight Gaussian blobs with large separation so the correct cluster
structure is unambiguous.

Run all tests:
    pytest test_dbscan.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from dbscan import DBSCAN


# ── Shared data factory ───────────────────────────────────────────────────────

def make_blobs(n=100, k=2, seed=42):
    """
    Generate k tight Gaussian clusters with well-separated centres.

    Centres are drawn from N(0, 6²) so blobs are far apart.
    Each blob has standard deviation 0.4 — tight enough that DBSCAN
    with a moderate eps clearly separates them.

    Returns
    -------
    X : (n, 2) float array
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, 2) * 6
    X = np.vstack([rng.randn(n//k, 2)*0.4 + c for c in centers])
    return X


# ── Cluster discovery tests ───────────────────────────────────────────────────

def test_finds_two_blobs():
    """
    With well-separated blobs and a suitable eps, DBSCAN must discover
    exactly two clusters. This is the primary functional test.
    """
    X = make_blobs(k=2)
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert db.n_clusters_ == 2,         "DBSCAN must find exactly 2 clusters in clearly separated blob data"


# ── Output shape and type tests ───────────────────────────────────────────────

def test_labels_shape():
    """
    labels_ must have one integer entry per input sample.
    Noise points receive label -1; cluster points receive a non-negative index.
    """
    X = make_blobs()
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert db.labels_.shape == (len(X),),         "labels_ must have one entry per input sample"

def test_noise_label_is_minus_one():
    """
    An isolated outlier at (100, 100) — far from all other points — must
    receive label -1, confirming the noise classification logic works.
    """
    X = make_blobs()
    X_with_noise = np.vstack([X, [[100, 100]]])
    db = DBSCAN(eps=1.0, min_samples=3).fit(X_with_noise)
    assert db.labels_[-1] == -1,         "An isolated outlier point must be labelled -1 (noise)"

def test_core_samples_are_subset():
    """
    core_samples_ stores the indices of core points (those with enough
    neighbours). Every index must refer to a valid sample in the input.
    """
    X = make_blobs()
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert set(db.core_samples_).issubset(set(range(len(X)))),         "core_samples_ must only contain valid sample indices"


# ── Parameter sensitivity tests ───────────────────────────────────────────────

def test_tight_eps_creates_noise():
    """
    When eps is extremely small (0.01), almost no points have enough neighbours
    to be core points, so the vast majority are classified as noise.
    This confirms that eps correctly controls neighbourhood size.
    """
    X = make_blobs()
    db = DBSCAN(eps=0.01, min_samples=5).fit(X)
    assert (db.labels_ == -1).sum() > 0,         "A very small eps must cause some points to be classified as noise"

def test_large_eps_one_cluster():
    """
    When eps is enormous (100), every point is in every other point's
    neighbourhood, so all points are merged into a single cluster.
    This confirms eps correctly controls the cluster connectivity threshold.
    """
    X = make_blobs()
    db = DBSCAN(eps=100.0, min_samples=2).fit(X)
    assert db.n_clusters_ == 1,         "A very large eps must merge all points into a single cluster"


# ── Metric and API tests ──────────────────────────────────────────────────────

def test_manhattan_metric():
    """
    Switching to Manhattan distance must still find at least one cluster
    on the blob data, confirming the metric parameter is functional.
    """
    X = make_blobs()
    db = DBSCAN(eps=1.5, min_samples=3, metric='manhattan').fit(X)
    assert db.n_clusters_ >= 1,         "Manhattan-distance DBSCAN must find at least 1 cluster on blob data"

def test_fit_predict():
    """
    fit_predict() is a convenience method equivalent to fit() followed by
    accessing labels_. The returned array must have the correct shape.
    """
    X = make_blobs()
    db = DBSCAN(eps=1.0, min_samples=3)
    labels = db.fit_predict(X)
    assert labels.shape == (len(X),),         "fit_predict() must return a labels array with one entry per sample"

def test_repr():
    """
    repr() must include the class name for identification in logs.
    """
    assert "DBSCAN" in repr(DBSCAN()),         "repr() must contain the class name"
