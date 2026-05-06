"""
test_k_means_clustering.py
==========================
Unit tests for the KMeans clustering algorithm, which partitions data into
k clusters by iterating between assignment and centroid update steps.

Tests use synthetic Gaussian blob data with well-separated centres so that
the expected cluster structure is unambiguous.

Run all tests:
    pytest test_k_means_clustering.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from k_means_clustering import KMeans


# ── Shared data factory ───────────────────────────────────────────────────────

def make_blobs(n=200, k=3, seed=42):
    """
    Generate k isotropic Gaussian clusters with well-separated centres.

    Centres are drawn from N(0, 5²) so they are typically far apart.
    Each cluster contains n//k samples from N(centre, 1).

    Returns
    -------
    X       : (n, 2) float array
    centers : (k, 2) ground-truth cluster centres
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, 2) * 5
    X = np.vstack([rng.randn(n//k, 2) + c for c in centers])
    return X, centers


# ── Output shape and type tests ───────────────────────────────────────────────

def test_labels_shape():
    """
    labels_ must have one integer entry per input sample — the cluster index
    for that sample.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.labels_.shape == (len(X),),         "labels_ must have one entry per input sample"

def test_label_values_in_range():
    """
    All cluster labels must be valid indices in [0, k). Any label outside
    this range would indicate an off-by-one or initialisation error.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert set(km.labels_).issubset(set(range(3))),         "All labels must be valid cluster indices in [0, k)"

def test_centroids_shape():
    """
    centroids_ must be a (k, n_features) array — one centroid vector per
    cluster, with the same dimensionality as the input data.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.centroids_.shape == (3, 2),         "centroids_ must have shape (k, n_features)"


# ── Inertia and quality tests ─────────────────────────────────────────────────

def test_inertia_positive():
    """
    Inertia (within-cluster sum of squares) is a sum of squared distances,
    so it must always be strictly positive for non-degenerate data.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.inertia_ > 0, "Inertia must be strictly positive"

def test_more_clusters_lower_inertia():
    """
    Adding more clusters can only reduce or maintain inertia — it never
    increases it. This is a fundamental property of k-means: a finer
    partition always fits the training data at least as well.
    """
    X, _ = make_blobs(k=4)
    km3 = KMeans(k=3, n_init=3, random_state=42).fit(X)
    km5 = KMeans(k=5, n_init=3, random_state=42).fit(X)
    assert km5.inertia_ < km3.inertia_,         "More clusters must produce lower or equal inertia on the same data"

def test_kmeanspp_vs_random():
    """
    k-means++ initialisation should find solutions at least as good as random
    initialisation (within a 1.5× inertia factor). k-means++ is designed to
    avoid bad starting configurations that lead to poor local minima.
    """
    X, _ = make_blobs()
    km_pp  = KMeans(k=3, init='k-means++', n_init=5, random_state=42).fit(X)
    km_rnd = KMeans(k=3, init='random',    n_init=5, random_state=42).fit(X)
    assert km_pp.inertia_ <= km_rnd.inertia_ * 1.5,         "k-means++ inertia must be within 1.5× of random initialisation inertia"

def test_silhouette_range():
    """
    The silhouette score measures cluster cohesion vs separation and is
    defined in [-1, 1]. A value outside this range would indicate a
    computation error.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    s = km.silhouette_score(X)
    assert -1.0 <= s <= 1.0,         "Silhouette score must be in the valid range [-1, 1]"


# ── predict() and transform() tests ──────────────────────────────────────────

def test_predict_consistency():
    """
    predict(X_train) must return the same labels as labels_ set during fit().
    Inconsistency here would indicate the centroid assignment logic differs
    between fit and predict.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert np.array_equal(km.predict(X), km.labels_),         "predict() on training data must return the same labels as labels_"

def test_transform_shape():
    """
    transform() returns the distance from each sample to each centroid,
    giving an (n_samples, k) array. This is used for soft cluster membership.
    """
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.transform(X).shape == (len(X), 3),         "transform() must return shape (n_samples, k)"

def test_repr():
    """
    repr() must include the class name and k value for identification.
    """
    assert "KMeans" in repr(KMeans(k=4)),         "repr() must contain the class name"
