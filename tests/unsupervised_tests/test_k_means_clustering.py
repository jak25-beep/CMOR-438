"""
Unit tests for KMeans.
Run with: pytest test_k_means_clustering.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from k_means_clustering import KMeans


def make_blobs(n=200, k=3, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, 2) * 5
    X = np.vstack([rng.randn(n//k, 2) + c for c in centers])
    return X, centers


def test_labels_shape():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.labels_.shape == (len(X),)

def test_label_values_in_range():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert set(km.labels_).issubset(set(range(3)))

def test_centroids_shape():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.centroids_.shape == (3, 2)

def test_inertia_positive():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.inertia_ > 0

def test_more_clusters_lower_inertia():
    X, _ = make_blobs(k=4)
    km3 = KMeans(k=3, n_init=3, random_state=42).fit(X)
    km5 = KMeans(k=5, n_init=3, random_state=42).fit(X)
    assert km5.inertia_ < km3.inertia_

def test_kmeanspp_vs_random():
    X, _ = make_blobs()
    km_pp  = KMeans(k=3, init='k-means++', n_init=5, random_state=42).fit(X)
    km_rnd = KMeans(k=3, init='random',    n_init=5, random_state=42).fit(X)
    assert km_pp.inertia_ <= km_rnd.inertia_ * 1.5

def test_silhouette_range():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    s = km.silhouette_score(X)
    assert -1.0 <= s <= 1.0

def test_predict_consistency():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert np.array_equal(km.predict(X), km.labels_)

def test_transform_shape():
    X, _ = make_blobs()
    km = KMeans(k=3, n_init=3, random_state=42).fit(X)
    assert km.transform(X).shape == (len(X), 3)

def test_repr():
    assert "KMeans" in repr(KMeans(k=4))
