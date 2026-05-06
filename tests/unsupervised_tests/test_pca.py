"""
Unit tests for PCA.
Run with: pytest test_pca.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pca import PCA


def make_data(n=200, d=5, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d)


def test_transform_shape():
    X = make_data()
    p = PCA(n_components=2).fit(X)
    assert p.transform(X).shape == (200, 2)

def test_explained_variance_ratio_sums_to_one():
    X = make_data()
    p = PCA(n_components=None).fit(X)
    assert abs(p.explained_variance_ratio_.sum() - 1.0) < 1e-6

def test_explained_variance_ratio_nonnegative():
    X = make_data()
    p = PCA(n_components=3).fit(X)
    assert (p.explained_variance_ratio_ >= 0).all()

def test_explained_variance_ratio_descending():
    X = make_data()
    p = PCA(n_components=None).fit(X)
    evr = p.explained_variance_ratio_
    assert all(evr[i] >= evr[i+1] for i in range(len(evr)-1))

def test_reconstruction_error_zero_full_rank():
    X = make_data(d=3)
    p = PCA(n_components=None).fit(X)
    assert p.reconstruction_error(X) < 1e-10

def test_float_n_components():
    X = make_data()
    p = PCA(n_components=0.95).fit(X)
    assert p.explained_variance_ratio_.sum() >= 0.95

def test_fit_transform_equals_fit_then_transform():
    X = make_data()
    p = PCA(n_components=2)
    X1 = p.fit_transform(X)
    X2 = PCA(n_components=2).fit(X).transform(X)
    assert np.allclose(np.abs(X1), np.abs(X2))

def test_components_shape():
    X = make_data()
    p = PCA(n_components=3).fit(X)
    assert p.components_.shape == (3, 5)

def test_repr():
    assert "PCA" in repr(PCA(n_components=2))
