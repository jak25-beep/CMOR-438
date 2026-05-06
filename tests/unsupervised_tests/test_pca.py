"""
test_pca.py
===========
Unit tests for PCA (Principal Component Analysis), which finds the directions
of maximum variance in data via eigendecomposition of the covariance matrix
and projects onto the top k eigenvectors.

Run all tests:
    pytest test_pca.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from pca import PCA


# ── Shared data factory ───────────────────────────────────────────────────────

def make_data(n=200, d=5, seed=42):
    """
    Generate isotropic Gaussian data with n samples and d features.
    No structure is imposed — this tests the mechanics of PCA, not
    the ability to discover a specific pattern.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(n, d)


# ── Output shape tests ────────────────────────────────────────────────────────

def test_transform_shape():
    """
    Projecting 200 samples from 5D onto 2 principal components must give
    a (200, 2) array. This is the fundamental dimensionality reduction check.
    """
    X = make_data()
    p = PCA(n_components=2).fit(X)
    assert p.transform(X).shape == (200, 2),         "transform() must output shape (n_samples, n_components)"

def test_components_shape():
    """
    components_ stores the k principal component directions as row vectors.
    For 3 components from 5D data the shape must be (3, 5).
    """
    X = make_data()
    p = PCA(n_components=3).fit(X)
    assert p.components_.shape == (3, 5),         "components_ must have shape (n_components, n_features)"


# ── Explained variance tests ──────────────────────────────────────────────────

def test_explained_variance_ratio_sums_to_one():
    """
    When keeping all components (n_components=None), the explained variance
    ratios must sum to exactly 1.0 — all variance is accounted for.
    """
    X = make_data()
    p = PCA(n_components=None).fit(X)
    assert abs(p.explained_variance_ratio_.sum() - 1.0) < 1e-6,         "Explained variance ratios must sum to 1.0 when keeping all components"

def test_explained_variance_ratio_nonnegative():
    """
    Eigenvalues of the covariance matrix are always non-negative, so
    explained variance ratios must also be >= 0.
    """
    X = make_data()
    p = PCA(n_components=3).fit(X)
    assert (p.explained_variance_ratio_ >= 0).all(),         "All explained variance ratios must be non-negative"

def test_explained_variance_ratio_descending():
    """
    PCA sorts components by descending eigenvalue, so each component must
    explain at least as much variance as the next one. This ordering is
    what makes the first k components the most informative choice.
    """
    X = make_data()
    p = PCA(n_components=None).fit(X)
    evr = p.explained_variance_ratio_
    assert all(evr[i] >= evr[i+1] for i in range(len(evr)-1)),         "Explained variance ratios must be in descending order"


# ── Reconstruction and float n_components tests ───────────────────────────────

def test_reconstruction_error_zero_full_rank():
    """
    Projecting onto all d components and inverting is a lossless round-trip.
    The reconstruction error must be essentially zero (< 1e-10) on 3D data
    with all 3 components retained.
    """
    X = make_data(d=3)
    p = PCA(n_components=None).fit(X)
    assert p.reconstruction_error(X) < 1e-10,         "Full-rank PCA must reconstruct the data with zero error"

def test_float_n_components():
    """
    When n_components is given as a float (e.g. 0.95), PCA must automatically
    choose the minimum number of components needed to explain at least that
    fraction of total variance.
    """
    X = make_data()
    p = PCA(n_components=0.95).fit(X)
    assert p.explained_variance_ratio_.sum() >= 0.95,         "Float n_components must retain enough components to explain the requested variance fraction"


# ── Consistency tests ─────────────────────────────────────────────────────────

def test_fit_transform_equals_fit_then_transform():
    """
    fit_transform(X) must produce the same result as fit(X).transform(X).
    PCA eigenvectors are defined up to sign, so we compare absolute values
    to avoid false negatives from sign flips.
    """
    X = make_data()
    p = PCA(n_components=2)
    X1 = p.fit_transform(X)
    X2 = PCA(n_components=2).fit(X).transform(X)
    assert np.allclose(np.abs(X1), np.abs(X2)),         "fit_transform() must be equivalent to fit() followed by transform() (up to sign)"

def test_repr():
    """
    repr() must include the class name for identification in logs.
    """
    assert "PCA" in repr(PCA(n_components=2)),         "repr() must contain the class name"
