"""
Unit tests for LabelPropagation.
Run with: pytest test_community_detection.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from community_detection import LabelPropagation


def two_cliques():
    """Two triangles connected by a single bridge edge."""
    A = np.zeros((6, 6))
    for i, j in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(2,3)]:
        A[i,j] = A[j,i] = 1.0
    return A

def three_cliques():
    A = np.zeros((9, 9))
    for group in [range(0,3), range(3,6), range(6,9)]:
        g = list(group)
        for i in g:
            for j in g:
                if i != j: A[i,j] = 1.0
    A[2,3] = A[3,2] = 1.0
    A[5,6] = A[6,5] = 1.0
    return A


def test_two_cliques_finds_two_communities():
    A = two_cliques()
    lp = LabelPropagation(max_iter=50, random_state=42).fit(A)
    assert lp.n_communities_ == 2

def test_labels_shape():
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    assert lp.labels_.shape == (6,)

def test_labels_contiguous():
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    assert set(lp.labels_) == set(range(lp.n_communities_))

def test_get_communities_covers_all_nodes():
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    all_nodes = [n for nodes in lp.get_communities().values() for n in nodes]
    assert sorted(all_nodes) == list(range(6))

def test_modularity_positive():
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    assert lp.modularity(A) > 0

def test_fit_from_edges():
    edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(2,3)]
    lp = LabelPropagation(random_state=42).fit_from_edges(edges, n_nodes=6)
    assert lp.n_communities_ == 2

def test_three_cliques():
    A = three_cliques()
    lp = LabelPropagation(max_iter=100, random_state=42).fit(A)
    assert lp.n_communities_ >= 2

def test_isolated_node_is_own_community():
    A = np.zeros((5, 5))
    for i, j in [(0,1),(1,2),(2,0)]:
        A[i,j] = A[j,i] = 1.0
    lp = LabelPropagation(random_state=42).fit(A)
    assert lp.n_communities_ >= 2

def test_repr():
    assert "LabelPropagation" in repr(LabelPropagation())
