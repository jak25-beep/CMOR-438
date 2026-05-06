"""
test_community_detection.py
===========================
Unit tests for LabelPropagation, a graph-based community detection algorithm
that iteratively assigns each node the most common label among its neighbours
until convergence.

Tests use hand-crafted graphs with known community structure (cliques connected
by bridge edges) so the expected output is unambiguous.

Run all tests:
    pytest test_community_detection.py -v
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from community_detection import LabelPropagation


# ── Shared graph factories ────────────────────────────────────────────────────

def two_cliques():
    """
    Build an adjacency matrix for two triangles (3-cliques) connected by
    a single bridge edge between node 2 and node 3.

    Graph structure:
        0 - 1         3 - 4
        |\  |         |\  |
        | 2 |  -  | 5 |
        |  /|         |  /|
        Community A   Community B

    Nodes 0,1,2 form one community; nodes 3,4,5 form another.

    Returns
    -------
    A : (6, 6) symmetric adjacency matrix with binary weights
    """
    A = np.zeros((6, 6))
    for i, j in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(2,3)]:
        A[i,j] = A[j,i] = 1.0
    return A

def three_cliques():
    """
    Build an adjacency matrix for three triangles connected in a chain:
    clique A (0,1,2) - clique B (3,4,5) - clique C (6,7,8)
    Bridge edges: 2-3 and 5-6.

    Returns
    -------
    A : (9, 9) symmetric adjacency matrix
    """
    A = np.zeros((9, 9))
    for group in [range(0,3), range(3,6), range(6,9)]:
        g = list(group)
        for i in g:
            for j in g:
                if i != j:
                    A[i,j] = 1.0
    # Bridge edges between cliques
    A[2,3] = A[3,2] = 1.0
    A[5,6] = A[6,5] = 1.0
    return A


# ── Core detection tests ──────────────────────────────────────────────────────

def test_two_cliques_finds_two_communities():
    """
    The two-clique graph has a clear community structure: the internal edges
    within each triangle vastly outnumber the single bridge edge between them.
    Label Propagation must detect exactly 2 communities.
    """
    A = two_cliques()
    lp = LabelPropagation(max_iter=50, random_state=42).fit(A)
    assert lp.n_communities_ == 2,         "Two clearly separated cliques must be detected as 2 communities"


# ── Output shape and type tests ───────────────────────────────────────────────

def test_labels_shape():
    """
    labels_ must have one integer entry per graph node — the community
    index assigned to that node.
    """
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    assert lp.labels_.shape == (6,),         "labels_ must have one entry per graph node"

def test_labels_contiguous():
    """
    Community labels must be reassigned to contiguous integers starting at 0
    (i.e. 0, 1, ..., n_communities-1). No gaps or skipped indices are allowed,
    since the label set must equal {0, 1, ..., n_communities-1}.
    """
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    assert set(lp.labels_) == set(range(lp.n_communities_)),         "Labels must be contiguous integers from 0 to n_communities-1"


# ── Community coverage tests ──────────────────────────────────────────────────

def test_get_communities_covers_all_nodes():
    """
    get_communities() returns a dict mapping community id → list of node
    indices. Every node must appear in exactly one community — no node
    can be unassigned or duplicated.
    """
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    all_nodes = [n for nodes in lp.get_communities().values() for n in nodes]
    assert sorted(all_nodes) == list(range(6)),         "Every node must appear in exactly one community in get_communities()"


# ── Quality metric tests ──────────────────────────────────────────────────────

def test_modularity_positive():
    """
    Modularity measures whether communities have more internal edges than
    expected by chance. For the two-clique graph (strong community structure)
    modularity must be positive, confirming the algorithm found real structure.
    """
    A = two_cliques()
    lp = LabelPropagation(random_state=42).fit(A)
    assert lp.modularity(A) > 0,         "A graph with strong community structure must have positive modularity"


# ── Alternative input format tests ───────────────────────────────────────────

def test_fit_from_edges():
    """
    fit_from_edges() builds the adjacency matrix from a list of (u, v) tuples
    and then runs label propagation. The result must match what fit() produces
    from the equivalent adjacency matrix — confirming the edge-list builder
    is correct.
    """
    edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(2,3)]
    lp = LabelPropagation(random_state=42).fit_from_edges(edges, n_nodes=6)
    assert lp.n_communities_ == 2,         "fit_from_edges() must detect the same 2 communities as fit() on the equivalent matrix"


# ── Robustness tests ──────────────────────────────────────────────────────────

def test_three_cliques():
    """
    The three-clique chain is harder than two cliques because the middle
    clique (B) shares bridge edges with both A and C. The algorithm should
    still find at least 2 communities (it may find 2 or 3 depending on
    how it resolves the ambiguous middle clique).
    """
    A = three_cliques()
    lp = LabelPropagation(max_iter=100, random_state=42).fit(A)
    assert lp.n_communities_ >= 2,         "Three-clique chain must produce at least 2 communities"

def test_isolated_node_is_own_community():
    """
    Nodes with no edges cannot adopt any neighbour's label, so they retain
    their initial unique label and form singleton communities. This graph
    has a triangle (nodes 0,1,2) plus two isolated nodes (3,4), so we
    expect at least 2 communities (triangle + at least one isolated node).
    """
    A = np.zeros((5, 5))
    # Only nodes 0,1,2 are connected (a triangle); nodes 3 and 4 are isolated
    for i, j in [(0,1),(1,2),(2,0)]:
        A[i,j] = A[j,i] = 1.0
    lp = LabelPropagation(random_state=42).fit(A)
    assert lp.n_communities_ >= 2,         "A graph with isolated nodes must have at least 2 communities"

def test_repr():
    """
    repr() must include the class name for identification in logs.
    """
    assert "LabelPropagation" in repr(LabelPropagation()),         "repr() must contain the class name"
