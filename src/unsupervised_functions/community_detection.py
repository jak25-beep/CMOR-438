"""
community_detection.py
-----------------------
Graph-based community detection via Label Propagation Algorithm (LPA).

The algorithm iteratively updates each node's label to the most
frequent label among its neighbours until convergence.

Usage
-----
from unsupervised_learning.community_detection import LabelPropagation

lp = LabelPropagation(max_iter=100)
lp.fit(adjacency_matrix)
print(lp.labels_)          # community id per node
print(lp.n_communities_)

# Or build the graph from an edge list:
lp2 = LabelPropagation()
lp2.fit_from_edges(edges=[(0,1),(1,2),(2,3)], n_nodes=4)
"""

import numpy as np
from collections import Counter


class LabelPropagation:
    """
    Label Propagation Algorithm for community detection.

    Works on undirected (or directed) graphs represented as an
    adjacency matrix or edge list.

    Parameters
    ----------
    max_iter : int
    random_state : int | None
    """

    def __init__(self, max_iter: int = 100, random_state: int | None = None):
        self.max_iter = max_iter
        self.random_state = random_state

        self.labels_: np.ndarray | None = None
        self.n_communities_: int = 0
        self.n_iter_: int = 0

    # ── Fit from adjacency matrix ─────────────────────────────────────────────

    def fit(self, adjacency: np.ndarray) -> "LabelPropagation":
        """
        Parameters
        ----------
        adjacency : (n_nodes, n_nodes) array — weighted or binary adjacency matrix
        """
        A = np.asarray(adjacency, dtype=float)
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "adjacency must be square"
        n = A.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Initialise: each node gets its own label
        labels = np.arange(n)

        for iteration in range(self.max_iter):
            old_labels = labels.copy()
            order = rng.permutation(n)   # random update order breaks ties

            for i in order:
                neighbors_idx = np.where(A[i] > 0)[0]
                if len(neighbors_idx) == 0:
                    continue

                neighbor_labels = labels[neighbors_idx]
                weights = A[i, neighbors_idx]

                # Weighted label frequency
                freq: dict = {}
                for lbl, w in zip(neighbor_labels, weights):
                    freq[lbl] = freq.get(lbl, 0.0) + w

                max_w = max(freq.values())
                candidates = [lbl for lbl, w in freq.items() if w == max_w]
                labels[i] = rng.choice(candidates)

            self.n_iter_ = iteration + 1
            if np.array_equal(labels, old_labels):
                break

        # Remap labels to 0 … (n_communities - 1)
        unique, inverse = np.unique(labels, return_inverse=True)
        self.labels_ = inverse
        self.n_communities_ = len(unique)
        return self

    # ── Fit from edge list ────────────────────────────────────────────────────

    def fit_from_edges(
        self,
        edges: list[tuple],
        n_nodes: int | None = None,
        weights: list[float] | None = None,
    ) -> "LabelPropagation":
        """
        Build an adjacency matrix from edges and call fit().

        Parameters
        ----------
        edges : list of (u, v) or (u, v, weight) tuples
        n_nodes : total nodes (inferred from edges if None)
        weights : optional edge weights (overrides third element in edges)
        """
        all_nodes = [n for e in edges for n in e[:2]]
        n = n_nodes or (max(all_nodes) + 1)
        A = np.zeros((n, n), dtype=float)

        for k, e in enumerate(edges):
            u, v = e[0], e[1]
            w = (weights[k] if weights else (e[2] if len(e) > 2 else 1.0))
            A[u, v] += w
            A[v, u] += w   # undirected

        return self.fit(A)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def modularity(self, adjacency: np.ndarray) -> float:
        """
        Compute Newman-Girvan modularity Q for the detected communities.
        Q in (-1, 1); higher is better.
        """
        A = np.asarray(adjacency, dtype=float)
        m = A.sum() / 2  # total edge weight
        if m == 0:
            return 0.0
        degrees = A.sum(axis=1)
        Q = 0.0
        for i in range(len(A)):
            for j in range(len(A)):
                if self.labels_[i] == self.labels_[j]:
                    Q += A[i, j] - (degrees[i] * degrees[j]) / (2 * m)
        return float(Q / (2 * m))

    def get_communities(self) -> dict[int, list[int]]:
        """Return a dict mapping community_id → list of node indices."""
        communities: dict[int, list[int]] = {}
        for node, community in enumerate(self.labels_):
            communities.setdefault(int(community), []).append(node)
        return communities

    def __repr__(self):
        return f"LabelPropagation(max_iter={self.max_iter})"
