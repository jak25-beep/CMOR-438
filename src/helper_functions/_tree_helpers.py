"""
_tree_helpers.py
----------------
Internal helpers for decision / regression tree models:
  - Gini impurity
  - Entropy & Information Gain
  - Variance reduction (for regression)
  - Best-split search
  - Node data structure
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ── Impurity Measures ─────────────────────────────────────────────────────────

def gini_impurity(y: np.ndarray) -> float:
    """Gini = 1 - sum(p_i^2)"""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))


def entropy(y: np.ndarray) -> float:
    """H = -sum( p_i log_2(p_i))"""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray,
                     criterion: str = "entropy") -> float:
    """
    IG = H(parent) - weighted_avg(H(children))
    Works with both 'entropy' and 'gini'.
    """
    fn = entropy if criterion == "entropy" else gini_impurity
    n = len(parent)
    if n == 0:
        return 0.0
    weighted_child = (len(left) / n) * fn(left) + (len(right) / n) * fn(right)
    return fn(parent) - weighted_child


def variance_reduction(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Used by regression trees: reduction in variance after a split."""
    n = len(parent)
    if n == 0:
        return 0.0
    w_var = (len(left) / n) * np.var(left) + (len(right) / n) * np.var(right)
    return float(np.var(parent) - w_var)


# ── Split Search ──────────────────────────────────────────────────────────────

def best_split(X: np.ndarray, y: np.ndarray,
               criterion: str = "gini",
               task: str = "classification",
               max_features: int | None = None):
    """
    Exhaustively search for the best (feature_index, threshold) split.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    criterion : 'gini' | 'entropy'  (ignored for regression)
    task : 'classification' | 'regression'
    max_features : randomly subsample this many features (None = all)

    Returns
    -------
    best_feat : int
    best_thresh : float
    best_gain : float
    """
    n_samples, n_features = X.shape
    feature_indices = np.arange(n_features)

    if max_features is not None and max_features < n_features:
        feature_indices = np.random.choice(n_features, max_features, replace=False)

    best_gain = -np.inf
    best_feat, best_thresh = None, None

    for feat in feature_indices:
        thresholds = np.unique(X[:, feat])
        for thresh in thresholds:
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            if task == "regression":
                gain = variance_reduction(y, y[left_mask], y[right_mask])
            else:
                gain = information_gain(y, y[left_mask], y[right_mask], criterion)

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


# ── Tree Node ─────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """A single node in a decision / regression tree."""
    feature_index: int | None = None
    threshold: float | None = None
    left: Any = None          # TreeNode | None
    right: Any = None         # TreeNode | None
    value: Any = None         # leaf prediction
    n_samples: int = 0
    impurity: float = 0.0


def is_leaf(node: TreeNode) -> bool:
    return node.value is not None
