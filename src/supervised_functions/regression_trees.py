"""
regression_trees.py
-------------------
DecisionTreeRegressor — specialised tree for continuous target prediction.

Uses variance-reduction as the split criterion.

Usage
-----
from supervised_learning.regression_trees import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5)
reg.fit(X_train, y_train)
print(reg.predict(X_test))
print(reg.score(X_test, y_test))   # R²
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
from _tree_helpers import TreeNode, best_split, is_leaf


class DecisionTreeRegressor:
    """
    Decision Tree Regressor (variance-reduction criterion).

    Parameters
    ----------
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    max_features : int | None
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.root_: TreeNode | None = None
        self.n_features_: int = 0
        self.feature_importances_: np.ndarray | None = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self._importance = np.zeros(self.n_features_)

        self.root_ = self._build(X, y, depth=0)

        total = self._importance.sum()
        self.feature_importances_ = (
            self._importance / total if total > 0 else self._importance
        )
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = len(y)
        node = TreeNode(n_samples=n_samples)

        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or np.var(y) == 0
        )

        if stop:
            node.value = float(np.mean(y))
            return node

        feat, thresh, gain = best_split(
            X, y,
            task="regression",
            max_features=self.max_features,
        )

        if feat is None or gain <= 0:
            node.value = float(np.mean(y))
            return node

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = float(np.mean(y))
            return node

        self._importance[feat] += gain * n_samples

        node.feature_index = feat
        node.threshold = thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    def _traverse(self, x: np.ndarray, node: TreeNode) -> float:
        if is_leaf(node):
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        return float(np.mean((y - self.predict(X)) ** 2))

    def get_depth(self) -> int:
        return self._depth(self.root_)

    def _depth(self, node) -> int:
        if node is None or is_leaf(node):
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def __repr__(self):
        return f"DecisionTreeRegressor(max_depth={self.max_depth})"
