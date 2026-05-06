"""
decision_trees.py
-----------------
DecisionTreeClassifier — recursive binary tree for classification.

Supports:
  - criterion: 'gini' | 'entropy'
  - max_depth, min_samples_split, min_samples_leaf
  - max_features (random subspace — used by Random Forest)
  - Feature importance

Usage
-----
from supervised_learning.decision_trees import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="gini", max_depth=10)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.feature_importances_)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
from _tree_helpers import TreeNode, best_split, is_leaf


class DecisionTreeClassifier:
    """
    Binary Decision Tree Classifier.

    Parameters
    ----------
    criterion : 'gini' | 'entropy'
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    max_features : int | None  (None = use all features)
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
    ):
        assert criterion in ("gini", "entropy")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.root_: TreeNode | None = None
        self.n_features_: int = 0
        self.feature_importances_: np.ndarray | None = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self._importance = np.zeros(self.n_features_)

        self.root_ = self._build(X, y, depth=0)

        # Normalise importances
        total = self._importance.sum()
        self.feature_importances_ = (
            self._importance / total if total > 0 else self._importance
        )
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = len(y)
        node = TreeNode(n_samples=n_samples)

        # Stopping conditions
        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        )

        if stop:
            node.value = self._leaf_value(y)
            return node

        feat, thresh, gain = best_split(
            X, y,
            criterion=self.criterion,
            task="classification",
            max_features=self.max_features,
        )

        if feat is None or gain <= 0:
            node.value = self._leaf_value(y)
            return node

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        # Enforce min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = self._leaf_value(y)
            return node

        # Track importance (weighted gain × n_samples)
        self._importance[feat] += gain * n_samples

        node.feature_index = feat
        node.threshold = thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    @staticmethod
    def _leaf_value(y: np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    def _traverse(self, x: np.ndarray, node: TreeNode):
        if is_leaf(node):
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft probabilities for the positive class (binary only)."""
        preds = self.predict(X)
        return preds.astype(float)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_depth(self) -> int:
        return self._depth(self.root_)

    def _depth(self, node: TreeNode | None) -> int:
        if node is None or is_leaf(node):
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def __repr__(self):
        return (f"DecisionTreeClassifier(criterion='{self.criterion}', "
                f"max_depth={self.max_depth})")
