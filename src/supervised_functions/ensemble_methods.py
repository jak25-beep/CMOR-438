"""
ensemble_methods.py
-------------------
Ensemble estimators built on top of the tree primitives:

  RandomForestClassifier  — bagging + random feature subspace
  RandomForestRegressor   — same, for continuous targets
  GradientBoostingRegressor — sequential residual-fitting boosting

Usage
-----
from supervised_learning.ensemble_methods import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
print(rf.predict(X_test))

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gbr.fit(X_train, y_train)
print(gbr.predict(X_test))
"""

import numpy as np
from collections import Counter
import sys, os
_sl = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(_sl, 'decision_trees'))
sys.path.insert(0, os.path.join(_sl, 'regression_trees'))
sys.path.insert(0, os.path.join(_sl, '_shared'))
from decision_trees import DecisionTreeClassifier
from regression_trees import DecisionTreeRegressor


# ── Utilities ─────────────────────────────────────────────────────────────────

def _bootstrap(X: np.ndarray, y: np.ndarray, random_state=None):
    rng = np.random.RandomState(random_state)
    n = len(y)
    idx = rng.choice(n, n, replace=True)
    return X[idx], y[idx]


# ── Random Forest Classifier ──────────────────────────────────────────────────

class RandomForestClassifier:
    """
    Random Forest Classifier (bagging + random subspace).

    Parameters
    ----------
    n_estimators : number of trees
    max_depth : max depth of each tree
    min_samples_split : int
    min_samples_leaf : int
    max_features : features considered at each split
                   'sqrt' (default), 'log2', int, or None (all)
    criterion : 'gini' | 'entropy'
    random_state : int | None
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str = "sqrt",
        criterion: str = "gini",
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.feature_importances_: np.ndarray | None = None

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if mf is None:
            return n_features
        return int(mf)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_features = X.shape[1]
        mf = self._resolve_max_features(n_features)
        self.estimators_ = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            seed = None if self.random_state is None else self.random_state + i
            X_b, y_b = _bootstrap(X, y, random_state=seed)
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
            )
            tree.fit(X_b, y_b)
            self.estimators_.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / importances.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])  # (n_trees, n_samples)
        # Majority vote
        return np.array([Counter(col).most_common(1)[0][0] for col in all_preds.T])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self):
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, max_depth={self.max_depth})"


# ── Random Forest Regressor ───────────────────────────────────────────────────

class RandomForestRegressor:
    """
    Random Forest Regressor (bagging + random subspace).

    Parameters
    ----------
    n_estimators : int
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    max_features : 'sqrt' | 'log2' | int | None
    random_state : int | None
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str = "sqrt",
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []
        self.feature_importances_: np.ndarray | None = None

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if mf is None:
            return n_features
        return int(mf)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        mf = self._resolve_max_features(n_features)
        self.estimators_ = []
        importances = np.zeros(n_features)

        for i in range(self.n_estimators):
            seed = None if self.random_state is None else self.random_state + i
            X_b, y_b = _bootstrap(X, y, random_state=seed)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
            )
            tree.fit(X_b, y_b)
            self.estimators_.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / importances.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        return all_preds.mean(axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def __repr__(self):
        return f"RandomForestRegressor(n_estimators={self.n_estimators}, max_depth={self.max_depth})"


# ── Gradient Boosting Regressor ───────────────────────────────────────────────

class GradientBoostingRegressor:
    """
    Gradient Boosting for regression (L2 loss).

    Fits each new tree to the negative gradient (= residuals for MSE).

    Parameters
    ----------
    n_estimators : int
    learning_rate : float — shrinkage factor
    max_depth : int
    min_samples_split : int
    subsample : float — fraction of samples per iteration (stochastic GB)
    random_state : int | None
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state

        self.estimators_: list[DecisionTreeRegressor] = []
        self.initial_pred_: float = 0.0
        self.train_loss_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.RandomState(self.random_state)

        self.initial_pred_ = float(np.mean(y))
        F = np.full(len(y), self.initial_pred_)

        for i in range(self.n_estimators):
            residuals = y - F                   # negative gradient for MSE

            # Subsample
            if self.subsample < 1.0:
                n_sub = max(1, int(len(y) * self.subsample))
                idx = rng.choice(len(y), n_sub, replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sub, r_sub)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean(residuals ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.initial_pred_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))

    def __repr__(self):
        return (f"GradientBoostingRegressor(n_estimators={self.n_estimators}, "
                f"lr={self.learning_rate}, max_depth={self.max_depth})")
