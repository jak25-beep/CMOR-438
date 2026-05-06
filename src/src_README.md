# src — Algorithm Implementations

This folder contains all machine learning algorithm implementations. Every file is written in pure NumPy with no scikit-learn models or other ML libraries under the hood.

---

## Structure

```
src/
├── supervised/             # Models that learn from labelled data
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── perceptron.py
│   ├── multilayer_perceptron.py
│   ├── k_nearest_neighbors.py
│   ├── decision_trees.py
│   ├── regression_trees.py
│   ├── ensemble_methods.py
│   ├── gradient_descent.py     # Shared GD base class
│   ├── _linear_helpers.py      # Shared loss functions and solvers
│   └── _tree_helpers.py        # Shared tree node and split utilities
│
└── unsupervised/           # Models that learn from unlabelled data
    ├── k_means_clustering.py
    ├── dbscan.py
    ├── pca.py
    └── community_detection.py
```

---

## Supervised Learning (`src/supervised/`)

Models in this folder learn a mapping from input features to a known target variable. See `src/supervised/README.md` for details on each algorithm.

| File | Class(es) | Task |
|---|---|---|
| `linear_regression.py` | `LinearRegression` | Regression — continuous target |
| `logistic_regression.py` | `LogisticRegression` | Binary classification |
| `perceptron.py` | `Perceptron` | Binary classification |
| `multilayer_perceptron.py` | `MLPClassifier` | Binary classification |
| `k_nearest_neighbors.py` | `KNNClassifier`, `KNNRegressor` | Classification and regression |
| `decision_trees.py` | `DecisionTreeClassifier` | Multi-class classification |
| `regression_trees.py` | `DecisionTreeRegressor` | Regression |
| `ensemble_methods.py` | `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingRegressor` | Classification and regression |

### Shared utilities

| File | Purpose |
|---|---|
| `gradient_descent.py` | `GradientDescentMixin` base class — shared by `LinearRegression` and `LogisticRegression` |
| `_linear_helpers.py` | Loss functions (MSE, BCE, R²), normal equation solvers, sigmoid |
| `_tree_helpers.py` | `TreeNode` dataclass, Gini/Entropy/Information Gain, best-split search |

---

## Unsupervised Learning (`src/unsupervised/`)

Models in this folder find structure in data without any labelled targets. See `src/unsupervised/README.md` for details on each algorithm.

| File | Class(es) | Task |
|---|---|---|
| `k_means_clustering.py` | `KMeans` | Clustering |
| `dbscan.py` | `DBSCAN` | Density-based clustering and outlier detection |
| `pca.py` | `PCA` | Dimensionality reduction |
| `community_detection.py` | `LabelPropagation` | Graph community detection |

---

## Usage

Import directly from the file:

```python
import sys
sys.path.insert(0, 'path/to/src/supervised')
from linear_regression import LinearRegression

model = LinearRegression(method='ols')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

Each class follows a consistent interface:

```python
model.fit(X, y)          # Train the model
model.predict(X)         # Generate predictions
model.score(X, y)        # Evaluate (R² for regression, accuracy for classification)
```
