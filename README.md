# CMOR-438 — Machine Learning from Scratch

A pure-NumPy implementation of common supervised and unsupervised machine learning algorithms, built from the ground up without relying on scikit-learn or other ML libraries. Each algorithm includes a clean implementation, a conceptual README, worked examples on real datasets, and unit tests.

---

## Repository Structure

```
CMOR-438/
│
├── src/                        # Algorithm implementations
│   ├── supervised/             # Supervised learning models
│   └── unsupervised/           # Unsupervised learning models
│
├── examples/                   # Jupyter notebook walkthroughs
│   ├── supervised/             # One notebook per supervised algorithm
│   └── unsupervised/           # One notebook per unsupervised algorithm
│
├── tests/                      # Unit test suites
│   ├── supervised/             # Tests for supervised models
│   └── unsupervised/           # Tests for unsupervised models
│
├── data/
│   ├── WineQT.csv              # Used by all supervised examples
│   └── Mall_Customers.csv      # Used by all unsupervised examples
│
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## Algorithms

### Supervised Learning

| Algorithm | Source | Example | Tests |
|---|---|---|---|
| Linear Regression (OLS, Ridge, GD) | `src/supervised/linear_regression.py` | `examples/supervised/` | `tests/supervised/` |
| Logistic Regression | `src/supervised/logistic_regression.py` | `examples/supervised/` | `tests/supervised/` |
| Perceptron | `src/supervised/perceptron.py` | `examples/supervised/` | `tests/supervised/` |
| Multilayer Perceptron (MLP) | `src/supervised/multilayer_perceptron.py` | `examples/supervised/` | `tests/supervised/` |
| K-Nearest Neighbours | `src/supervised/k_nearest_neighbors.py` | `examples/supervised/` | `tests/supervised/` |
| Decision Tree Classifier | `src/supervised/decision_trees.py` | `examples/supervised/` | `tests/supervised/` |
| Decision Tree Regressor | `src/supervised/regression_trees.py` | `examples/supervised/` | `tests/supervised/` |
| Random Forest + Gradient Boosting | `src/supervised/ensemble_methods.py` | `examples/supervised/` | `tests/supervised/` |

### Unsupervised Learning

| Algorithm | Source | Example | Tests |
|---|---|---|---|
| K-Means Clustering | `src/unsupervised/k_means_clustering.py` | `examples/unsupervised/` | `tests/unsupervised/` |
| DBSCAN | `src/unsupervised/dbscan.py` | `examples/unsupervised/` | `tests/unsupervised/` |
| PCA | `src/unsupervised/pca.py` | `examples/unsupervised/` | `tests/unsupervised/` |
| Community Detection (Label Propagation) | `src/unsupervised/community_detection.py` | `examples/unsupervised/` | `tests/unsupervised/` |

---

## Datasets

**Wine Quality (`data/WineQT.csv`)** — 1,143 red wine samples with 11 physicochemical features and a quality score from 3 to 8. Used for regression and classification tasks in all supervised examples.

**Mall Customers (`data/Mall_Customers.csv`)** — 200 mall shoppers described by Age, Annual Income, and Spending Score. Used for clustering and dimensionality reduction in all unsupervised examples.

---

## Getting Started

### Install dependencies

```bash
pip install numpy
```

For running examples and tests:

```bash
pip install -r requirements-dev.txt
```

### Run an example

```bash
cd examples/supervised
jupyter notebook example_linear_regression.ipynb
```

### Run all tests

```bash
pytest tests/ -v
```

### Run a single test file

```bash
pytest tests/supervised/test_linear_regression.py -v
```

---

## Requirements

| Package | Purpose |
|---|---|
| `numpy` | Core dependency — all algorithm implementations |
| `matplotlib` | Plots in example notebooks |
| `pandas` | Data loading in example notebooks |
| `scikit-learn` | Preprocessing utilities (`StandardScaler`, `train_test_split`) in examples |
| `jupyter` | Running `.ipynb` example notebooks |
| `pytest` | Running the test suite |

---

## Design Philosophy

Every algorithm is implemented using only NumPy — no calls to scikit-learn models, TensorFlow, or any other ML library. The goal is to make the internals of each algorithm transparent and readable. Each file is self-contained and extensively commented so it can be read alongside the corresponding README.
This is a template
This message should be added in with changes.
