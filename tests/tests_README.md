# tests — Unit Test Suites

This folder contains unit tests for every algorithm in the library. Tests are written to run with `pytest` and cover correctness, edge cases, and expected behaviour under a range of inputs.

---

## Structure

```
tests/
├── supervised/             # Tests for supervised learning algorithms
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_perceptron.py
│   ├── test_multilayer_perceptron.py
│   ├── test_k_nearest_neighbors.py
│   ├── test_decision_trees.py
│   ├── test_regression_trees.py
│   └── test_ensemble_methods.py
│
└── unsupervised/           # Tests for unsupervised learning algorithms
    ├── test_k_means_clustering.py
    ├── test_dbscan.py
    ├── test_pca.py
    └── test_community_detection.py
```

---

## Running the Tests

**Run the entire test suite:**

```bash
pytest tests/ -v
```

**Run only supervised tests:**

```bash
pytest tests/supervised/ -v
```

**Run only unsupervised tests:**

```bash
pytest tests/unsupervised/ -v
```

**Run a single file:**

```bash
pytest tests/supervised/test_linear_regression.py -v
```

**Run a single test function:**

```bash
pytest tests/supervised/test_linear_regression.py::test_ols_perfect_fit -v
```

---

## What Is Tested

Each test file generates small synthetic datasets internally — no external data files are required to run the tests. See `tests/supervised/README.md` and `tests/unsupervised/README.md` for a breakdown of what each file covers.

**Total tests: 109** across 12 algorithm test files.

---

## Dependencies

```bash
pip install numpy pytest
```

No other packages are required to run the tests.
