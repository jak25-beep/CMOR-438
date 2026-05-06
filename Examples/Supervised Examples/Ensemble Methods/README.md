# Ensemble Methods

This package provides from-scratch implementations of three ensemble learning algorithms: **Random Forest Classifier**, **Random Forest Regressor**, and **Gradient Boosting Regressor**. Ensemble methods combine multiple weak models to produce one strong model that is more accurate and robust than any individual learner.

---

## Architecture and Mechanism

All three methods are built on top of decision trees. They differ in *how* trees are combined:

| **Method** | **Strategy** | **Trees are...** |
|---|---|---|
| Random Forest | Parallel (bagging) | Trained independently, combined by voting/averaging |
| Gradient Boosting | Sequential (boosting) | Each tree corrects the errors of the previous ones |

---

## 1. Random Forest — Bagging + Random Subspace

### How it works

Two sources of randomness reduce variance and prevent trees from being correlated:

**Step 1 — Bootstrap sampling (Bagging):**
```
For each tree:
    Draw n samples WITH replacement from the training set
    Train a tree on this bootstrap sample
```

**Step 2 — Random feature subspace:**
```
At each node split:
    Randomly select max_features from all features
    Only consider these features for the split
```

**Step 3 — Aggregate predictions:**
```
Classifier:  Hard vote — majority class wins
Regressor:   Average predictions across all trees
```

### Why it works
Individual trees overfit, but they overfit in *different directions* because they see different data and different features. Averaging many uncorrelated trees cancels out the individual errors.

---

## 2. Gradient Boosting Regressor — Sequential Residual Fitting

### How it works

Trees are added one at a time, each correcting what the previous ensemble got wrong:

```
F₀ = mean(y)                              ← initial prediction

For each round t = 1 to n_estimators:
    residuals  = y - F_{t-1}(X)           ← what's left to explain
    tree_t     = DecisionTree(X, residuals) ← fit to the errors
    F_t        = F_{t-1} + lr × tree_t    ← update the ensemble
```

The `learning_rate` (shrinkage) multiplies each tree's contribution — smaller values require more trees but generalise better.

---

## 3. Key Parameters

| **Parameter** | **Applies to** | **What it does** |
|---|---|---|
| `n_estimators` | All | Number of trees to build |
| `max_depth` | All | Depth of each individual tree |
| `max_features` | Random Forest | Features per split: `'sqrt'`, `'log2'`, int, or None |
| `criterion` | RF Classifier | Split quality: `'gini'` or `'entropy'` |
| `learning_rate` | Gradient Boosting | Shrinkage factor — smaller = more trees needed |
| `subsample` | Gradient Boosting | Fraction of data per round (< 1.0 = stochastic GB) |
| `random_state` | All | Seed for reproducibility |

---

## 4. When to Use Them

| **Algorithm** | **Best for** |
|---|---|
| Random Forest Classifier | Robust classification baseline — works well with minimal tuning |
| Random Forest Regressor | Regression on tabular data — fast and reliable |
| Gradient Boosting | When you need the highest accuracy and are willing to tune hyperparameters |
