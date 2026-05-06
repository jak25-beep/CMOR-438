# Decision Tree Regressor

This package provides a from-scratch implementation of a **Decision Tree Regressor**, which applies the same tree-building logic as the classifier but predicts **continuous values** instead of class labels. Each leaf returns the mean of all training targets that fell into it.

---

## Architecture and Mechanism

The regressor recursively partitions the feature space into rectangular regions. Within each region (leaf), it predicts the mean of the training targets. This produces a **piecewise constant** approximation of any function.

```
         [alcohol ≤ 10.5?]
         /              \
    [pH ≤ 3.3?]        → predict 6.8
    /         \
→ predict 5.1  → predict 5.9
```

---

## 1. Structure

- **Internal node:** tests one feature against a threshold — routes samples left (≤) or right (>)
- **Leaf node:** returns `mean(y)` of all training samples that reached it
- **Depth:** controls how finely the feature space is partitioned

---

## 2. Splitting Criterion — Variance Reduction

Unlike the classifier which minimises impurity, the regressor minimises **variance** in the child nodes:

| **Component** | **Formula** | **Purpose** |
|---|---|---|
| Node variance | `Var(y) = (1/n) Σ(yᵢ - ȳ)²` | Measures spread of target values |
| Variance Reduction | `Var(parent) - [n_L/n × Var(left) + n_R/n × Var(right)]` | Gain from a candidate split |
| Leaf prediction | `ŷ = mean(y)` | Best constant prediction for a region |

The split that produces the **greatest variance reduction** is chosen at each node.

---

## 3. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `max_depth` | `None` | Controls tree complexity — primary lever for overfitting |
| `min_samples_split` | `2` | Minimum samples required to attempt a split |
| `min_samples_leaf` | `1` | Minimum samples required in each resulting leaf |
| `max_features` | `None` | Feature subsampling (used by Random Forest) |

---

## 4. Evaluation Metrics

| **Metric** | **Formula** | **Interpretation** |
|---|---|---|
| R² | `1 - SS_res / SS_tot` | 1.0 = perfect, 0.0 = predicts mean, <0 = worse than mean |
| MSE | `(1/n) Σ(y - ŷ)²` | Average squared error — lower is better |

---

## 5. Strengths and Limitations

| **Strengths** | **Limitations** |
|---|---|
| Handles non-linear relationships | High variance — small data changes → very different trees |
| No feature scaling required | Poor extrapolation beyond training range |
| Naturally interpretable | Single trees often overfit — prefer Random Forest in practice |

---

## 6. When to Use It

- **Non-linear regression** where a linear model clearly underfits
- As a **building block** inside Random Forest or Gradient Boosting
- When you need a quick, interpretable model with no preprocessing
- Always set `max_depth` or `min_samples_leaf` to avoid memorising the training data
