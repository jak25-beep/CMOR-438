# Decision Tree Classifier

This package provides a from-scratch implementation of a **Decision Tree Classifier**, a supervised learning algorithm that learns a hierarchy of if/else rules directly from data to separate classes as cleanly as possible.

---

## Architecture and Mechanism

A decision tree recursively splits the training data at each node by choosing the feature and threshold that best separates the classes. The result is a binary tree where each **internal node** is a rule, and each **leaf** returns a class prediction.

```
         [alcohol ≤ 10.5?]
         /              \
    [sulphates ≤ 0.6?]  → High Quality
    /           \
→ Low Quality   → Mid Quality
```

---

## 1. Structure

- **Internal node:** tests one feature against a threshold — routes samples left (≤) or right (>)
- **Leaf node:** stores the majority class of all training samples that reached it
- **Depth:** the longest path from root to leaf — controls model complexity

---

## 2. Splitting Criteria

| **Criterion** | **Formula** | **Purpose** |
|---|---|---|
| Gini Impurity | `1 - Σ pᵢ²` | Probability of mislabelling a random sample |
| Entropy | `-Σ pᵢ log₂(pᵢ)` | Information content of the class distribution |
| Information Gain | `H(parent) - weighted_avg(H(children))` | Reduction in impurity after a split |

The split that maximises **Information Gain** is chosen at every node.

---

## 3. Tree-Building Algorithm

```
function build(data, depth):
    if stopping condition met:
        return leaf(majority class)
    
    for each feature, for each threshold:
        compute Information Gain
    
    split on best (feature, threshold)
    node.left  = build(left_data,  depth + 1)
    node.right = build(right_data, depth + 1)
    return node
```

**Stopping conditions:** max depth reached, fewer than `min_samples_split` samples, or all samples already have the same class.

---

## 4. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `criterion` | `'gini'` | Split quality measure: `'gini'` or `'entropy'` |
| `max_depth` | `None` | Maximum tree depth — primary control for overfitting |
| `min_samples_split` | `2` | Minimum samples required to attempt a split |
| `min_samples_leaf` | `1` | Minimum samples required in each resulting leaf |
| `max_features` | `None` | Feature subsampling at each split (used by Random Forest) |

---

## 5. Feature Importance

Each feature is scored by the **total weighted impurity reduction** it causes across all splits it appears in, normalised so all importances sum to 1. Useful for understanding which features drive predictions.

---

## 6. When to Use It

- You need an **interpretable** model — the rules can be printed and read
- Data has **mixed feature types** (no scaling needed)
- As a **building block** for ensemble methods like Random Forest and Gradient Boosting
- Be aware: single trees **overfit easily** — tune `max_depth` or use an ensemble
