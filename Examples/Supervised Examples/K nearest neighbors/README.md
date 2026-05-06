# K-Nearest Neighbours (KNN)

This package provides a from-scratch implementation of **K-Nearest Neighbours**, a simple but powerful **lazy learning** algorithm used for both classification and regression. It makes predictions by finding the k most similar training examples to a query point.

---

## Architecture and Mechanism

Unlike most algorithms, KNN has **no explicit training step** — it simply memorises the training data. At prediction time, it searches for the k closest points and aggregates their labels or values.

```
Query point → Compute distances to all training points → Find k nearest → Aggregate
```

---

## 1. Structure

| **Component** | **Role** |
|---|---|
| **Training** | Store X and y — nothing else |
| **Distance computation** | Measure similarity to every training point |
| **Neighbour selection** | Keep the k closest points |
| **Aggregation** | Classifier: vote. Regressor: average. |

---

## 2. Distance Metrics

| **Metric** | **Formula** | **Best for** |
|---|---|---|
| Euclidean | `√Σ(xᵢ - yᵢ)²` | Continuous, similarly-scaled features |
| Manhattan | `Σ\|xᵢ - yᵢ\|` | Robust to outliers, grid-like data |
| Minkowski | `(Σ\|xᵢ - yᵢ\|ᵖ)^(1/p)` | Generalises both with parameter p |

---

## 3. Voting Strategies

| **Mode** | **Method** | **Effect** |
|---|---|---|
| `weights='uniform'` | Each neighbour gets one equal vote | Simple majority |
| `weights='distance'` | Closer neighbours get more weight: `1/distance` | Smoother, more accurate |

---

## 4. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `k` | `5` | Number of neighbours to consider |
| `metric` | `'euclidean'` | Distance function to use |
| `weights` | `'uniform'` | How to weight neighbour contributions |

---

## 5. Choosing k

- **Small k** → flexible, can overfit (sensitive to noise and outliers)
- **Large k** → smoother predictions, can underfit (ignores local structure)
- Always **sweep k values** and choose by cross-validation or a held-out validation set

---

## 6. Computational Complexity

| **Phase** | **Cost** |
|---|---|
| Training | O(1) — just stores data |
| Prediction (per query) | O(n × d) — must compare to all training points |

For large datasets, consider approximate nearest-neighbour methods (e.g. KD-trees).

---

## 7. When to Use It

- **Small to medium datasets** — prediction slows significantly on large data
- **Non-linear decision boundaries** that change locally
- Quick, interpretable **baseline** with no assumptions about data distribution
- Always **normalise features first** — KNN is sensitive to scale
