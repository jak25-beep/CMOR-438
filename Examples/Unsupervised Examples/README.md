# K-Means Clustering

This package provides a from-scratch implementation of **K-Means Clustering**, one of the most widely used unsupervised learning algorithms. It partitions data into **k non-overlapping clusters** by alternating between assigning points to their nearest centroid and updating centroids to the mean of their cluster.

---

## Architecture and Mechanism

K-Means finds a set of k cluster centres (centroids) that minimise the total distance from each point to its assigned centroid. The algorithm iterates until the centroids stop moving.

```
Initialise k centroids → Assign points → Update centroids → Repeat until convergence
```

---

## 1. Algorithm (Lloyd's Algorithm)

```
1. Initialise k centroids (random or k-means++)
2. Repeat until convergence:
   a. Assign each point to the nearest centroid:
         label[i] = argmin_j ||x_i - centroid_j||²
   b. Update each centroid to the mean of its assigned points:
         centroid_j = mean(x where label == j)
3. Stop when centroid shift < tol
```

---

## 2. Initialisation Strategies

| **Method** | **How it works** | **Pros / Cons** |
|---|---|---|
| `'random'` | Picks k data points at random | Fast, but can converge to bad local minima |
| `'k-means++'` | Picks centroids one at a time, favouring far-apart points | Dramatically better starts, almost always preferred |

**K-Means++ selection probability:**
```
P(x chosen) ∝ min distance to existing centroids²
```

---

## 3. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `k` | `3` | Number of clusters |
| `init` | `'k-means++'` | Initialisation strategy |
| `n_init` | `10` | Random restarts — best result (lowest inertia) is kept |
| `max_iter` | `300` | Maximum update iterations per run |
| `tol` | `1e-4` | Convergence threshold for centroid shift |

---

## 4. Choosing k

**Elbow Method** — plot inertia vs k and look for the point where gains slow:
```
Inertia = Σ_clusters Σ_points ||x - centroid||²
```

**Silhouette Score** — measures how similar a point is to its own cluster vs the nearest other cluster:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where `a` = mean distance to same-cluster points, `b` = mean distance to nearest other cluster. Ranges from **-1 to 1** — higher is better.

---

## 5. When to Use It

- Discovering **natural groupings** in unlabelled data
- **Customer segmentation**, document clustering, image compression
- When clusters are roughly **spherical and similarly sized**
- Always **normalise features** first — K-Means is sensitive to scale
