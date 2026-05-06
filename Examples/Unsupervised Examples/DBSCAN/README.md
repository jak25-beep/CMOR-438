# DBSCAN

This package provides a from-scratch implementation of **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise), a clustering algorithm that discovers clusters of **arbitrary shape** and explicitly marks outliers as **noise** — without requiring you to specify the number of clusters in advance.

---

## Architecture and Mechanism

DBSCAN defines clusters as **dense regions** of points separated by sparser regions. It classifies every point into one of three categories before building clusters.

```
Dense core points → connected via BFS → form a cluster
Sparse border points → attached to nearest cluster
Isolated noise points → labelled -1
```

---

## 1. Point Classification

For parameters `eps` (neighbourhood radius) and `min_samples`:

| **Point Type** | **Definition** | **Label** |
|---|---|---|
| **Core point** | Has ≥ `min_samples` neighbours within distance `eps` | cluster id |
| **Border point** | Within `eps` of a core point, but not a core point itself | cluster id (of nearest core) |
| **Noise point** | Neither core nor reachable from a core point | **-1** |

---

## 2. Algorithm

```
for each unvisited point p:
    find all points within eps of p (its neighbourhood)

    if |neighbourhood| < min_samples:
        label p as noise (-1)
    else:
        start a new cluster
        BFS: expand cluster by visiting all density-reachable points
             (neighbours of neighbours, if they are also core points)
```

Two points are **density-reachable** if there is a chain of core points connecting them, each within `eps` of the next.

---

## 3. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `eps` | `0.5` | Neighbourhood radius — smaller = tighter clusters, more noise |
| `min_samples` | `5` | Minimum neighbours to qualify as a core point |
| `metric` | `'euclidean'` | Distance function: `'euclidean'` or `'manhattan'` |

**Tuning guidance:**
- Plot a **k-distance graph** — sort distances to each point's k-th neighbour, look for an elbow — that value is a good `eps`
- A common starting point for `min_samples` is `2 × n_features`

---

## 4. Comparison to K-Means

| **Feature** | **DBSCAN** | **K-Means** |
|---|---|---|
| Number of clusters | Discovered automatically | Must specify k in advance |
| Cluster shape | Arbitrary | Assumes spherical |
| Outlier handling | Explicitly labels noise | Assigns all points to a cluster |
| Sensitivity | eps and min_samples | k and initialisation |

---

## 5. When to Use It

- Data has **irregularly shaped clusters** (rings, crescents, blobs of varying density)
- You **don't know k** in advance
- You expect **outliers** and want them identified, not forced into a cluster
- Struggles with **very high-dimensional data** or **clusters of widely varying density**
