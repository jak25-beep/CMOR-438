# Principal Component Analysis (PCA)

This package provides a from-scratch implementation of **Principal Component Analysis**, a dimensionality reduction technique that finds the directions of **maximum variance** in the data and projects onto them — transforming correlated features into a smaller set of uncorrelated **principal components**.

---

## Architecture and Mechanism

PCA finds a new coordinate system aligned with the directions of greatest spread in the data. The first principal component (PC1) explains the most variance, PC2 the second most, and so on. All components are orthogonal to each other.

```
Centre data → Covariance matrix → Eigendecomposition → Project onto top k eigenvectors
```

---

## 1. Algorithm — Step by Step

**Step 1 — Centre the data:**
```
X_centred = X - mean(X, axis=0)
```

**Step 2 — Compute the covariance matrix:**
```
C = (1 / (n-1)) × X_centred.T @ X_centred
```
C is a symmetric (n_features × n_features) matrix where Cᵢⱼ is the covariance between features i and j.

**Step 3 — Eigendecomposition:**
```
C = V Λ Vᵀ
```
Eigenvalues (Λ) measure how much variance each direction explains. Eigenvectors (V) define those directions. Sorted by eigenvalue descending.

**Step 4 — Project onto top k components:**
```
X_reduced = X_centred @ V[:, :k]    →  shape: (n_samples, k)
```

---

## 2. Key Components

| **Component** | **What it is** | **Shape** |
|---|---|---|
| `components_` | The k principal component directions (eigenvectors) | `(k, n_features)` |
| `explained_variance_` | Variance explained by each component (eigenvalues) | `(k,)` |
| `explained_variance_ratio_` | Fraction of total variance per component | `(k,)` |
| `mean_` | Per-feature mean subtracted during centring | `(n_features,)` |

---

## 3. Key Parameters

| **Parameter** | **Options** | **What it does** |
|---|---|---|
| `n_components` | int | Keep exactly this many components |
| `n_components` | float (0–1) | Keep enough components to explain this fraction of variance |
| `n_components` | `None` | Keep all components (no reduction) |

---

## 4. Explained Variance Ratio

The EVR tells you what fraction of total information each component captures:
```
EVR[i] = λᵢ / Σλ
```

**Scree plot** — bar chart of EVR by component — helps choose how many components to keep. Look for the "elbow" where additional components add little new information.

---

## 5. When to Use It

- **Visualising** high-dimensional data in 2D or 3D (project onto PC1 and PC2)
- **Noise removal** — keep top components, discard small-variance directions
- **Speeding up** downstream models by reducing input dimensionality
- **Removing multicollinearity** before linear regression
- Always apply PCA to **scaled data** — features with larger scales will dominate the covariance matrix
