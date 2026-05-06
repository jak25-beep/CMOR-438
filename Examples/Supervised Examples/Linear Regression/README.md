# Linear Regression

This package provides a from-scratch implementation of **Linear Regression**, one of the foundational algorithms in supervised machine learning, used to model the relationship between input features and a **continuous target variable**.

---

## Architecture and Mechanism

Linear Regression fits a weighted sum of input features to predict a target value:

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

The goal is to find weights **w** and bias **b** that minimise prediction error across the training set. Three fitting strategies are provided.

---

## 1. Fitting Methods

### Method 1 — Ordinary Least Squares (OLS)
Solves for the exact minimum of the Mean Squared Error in a single closed-form step using the **normal equation**:

```
w = (XᵀX)⁻¹ Xᵀy
```

No learning rate or iterations needed. Best for small-to-medium datasets where an exact solution is affordable.

### Method 2 — Ridge Regression
Extends OLS with an **L2 penalty** on the weights to prevent overfitting when features are correlated or there are many of them:

```
w = (XᵀX + αI)⁻¹ Xᵀy
```

The `alpha` parameter controls regularisation strength. Higher alpha shrinks weights toward zero, producing a simpler model.

### Method 3 — Gradient Descent
Iteratively adjusts weights by moving in the direction that reduces loss:

```
w = w - lr × (2/n) Xᵀ(Xw - y)
b = b - lr × (2/n) Σ(Xw - y)
```

Supports **full batch**, **stochastic (SGD)**, and **mini-batch** modes. Slower than the normal equation but scales to very large datasets.

---

## 2. Activation and Optimization

| **Component** | **Function** | **Purpose** |
|---|---|---|
| Prediction | Linear: `ŷ = Xw + b` | Direct weighted sum of features |
| Loss (OLS/Ridge) | MSE: `(1/n) Σ(y - ŷ)²` | Measures average squared prediction error |
| Solver (OLS) | Normal equation | Exact closed-form solution |
| Solver (Ridge) | Regularised normal equation | Closed-form with L2 penalty |
| Solver (GD) | Gradient descent | Iterative update via gradients |

---

## 3. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `method` | `'ols'` | Fitting strategy: `'ols'`, `'ridge'`, or `'gd'` |
| `alpha` | `1.0` | Ridge penalty strength (larger = more shrinkage) |
| `learning_rate` | `0.01` | GD step size (too large = diverges, too small = slow) |
| `n_iterations` | `1000` | Number of GD update steps |
| `batch_size` | `None` | GD batch size — `None`=full, `1`=SGD, `k`=mini-batch |

---

## 4. Evaluation Metrics

**R² (Coefficient of Determination)** — measures how well the model explains variance in the target:
```
R² = 1 - SS_res / SS_tot
```
- **1.0** = perfect fit
- **0.0** = no better than predicting the mean
- **< 0** = worse than predicting the mean

**MSE (Mean Squared Error)** — average squared difference between predictions and true values. Lower is better; in the same units as y².

---

## 5. When to Use It

- Target variable is **continuous** (price, temperature, quality score)
- You expect a roughly **linear relationship** between features and target
- Use **Ridge** when you have many correlated features or suspect multicollinearity
- Use **GD** when the dataset is too large for matrix inversion
