# Logistic Regression

This package provides a from-scratch implementation of **Logistic Regression**, a fundamental supervised learning algorithm for **binary classification**. Despite its name, it predicts the *probability* that a sample belongs to a class, not a continuous value.

---

## Architecture and Mechanism

Logistic Regression applies a linear combination of features, then passes the result through a **sigmoid function** to produce a probability in (0, 1):

```
z = w·x + b
P(y=1 | x) = σ(z) = 1 / (1 + e⁻ᶻ)
```

A threshold (default 0.5) converts the probability into a hard class label.

---

## 1. Structure

The model has three components working in sequence:

- **Linear layer:** computes the raw score `z = Xw + b`
- **Sigmoid activation:** squashes z into a probability between 0 and 1
- **Decision threshold:** converts probability to a class label {0, 1}

---

## 2. Activation and Optimization

| **Component** | **Function** | **Purpose** |
|---|---|---|
| Activation | Sigmoid: `1 / (1 + e⁻ᶻ)` | Converts raw score to probability |
| Loss | Binary Cross-Entropy | Penalises confident wrong predictions |
| Optimizer | Gradient Descent | Iteratively minimises the loss |
| Regularisation | L2 penalty (optional) | Prevents overfitting by shrinking weights |

**Binary Cross-Entropy Loss:**
```
L = -(1/n) Σ [ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ) ]
```

**Gradient update:**
```
w = w - lr × (1/n) Xᵀ(ŷ - y)
b = b - lr × (1/n) Σ(ŷ - y)
```

---

## 3. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `learning_rate` | `0.01` | GD step size |
| `n_iterations` | `1000` | Training epochs |
| `threshold` | `0.5` | Probability cutoff for class 1 |
| `l2` | `0.0` | L2 regularisation strength (0 = none) |
| `batch_size` | `None` | Mini-batch size for GD |

---

## 4. Evaluation Metrics

**Accuracy** — fraction of correctly classified samples:
```
Accuracy = (TP + TN) / n
```

**Probability calibration** — the predicted probabilities should reflect true likelihoods. A well-calibrated model that predicts 0.8 should be correct ~80% of the time.

---

## 5. When to Use It

- **Binary classification** tasks (spam/not spam, high quality/low quality, disease/healthy)
- When you need **probability outputs**, not just class labels
- As a fast, interpretable **baseline** before trying more complex models
- When the decision boundary is expected to be roughly **linear**
