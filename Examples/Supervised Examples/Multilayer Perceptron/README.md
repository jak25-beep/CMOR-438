# Multilayer Perceptron (MLP) Binary Classifier

This package provides a from-scratch implementation of a **Multilayer Perceptron (MLP)**, a feedforward artificial neural network used for complex **binary classification** tasks.

---

## Architecture and Mechanism

The MLP overcomes the linear separability limitations of the simple Perceptron by introducing one or more hidden layers, allowing it to model non-linear relationships.

```
Input Layer  →  Hidden Layer(s)  →  Output Layer
 (features)       (ReLU)            (Sigmoid → probability)
```

---

## 1. Structure

The network is composed of interconnected layers:

- **Input Layer:** Receives the feature data
- **Hidden Layers:** One or more layers that transform the data non-linearly using ReLU
- **Output Layer:** Produces the final prediction as a probability via Sigmoid

For each layer i, the forward computation is:
```
z[i] = a[i-1] @ W[i] + b[i]
a[i] = activation(z[i])
```

---

## 2. Activation and Optimization

| **Component** | **Function** | **Purpose** |
|---|---|---|
| Hidden Layer Activation | Rectified Linear Unit (ReLU): `max(0, z)` | Introduces non-linearity to learn complex patterns |
| Output Activation | Sigmoid: `1 / (1 + e⁻ᶻ)` | Converts final score to a probability in (0, 1) |
| Loss Function | Binary Cross-Entropy | Penalises confident wrong predictions heavily |
| Optimizer | Mini-batch Gradient Descent | Updates all weights using backpropagated gradients |
| Weight Init | He Initialisation: `N(0, √(2/fan_in))` | Prevents vanishing/exploding gradients with ReLU |

---

## 3. Backpropagation

Gradients flow backwards through the network using the chain rule:

```
δ[output] = ŷ - y                         (sigmoid + BCE, exact gradient)
δ[hidden] = (δ[next] @ W[next]ᵀ) × ReLU'(z)
dW[i]     = a[i-1]ᵀ @ δ[i] / n
db[i]     = mean(δ[i], axis=0)
```

Weights are then updated:
```
W[i] = W[i] - lr × dW[i]
b[i] = b[i] - lr × db[i]
```

---

## 4. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `hidden_layers` | `(64,)` | Tuple of neuron counts per hidden layer e.g. `(64, 32)` |
| `learning_rate` | `0.01` | Step size for gradient descent |
| `n_iterations` | `500` | Number of training epochs |
| `l2` | `0.0` | L2 regularisation coefficient on weights |
| `batch_size` | `None` | Mini-batch size — `None`=full batch, `k`=mini-batch |

---

## 5. When to Use It

- When **linear models underfit** — the decision boundary is non-linear
- Binary classification with **complex feature interactions**
- As a stepping stone to understanding **deep learning**
- When you have enough data to train a multi-layer model without overfitting
