# Community Detection — Label Propagation

This package provides a from-scratch implementation of the **Label Propagation Algorithm (LPA)** for community detection in graphs. It discovers groups of nodes that are more densely connected to each other than to the rest of the network — without requiring the number of communities to be specified in advance.

---

## Architecture and Mechanism

Label Propagation works by letting labels "spread" through the graph. Nodes adopt the most common label among their neighbours. After many rounds, densely connected groups converge to the same label — forming a community.

```
Each node gets its own label → Labels spread via neighbour voting → Convergence = communities
```

---

## 1. Input — The Adjacency Matrix

The algorithm operates on a graph represented as an **adjacency matrix** A:

| **Entry** | **Meaning** |
|---|---|
| `A[i][j] > 0` | Nodes i and j are connected (value = edge weight) |
| `A[i][j] = 0` | No connection between i and j |
| `A[i][i] = 0` | No self-loops |

For non-graph data (e.g. customer records), build a similarity graph:
```
A[i][j] = 1   if distance(xᵢ, xⱼ) < threshold
A[i][j] = 0   otherwise
```

---

## 2. Algorithm

```
1. Initialise: assign each node a unique label
2. Repeat until convergence:
   for each node i (in random order to break ties):
       count weighted votes from all neighbours
       assign i the label with the highest total weight
3. Stop when no labels change between rounds
```

The random update order is important — it breaks ties stochastically, which prevents the algorithm from getting stuck.

---

## 3. Key Parameters

| **Parameter** | **Default** | **What it does** |
|---|---|---|
| `max_iter` | `100` | Maximum label update rounds |
| `random_state` | `None` | Seed for reproducible update order |

---

## 4. Evaluating Community Quality — Modularity

Modularity Q measures whether the detected communities have more internal edges than expected by chance:

```
Q = (1/2m) Σᵢⱼ [ Aᵢⱼ - kᵢkⱼ/2m ] × δ(cᵢ, cⱼ)
```

Where `m` = total edge weight, `kᵢ` = degree of node i, `δ` = 1 if same community. Ranges from **-1 to 1** — values above **0.3** typically indicate meaningful community structure.

---

## 5. When to Use It

- **Social network analysis** — finding friend groups, research communities
- **Gene co-expression networks** — grouping genes that activate together
- Any dataset where a **similarity graph** can be constructed from features
- When you want natural groupings **without specifying k** in advance
- Very fast — typically converges in fewer than 20 iterations on real-world graphs
