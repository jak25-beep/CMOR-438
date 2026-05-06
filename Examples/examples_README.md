# examples — Worked Notebook Walkthroughs

This folder contains Jupyter notebook walkthroughs for every algorithm in the library. Each notebook demonstrates a complete end-to-end workflow — loading data, preprocessing, training, evaluating, and visualising results — on a real dataset.

---

## Structure

```
examples/
├── supervised/             # Notebooks using the Wine Quality dataset
│   ├── example_linear_regression.ipynb
│   ├── example_logistic_regression.ipynb
│   ├── example_perceptron.ipynb
│   ├── example_multilayer_perceptron.ipynb
│   ├── example_k_nearest_neighbors.ipynb
│   ├── example_decision_trees.ipynb
│   ├── example_regression_trees.ipynb
│   └── example_ensemble_methods.ipynb
│
└── unsupervised/           # Notebooks using the Mall Customers dataset
    ├── example_k_means_clustering.ipynb
    ├── example_dbscan.ipynb
    ├── example_pca.ipynb
    └── example_community_detection.ipynb
```

---

## Datasets

All data files are in `data/` at the repo root.

**`WineQT.csv`** — Used by all supervised examples. 1,143 red wine samples with 11 physicochemical features (alcohol, acidity, sulphates, etc.) and a quality score from 3 to 8. Depending on the algorithm, quality is used as a continuous regression target, a 3-class classification label (Low/Mid/High), or a binary label (High vs rest).

**`Mall_Customers.csv`** — Used by all unsupervised examples. 200 mall shoppers described by Age, Annual Income (k$), and Spending Score (1–100). No labels — the goal is to discover natural groupings.

---

## Running the Notebooks

```bash
cd examples/supervised
jupyter notebook example_linear_regression.ipynb
```

Or launch Jupyter from the repo root and navigate to the notebook:

```bash
jupyter notebook
```

---

## Notebook Format

Every notebook follows the same structure:

1. **Title and introduction** — what the algorithm does and the goal of this example
2. **Setup and data loading** — imports and a first look at the dataset
3. **Preprocessing** — standardisation, target encoding, train/test split
4. **Training** — fitting the model with chosen hyperparameters
5. **Results and visualisation** — evaluation metrics and plots with explanations of what to look for

See `examples/supervised/README.md` and `examples/unsupervised/README.md` for details on each notebook.
