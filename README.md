# GEMS-Mir
Official implementation of **GEMS-Mir: Graph Learning from Thermodynamic Ensembles of miRNA–mRNA Duplexes for miRNA–Target Interaction Prediction**.
<img width="5315" height="3543" alt="summary_figure" src="https://github.com/user-attachments/assets/12838fef-12e6-4b35-a3d1-024895be7e04" />


# Repository Organization

The repository is organized according to the major stages of the GEMS-Mir framework.

| Folder | Description |
|---------|-------------|
| **features/** | Extraction of handcrafted biological features. |
| **duplex/** | Utilities for duplex parsing and thermodynamic analysis. |
| **BreathesDuplex/** | Construction and processing of thermodynamic sub-duplex ensembles. |
| **Prepare_Data/** | Data preprocessing, preparation of positive and negative MTI datasets, and extraction of handcrafted biological features. |
| **Representative_Graph/** | Construction of probabilistic weighted nucleotide-level graphs from thermodynamic sub-optimal duplex ensembles (Probability_Matrix.py) and learning of structure-aware graph embeddings using a Graph Convolutional Network (GCNN_embedding_StratifiedKFold.py). |
| **Classifier/** | Hybrid XGBoost classification framework. |
| **mirna_utils/** | General utilities for miRNA processing. |
| **utils/** | General helper functions used throughout the project. |
| **consts/** | Global configuration files and constants. |

---

# Running the GEMS-Mir Pipeline

The repository provides two main execution scripts.

## Step 1 — Thermodynamic Ensemble Generation
```
Generate_sub_duplexes.py
```

This script performs the complete preprocessing stage starting from the curated MTI datasets.

The pipeline includes:

- generation of thermodynamically plausible sub-duplexes using RNAduplex;
- construction of an interaction-specific probabilistic pairing matrix that summarizes the ensemble of sub-optimal duplex conformations;
- extraction of handcrafted biological descriptors.

The generated probabilistic matrices are subsequently used to construct weighted nucleotide-level graphs and train the Graph Convolutional Network (GCN) in the second stage of the pipeline.
---

## Step 2 — Graph Representation Learning and Classification

```
main_experiment.py
```

This script performs the machine learning stage of GEMS-Mir.

The workflow includes:

- training the Graph Convolutional Network (GCN);
- extraction of graph embeddings;
- integration of graph embeddings with handcrafted biological features;
- training of the XGBoost classifier;
- evaluation using the predefined 10-fold cross-validation protocol;
- generation of the final prediction scores and evaluation metrics reported in the manuscript.
---
## Optional – Running the Complete Experimental Suite

```
run_all_experiments.py
```

This script automates the execution of multiple experimental configurations reported in the manuscript.
---

# Dataset

The datasets accompanying this work are available from Zenodo: https://doi.org/10.5281/zenodo.21220858.

They include:

- curated positive MTIs;
- curated negative MTIs;
- predefined 10-fold cross-validation splits.

The complete preprocessing pipeline is implemented in this repository.

---

# Citation

If you use this repository, please cite:

Efrat Cohen-Davidi and Isana Veksler-Lublinsky.

**GEMS-Mir: Graph Learning from Thermodynamic Ensembles of miRNA–mRNA Duplexes for miRNA–Target Interaction Prediction.**

---

# License

MIT License.
