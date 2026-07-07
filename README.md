# GEMS-Mir
Implementation of GEMS-Mir, a graph-based framework for miRNA–target interaction prediction using thermodynamic ensemble representations and Graph Neural Networks.
<img width="5315" height="3543" alt="summary_figure" src="https://github.com/user-attachments/assets/12838fef-12e6-4b35-a3d1-024895be7e04" />


# Repository Organization

The repository is organized according to the major stages of the GEMS-Mir framework.

| Folder | Description |
|---------|-------------|
| **features/** | Extraction of handcrafted biological features. |
| **duplex/** | Utilities for duplex parsing and thermodynamic analysis. |
| **BreathesDuplex/** | Construction and processing of thermodynamic sub-duplex ensembles. |
| **Prepare_Data/** | Data preprocessing and preparation of the sub-duplexes' positive and negative MTI datasets, including handcrafted featuers extraction. |
| **Representative_Graph/** | Construction of probabilistic weighted nucleotide-level graphs from thermodynamic sub-optimal duplex ensembles (Probability_Matrix.py) and learning of structure-aware graph embeddings using a Graph Convolutional Network (GCNN_embedding_StratifiedKFold.py). |
| **Classifier/** | Hybrid XGBoost classification framework. |
| **mirna_utils/** | General utilities for miRNA processing. |
| **utils/** | General helper functions used throughout the project. |
| **consts/** | Global configuration files and constants. |

---

# Running the GEMS-Mir Pipeline

The repository provides two main execution scripts.

## Step 1 — Generation of Structural Representations

```
Generate_sub_duplexes.py
```

This script performs the complete preprocessing stage starting from the curated MTI datasets.

The pipeline includes:

- generation of thermodynamically plausible sub-duplexes using RNAduplex;
- filtering of biologically valid duplexes;
- construction of probabilistic pairing matrices;
- generation of weighted nucleotide-level graph representations;
- extraction of handcrafted biological descriptors.

The resulting processed datasets are used as input for graph representation learning.

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

# Dataset

The datasets accompanying this work are available from Zenodo.

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
