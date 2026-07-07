# GEMS-Mir
Official implementation of **GEMS-Mir: Graph Learning from Thermodynamic Ensembles of miRNA–mRNA Duplexes for miRNA–Target Interaction Prediction**.
<img width="5315" height="3543" alt="summary_figure" src="https://github.com/user-attachments/assets/12838fef-12e6-4b35-a3d1-024895be7e04" />


## Overview

GEMS-Mir is a hybrid machine learning framework for predicting miRNA–target interactions (MTIs). Unlike conventional approaches that rely on a single minimum free-energy (MFE) duplex, GEMS-Mir explicitly models the thermodynamic ensemble of plausible miRNA–mRNA binding conformations.

Each interaction is represented as an ensemble of thermodynamically plausible sub-optimal duplexes. These structures are integrated into a probabilistic nucleotide-level representation, from which weighted graphs are constructed. A Graph Convolutional Network (GCN) learns structure-aware graph embeddings that are subsequently combined with handcrafted biological features within an XGBoost classifier for final MTI prediction.

The repository contains the complete implementation of the computational pipeline described in the accompanying manuscript.

---

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

# Installation

Clone the repository:

```bash
git clone https://github.com/EfratCoh/GEMS-Mir.git
cd GEMS-Mir
```

Create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

# Dataset

The datasets accompanying this work are publicly available through Zenodo: https://doi.org/10.5281/zenodo.21220858.

They include:

- Curated positive MTIs
- Curated negative MTIs
- Predefined 10-fold cross-validation splits

After downloading the archive, place the dataset under the project data directory.

---

# Experiments Reproduction

The complete experimental pipeline consists of two consecutive stages.

## Step 1 — Thermodynamic Ensemble Generation

Run:

```bash
python generate_structural_representation.py
```

This stage performs:

- generation of thermodynamically plausible sub-duplexes using RNAduplex;
- construction of an interaction-specific probabilistic pairing matrix by integrating the complete ensemble of sub-optimal duplex conformations;
- extraction of handcrafted biological features.

The output of this stage is a structural representation for every miRNA–target interaction, consisting of:

- thermodynamic sub-duplex ensembles;
- probabilistic pairing matrices;
- handcrafted biological features.

These outputs serve as the input for the graph construction and graph representation learning stage.

---

## Step 2 — Run the GEMS-Mir Experiments

```bash
python run_all_experiments.py
```

This script automatically executes the complete experimental workflow described in the manuscript, including graph representation learning, GCN training, graph embedding extraction, hybrid classification, and performance evaluation.
---

# Citation

If you use this repository, please cite:

> Efrat Cohen-Davidi and Isana Veksler-Lublinsky.
>
> **GEMS-Mir: Graph Learning from Thermodynamic Ensembles of miRNA–mRNA Duplexes for miRNA–Target Interaction Prediction.**

---

# License

This project is distributed under the MIT License.
