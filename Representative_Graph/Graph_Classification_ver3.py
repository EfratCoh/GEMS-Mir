from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import DGLDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import dgl
import torch
import matplotlib.pyplot as plt
import networkx as nx
from deepwalk import graph
from deepwalk import walks as serialized_walks
from torch.nn.functional import softmax
from collections import Counter
import re
import os
from sklearn.utils import shuffle

ROOT_PATH_PHD_GOAL_ONE = Path("/sise/home/efrco/PHD/Goal_One/")
dict_map_label_dataset = {"darnell_human_dataset":0, "NPS_darnell_human_dataset":1}


# Ensure to define ROOT_PATH_PHD_GOAL_ONE and dict_map_label_dataset before using

########################################################################################################
# Class of classifier GCNN
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature.
        # h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = g.ndata['feature']  # Use node features instead of degrees

        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        # h = F.relu(self.conv5(g, h))
        # h = F.relu(self.conv6(g, h))
        g.ndata['h'] = h

        # # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        # Print the shape of hg
        # Pass hg through the final classification layer
        output = self.classify(hg)

        return output, hg

########################################################################################################
# Class that represent Dataset
class InteractionDataset:
    def __init__(self, data_dir, data_dir_sequences, target_size_miRNA, target_size_mRNA):
        self.data_dir = Path(data_dir)
        self.data_dir_sequences = Path(data_dir_sequences)
        self.labels = []
        self.matrices = []
        self.sequence_mirna = []
        self.sequence_mrna = []
        self.target_size_miRNA = target_size_miRNA
        self.target_size_mRNA = target_size_mRNA

        # Process each CSV file in the directory
        for file_path in self.data_dir.glob("*.csv"):
            if "1020" not in str(file_path.stem):
                continue
            print("i am here")
            df = pd.read_csv(file_path)
            matrix = df.iloc[1:, 1:].values.astype(np.float64)
            standardized_matrix = self.padding_matrix(matrix)
            # add matrix
            # add label
            self.matrices.append(standardized_matrix)
            dataset_name = self.data_dir.name
            if dataset_name in dict_map_label_dataset:
                self.labels.append(dict_map_label_dataset[dataset_name])
            else:
                raise ValueError(f"Label for dataset '{dataset_name}' not found in mapping.")

    def padding_matrix(self, matrix):
        matrix = np.array(matrix, dtype=np.float64)
        new_matrix = np.zeros((self.target_size_miRNA, self.target_size_mRNA), dtype=np.float64)
        new_matrix[:min(matrix.shape[0], self.target_size_miRNA),
        :min(matrix.shape[1], self.target_size_mRNA)] = matrix[:min(matrix.shape[0], self.target_size_miRNA),
                                                        :min(matrix.shape[1], self.target_size_mRNA)]
        return new_matrix

    def extract_data_from_csv(self,file_path):
        # Assuming the CSV has columns 'miRNA_sequence' and 'fragment_x'
        df = pd.read_csv(file_path)
        # Extracting values from specific columns
        miRNA_sequence = df['miRNA_sequence'].iloc[1] if 'miRNA_sequence' in df.columns else None
        fragment_x = df['fragment_x'].iloc[1] if 'fragment_x' in df.columns else None
        return miRNA_sequence, fragment_x

    def create_sequence_dict(self):
        # for each file we save the mirna and fragment sequence
        sequence_dict = {}
        for file_path in self.data_dir_sequences.glob("*.csv"):
            # Extract number from filename
            match = re.search(r'\d+', file_path.stem)
            if match:
                file_number = int(match.group())
                miRNA_sequence, fragment_x = self.extract_data_from_csv(file_path)
                sequence_dict[file_number] = {'miRNA_sequence': miRNA_sequence, 'fragment': fragment_x}
        return sequence_dict

    def get_data(self):
        return self.matrices, self.labels, self.create_sequence_dict()


########################################################################################################
# visualisation of graph
def print_graph_info(graphs):
    for i, g in enumerate(graphs):
        num_nodes = g.num_nodes()
        num_edges = g.num_edges()
        print(f"Graph {i + 1}: Nodes = {num_nodes}, Edges = {num_edges}")


def print_edges(g, node_names):
    src, dst = g.edges()  # Get source and destination nodes
    weights = g.edata['weight']  # Get edge weights

    for i in range(g.num_edges()):
        # Print in the desired format
        print(f"{node_names[src[i].item()]} -> {node_names[dst[i].item()]} (weight: {weights[i].item():.3e})")

########################################################################################################
# features to node