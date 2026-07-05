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

ROOT_PATH_PHD_GOAL_ONE = Path("/sise/home/efrco/PHD/Goal_One/")
dict_map_label_dataset = {"darnell_human_dataset":0, "NPS_darnell_human_dataset":1}


# Ensure to define ROOT_PATH_PHD_GOAL_ONE and dict_map_label_dataset before using

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        # self.conv5 = GraphConv(hidden_dim, hidden_dim)
        # self.conv6 = GraphConv(hidden_dim, hidden_dim)
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
            # if "1020" not in str(file_path.stem):
            #     continue
            df = pd.read_csv(file_path)
            matrix = df.iloc[1:, 1:].values.astype(np.float64)
            standardized_matrix = self.padding_matrix(matrix)

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


# Create graph and add features
def nucleotide_encoding_func(nucleotide):
    if nucleotide == 'A':
        return [1, 0, 0, 0, 0]  # A -> [1, 0, 0, 0]
    elif nucleotide == 'U':
        return [0, 1, 0, 0, 0]  # U -> [0, 1, 0, 0]
    elif nucleotide == 'C':
        return [0, 0, 1, 0, 0]  # C -> [0, 0, 1, 0]
    elif nucleotide == 'G':
        return [0, 0, 0, 1, 0]  # G -> [0, 0, 0, 1]
    elif nucleotide == 'N':
        return [0, 0, 0, 0, 1]  # N -> [0, 0, 0, 0] (representing unknown nucleotide)
    else:
        raise ValueError(f"Invalid nucleotide: {nucleotide}")

def add_features_to_graphs(g, sequence_dict, src_mirna_nodes, dst_mrna_nodes, feature_dim=5):
    node_features = []
    node_names = []

    miRNA_sequence = sequence_dict["miRNA_sequence"]
    fragment = sequence_dict["fragment"]


    # Set features for miRNA nodes based on `src` indices
    for miRNA_node_id in range(src_mirna_nodes):
        position = miRNA_node_id
        nucleotide = miRNA_sequence[position] if position < len(miRNA_sequence) else "N"
        next_position = position + 1 if position+1 < len(miRNA_sequence) else -1
        nucleotide_encoding = nucleotide_encoding_func(nucleotide)

        node_feature = [position, next_position] + nucleotide_encoding

        node_features.append(node_feature)
        node_names.append(f"miRNA_{miRNA_node_id}")

    # Set features for mRNA nodes based on `dst` indices
    for mRNA_node_id in range(dst_mrna_nodes):
        mRNA_position = mRNA_node_id + src_mirna_nodes
        nucleotide = fragment[mRNA_node_id] if mRNA_node_id < len(fragment) else "N"
        next_mRNA_position = mRNA_node_id + 1 if mRNA_node_id +1 < len(fragment) else -1
        mRNA_nucleotide_encoding = nucleotide_encoding_func(nucleotide)

        mRNA_node_feature = [mRNA_position, next_mRNA_position+src_mirna_nodes] + mRNA_nucleotide_encoding

        node_features.append(mRNA_node_feature)
        node_names.append(f"mRNA_{mRNA_node_id}")

        # Compute in-degrees for all nodes and add as a feature
    in_degrees = g.in_degrees().tolist()  # Convert to a list for easier access
    for i in range(len(node_features)):
        node_features[i].append(in_degrees[i])  # Append in-degree as an additional feature      # Compute in-degrees for all nodes and add as a feature

    out_degrees = g.out_degrees().tolist()  # Convert to a list for easier access
    for i in range(len(node_features)):
        node_features[i].append(out_degrees[i])  # Append in-degree as an additional feature


    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    g.ndata['feature'] = node_features_tensor

    # # Print all nodes with their features and names
    # for i in range(g.num_nodes()):
    #     print(f"Node {i} - Name: {node_names[i]}, Features: {g.ndata['feature'][i].tolist()}")

    return g



# Example function to create the graph and add features
def create_graph_with_features(matrices, sequence_dict, weight_scale_factor=1e12, feature_dim=5):
    graphs = []
    num_graph = 0
    for matrix in matrices:
        matrix = torch.tensor(matrix, dtype=torch.float32)

        # Get miRNA (row) and mRNA (column) indices for non-zero entries
        src, dst = matrix.nonzero(as_tuple=True)

        # Ensure the number of nodes includes all miRNA and mRNA nodes
        num_mirna_nodes = matrix.shape[0]  # miRNA nodes (rows)
        num_mrna_nodes = matrix.shape[1]  # mRNA nodes (columns)

        # Offset mRNA nodes to avoid overlap with miRNA node indices
        dst += num_mirna_nodes

        # Create the graph with miRNA -> mRNA edges
        weights = matrix[src, dst - num_mirna_nodes]
        g = dgl.graph((src, dst), num_nodes=num_mirna_nodes + num_mrna_nodes)
        g.edata['weight'] = weights * weight_scale_factor
        g = dgl.add_self_loop(g, fill_data=0.001)

        # Add node features for each node
        g = add_features_to_graphs(g, sequence_dict[num_graph], num_mirna_nodes, num_mrna_nodes, feature_dim)
        num_graph +=1

        graphs.append(g)

    return graphs


# Wrap graphs and labels in a custom DGL Dataset if needed
class GraphDataset(DGLDataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        super().__init__(name='interaction_graphs')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

# Custom collate function to batch graphs and labels
def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))  # Unpack graphs and labels
    batched_graph = dgl.batch(graphs)  # Batch the list of graphs
    labels = torch.tensor(labels)  # Convert labels to tensor if needed
    return batched_graph, labels


def graphClassification(in_dim, hidden_dim,n_classes, num_epoch, learning_rate):
    positive_dataset_name = "darnell_human_dataset"
    negative_dataset_name = "NPS_darnell_human_dataset"
    positive_data_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / positive_dataset_name
    negative_data_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / negative_dataset_name

    positive_data_dir_duplex = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / positive_dataset_name / "duplex_step"
    negative_data_dir_duplex = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / negative_dataset_name / "duplex_step"

    positive_dataset = InteractionDataset(positive_data_dir,positive_data_dir_duplex, 25, 75)
    negative_dataset = InteractionDataset(negative_data_dir,negative_data_dir_duplex, 25, 75)

    positive_matrices, positive_labels, pos_sequence_dict = positive_dataset.get_data()
    negative_matrices, negative_labels, neg_sequence_dict = negative_dataset.get_data()


    # Create graphs from matrices
    graphs_pos = create_graph_with_features(positive_matrices, pos_sequence_dict, feature_dim=in_dim)
    graphs_neg = create_graph_with_features(negative_matrices, neg_sequence_dict, feature_dim=in_dim)
    print_graph_info(graphs_pos)
    print("i am here")

    # Combine graphs and labels for classification
    all_graphs = graphs_pos + graphs_neg
    all_labels = positive_labels + negative_labels

    # Convert labels to a tensor
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    # Split dataset into training and testing sets
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(all_graphs, all_labels, test_size=0.2, random_state=42)

    # Count the labels in training and testing sets
    train_label_counts = Counter(train_labels.numpy())
    test_label_counts = Counter(test_labels.numpy())

    print("Training set:")
    print(f"  Positive labels: {train_label_counts[1]}")
    print(f"  Negative labels: {train_label_counts[0]}")

    print("Testing set:")
    print(f"  Positive labels: {test_label_counts[1]}")
    print(f"  Negative labels: {test_label_counts[0]}")

    train_dataset = GraphDataset(train_graphs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_dataset = GraphDataset(test_graphs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


    model = Classifier(in_dim, hidden_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Training loop
    for epoch in range(num_epoch):  # Number of epochs
        for graphs, labels in train_dataloader:
            logits, _ = model(graphs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {loss.item()}")

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for graphs, labels in test_dataloader:
            logits, _ = model(graphs)

            # Apply softmax to obtain probabilities
            probabilities = softmax(logits, dim=1)

            # Get the predicted class with the highest probability
            _, predicted = torch.max(probabilities, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')
        print("Training and evaluation completed.")


def fine_tuning_hyper_parameter():
    in_dim = 9
    # Define model parameters
    hidden_dim = 64  # Hidden layer size
    n_classes = 2  # Number of output classes
    num_epoch = 60
    learning_rate = 0.001

    graphClassification(in_dim, hidden_dim, n_classes, num_epoch, learning_rate)


fine_tuning_hyper_parameter()