import numpy as np
import pandas as pd
from pathlib import Path
import torch
import dgl
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from dgl.dataloading import GraphDataLoader  # Use DGL DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
# from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
from dgl.nn.pytorch import GraphConv
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dgl.data import MiniGCDataset


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
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        g.ndata['h'] = h

        # # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        # Print the shape of hg
        # Pass hg through the final classification layer
        output = self.classify(hg)

        # Print the shape of output
        print(f"Shape of output (after classification): {output.shape}")
        # print(output)

        return output, hg

class InteractionDataset:
    def __init__(self, data_dir, target_size_miRNA, target_size_mRNA):
        self.data_dir = Path(data_dir)
        self.labels = []
        self.matrices = []
        self.target_size_miRNA = target_size_miRNA
        self.target_size_mRNA = target_size_mRNA

        # Process each CSV file in the directory
        for file_path in self.data_dir.glob("*.csv"):
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

    def get_data(self):
        return self.matrices, self.labels


def matrix_to_graph(matrix, row_label="miRNA", col_label="fragment"):
    G = nx.DiGraph()
    num_rows, num_cols = matrix.shape

    # Create row nodes (e.g., miRNA) and add them to the graph
    row_nodes = [f"{i}_{row_label}" for i in range(num_rows)]
    for row_node in row_nodes:
        G.add_node(row_node,feat=torch.tensor([1, 0], dtype=torch.float))

    # Create column nodes (e.g., fragments) and add them to the graph
    col_nodes = [f"{j}_{col_label}" for j in range(num_cols)]
    for col_node in col_nodes:
        G.add_node(col_node, feat=torch.tensor([0, 1], dtype=torch.float))

    # Add edges for non-zero matrix entries
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i, j] != 0:
                G.add_edge(f"{i}_{row_label}", f"{j}_{col_label}", weight=matrix[i, j])


    # Convert the NetworkX graph to a DGL graph
    dgl_graph = dgl.from_networkx(G,  node_attrs=['feat'], edge_attrs=['weight'])

    # Add node features to the DGL graph
    # Here, we simply add identity features for each node (change as needed)
    # dgl_graph.ndata['feat'] = torch.eye(dgl_graph.number_of_nodes())  # Example node features
    dgl_graph.set_n_initializer(dgl.init.zero_initializer)

    return dgl_graph

#
# class GraphDataset():
#     def __init__(self, graphs, labels):
#         assert len(graphs) == len(labels), "The number of graphs and labels must match."
#         self.graphs = graphs
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.graphs)
#
#     def __getitem__(self, idx):
#         graph, label = self.graphs[idx], self.labels[idx]
#         # Ensure the label is a tensor
#         label = torch.tensor(label, dtype=torch.long)
#         return graph, label
#


# Define the collate function
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.long)


batch_size = 3

def graphClassification():
    positive_dataset_name = "darnell_human_dataset"
    negative_dataset_name = "NPS_darnell_human_dataset"

    # Load positive dataset
    positive_data_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / positive_dataset_name
    positive_dataset = InteractionDataset(positive_data_dir, 32, 75)
    positive_matrices, positive_labels = positive_dataset.get_data()
    positive_graphs = [matrix_to_graph(matrix) for matrix in positive_matrices]

    # Load negative dataset
    negative_data_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / negative_dataset_name
    negative_dataset = InteractionDataset(negative_data_dir, 32, 75)
    negative_matrices, negative_labels = negative_dataset.get_data()
    negative_graphs = [matrix_to_graph(matrix) for matrix in negative_matrices]

    # Combine positive and negative data
    graphs = positive_graphs + negative_graphs
    labels = positive_labels + negative_labels

    # Create a balanced train-test split
    num_positive = len(positive_graphs)
    num_negative = len(negative_graphs)
    min_count = min(num_positive, num_negative)

    # Use only min_count samples from each dataset for balanced splitting
    balanced_graphs = graphs[:min_count * 2]  # Take equal samples from both
    balanced_labels = labels[:min_count * 2]

    # Shuffle the combined dataset
    combined = list(zip(balanced_graphs, balanced_labels))
    np.random.shuffle(combined)
    balanced_graphs[:], balanced_labels[:] = zip(*combined)

    # Split into train and test (80-20 split)
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        balanced_graphs,
        balanced_labels,
        test_size=0.2,
        random_state=42
    )



    # Create the data loaders
    train_loader = DataLoader(list(zip(train_graphs, train_labels)), batch_size=batch_size, shuffle=False,
                              collate_fn=collate)
    test_loader = DataLoader(list(zip(test_graphs, test_labels)), batch_size=batch_size, shuffle=False,
                             collate_fn=collate)

    in_feats = 2  # Number of input features per node
    hidden_size = 64
    num_classes = 2
    num_epochs = 30
    learning_rate = 0.006

    model = Classifier(in_feats, hidden_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_func = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        print(model)
        for iter, (bg, label) in enumerate(train_loader):
            print("$$$")
            # Print batch details
            print("Number of nodes in the batch:", bg.number_of_nodes())
            print("Number of edges in the batch:", bg.number_of_edges())

            # Print node features for each graph in the batch
            print("Node features in the batch (bg.ndata['feat']):")
            # print(bg.ndata['feat'])  # Ensure 'feat' is the key for your node features

            # Print labels
            print("Labels in this batch:", label)
            print("=" * 50)  # Separator for clarity
            print("$$$$")
            bg = dgl.add_self_loop(bg)  # Add self-loops here
            prediction, _ = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    # Evaluation on the test set
    if test_loader:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batched_graph, batched_labels in test_loader:
                batched_graph = dgl.add_self_loop(batched_graph)
                batched_graph.ndata['feat'] = torch.eye(batched_graph.number_of_nodes())
                outputs, _ = model(batched_graph)
                _, predicted = torch.max(outputs, dim=1)  # Use dim=1 to get predictions for each sample
                total += batched_labels.size(0)
                correct += (predicted == batched_labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')
    else:
        print("Not enough samples for testing.")


# Call the function to run graph classification
graphClassification()

