import pandas as pd
import numpy as np
from pathlib import Path
import torch
import dgl
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
# from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
from dgl.nn.pytorch import GraphConv
import matplotlib.pyplot as plt
import networkx as nx


ROOT_PATH_PHD_GOAL_ONE = Path("/sise/home/efrco/PHD/Goal_One/")

dict_map_label_dataset = {"darnell_human_dataset":0, "NPS_darnell_human_dataset":1}

# Load the data - matrix and label

class InteractionDataset:
    def __init__(self, data_dir, target_size_miRNA, target_size_mRNA):
        self.data_dir = Path(data_dir)
        self.labels = []
        self.matrices = []
        self.target_size_miRNA = target_size_miRNA  # Target size for standardization
        self.target_size_mRNA = target_size_mRNA  # Target size for standardization

        # Process each CSV file in the directory
        for file_path in self.data_dir.glob("*.csv"):
            if "1020" not in str(file_path.stem):
                continue
            print(file_path)
            # Load the matrix from CSV, skipping the first row and column
            df = pd.read_csv(file_path)
            matrix = df.iloc[1:, 1:].values.astype(np.float64)  # Convert to float
            standardized_matrix = self.standardize_matrix(matrix)

            # Append the matrix to the list
            self.matrices.append(standardized_matrix)

            # Map the dataset directory name to its label
            dataset_name = self.data_dir.name
            if dataset_name in dict_map_label_dataset:
                self.labels.append(dict_map_label_dataset[dataset_name])
            else:
                raise ValueError(f"Label for dataset '{dataset_name}' not found in mapping.")


    def standardize_matrix(self, matrix):
        # Ensure the matrix is a float array
        matrix = np.array(matrix, dtype=np.float64)

        # Get the current shape
        num_rows, num_cols = matrix.shape

        # Create a new matrix with the target sizes
        new_matrix = np.zeros((self.target_size_miRNA, self.target_size_mRNA), dtype=np.float64)

        # Copy the values from the original matrix to the new matrix
        new_matrix[:min(num_rows, self.target_size_miRNA), :min(num_cols, self.target_size_mRNA)] = matrix[:min(num_rows, self.target_size_miRNA), :min(num_cols, self.target_size_mRNA)]

        return new_matrix

    def get_data(self):
        return self.matrices, self.labels


def matrix_to_graph(matrix):
    matrix = np.array(matrix, dtype=np.float64)  # Ensure the matrix is of type float
    edge_index = np.transpose(np.nonzero(matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(matrix[matrix != 0], dtype=torch.float)
    x = torch.eye(matrix.shape[0], dtype=torch.float)
    G = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(G)
    return G


def plot_graph(matrix, fragment_label="fragment", mirna_label="miRNA"):
    # Ensure the matrix is of type float64
    matrix = np.array(matrix, dtype=np.float64)
    # Create graph using NetworkX
    G = nx.Graph()

    num_mirna = matrix.shape[0]  # Number of rows (miRNA)
    num_fragment = matrix.shape[1]  # Number of columns (mRNA fragment)

    # Add edges only between miRNA and fragment (mRNA) nodes where the matrix has non-zero values
    for i in range(num_mirna):  # Loop over miRNA (rows)
        for j in range(num_fragment):  # Loop over fragments (columns)

            if matrix[i, j] != 0:
                # Create edge between miRNA and fragment nodes
                G.add_edge(f"{i}_{mirna_label}", f"{j}_{fragment_label}", weight=matrix[i, j])
                print(f"{i}_{mirna_label}", f"{j}_{fragment_label}", matrix[i, j])

    pos = nx.spring_layout(G, k=0.7, iterations=50)  # Adjust k for more spacing

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', node_size=900)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Show the plot
    plt.show()
    print(G)

class GraphDataset:
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return graph

class GCNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNN, self).__init__()
        # self.conv1 = GCNConv(num_features, 16)
        # self.conv2 = GCNConv(16, 32)
        # self.fc1 = torch.nn.Linear(32, num_classes)
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    # def forward(self, data):
    #     x, edge_index, batch = data.x, data.edge_index, data.batch
    #     x = self.conv1(x, edge_index)
    #     x = F.relu(x)
    #     x = self.conv2(x, edge_index)
    #     x = F.relu(x)
    #     x = global_mean_pool(x, batch)
    #     x = self.fc1(x)
    #     return F.log_softmax(x, dim=1)

    def forward(self, g, embedding):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees()
        h = g.in_degrees().view(-1, 1).float()
        h.data = embedding
        # h = embedding.tousertensor()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        # h = F.relu(self.conv5(g, h))
        # h = F.relu(self.conv6(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg), hg




# Example usage
dataset_name = "darnell_human_dataset"
data_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix"/ dataset_name
dataset = InteractionDataset(data_dir, 32, 75)
matrices, labels = dataset.get_data()
for i, matrix in enumerate(matrices):
    print(f"Matrix {i} shape: {matrix.shape}")
# Convert matrices to graphs
graphs = [matrix_to_graph(matrix) for matrix in matrices]
graphs = [plot_graph(matrix) for matrix in matrices]


# # Encode string labels to numerical labels
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)
#
# # Create dataset
# graph_dataset = GraphDataset(graphs, encoded_labels)
#
# # Prepare data loader
# batch_size = 32
# loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
#
# # Initialize model
# num_features = matrices[0].shape[0]  # Assuming square matrices
# num_classes = len(set(encoded_labels))  # Number of unique labels
# model = GCNN(num_features, num_classes)
#
# # Optimizer and loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# loss_fn = torch.nn.CrossEntropyLoss()
#
# # Training loop
# model.train()
# for epoch in range(100):
#     for data in loader:
#         data = data.to('cpu')
#         optimizer.zero_grad()
#         out = model(data)
#         loss = loss_fn(out, data.y)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#
# # Evaluation
# model.eval()
# correct = 0
# for data in loader:
#     data = data.to('cpu')
#     with torch.no_grad():
#         out = model(data)
#         pred = out.argmax(dim=1)
#         correct += (pred == data.y).sum().item()
# accuracy = correct / len(graphs)
# print(f'Accuracy: {accuracy:.4f}')
# with torch.no_grad():
#     node_embeddings = model(data)
#     print(node_embeddings)


###############Good data appended########################

#
# def plot_graph(g):
#     # Convert DGL graph to NetworkX graph
#     nx_graph = g.to_networkx()
#
#     # Draw the graph
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(nx_graph)  # positions for all nodes
#
#     # Draw nodes and edges
#     nx.draw_networkx_nodes(nx_graph, pos, node_size=700)
#     nx.draw_networkx_edges(nx_graph, pos, width=1.0, alpha=0.5)
#
#     # Draw labels with node names
#     labels = {i: g.graph_name[i] for i in range(g.num_nodes())}  # Use names from g.graph_name
#     nx.draw_networkx_labels(nx_graph, pos, labels, font_size=8)
#
#     # Display the graph
#     plt.title("Graph Visualization")
#     plt.axis('off')  # Turn off the axis
#     plt.show()

# def print_edges(g):
#     # Get the source and destination nodes for all edges
#     src, dst = g.edges()
#
#     # Print each edge with node names
#     print("Edges in the graph:")
#     print("#############################################################3")
#     print(g.graph_name)
#     for s, d in zip(src.tolist(), dst.tolist()):
#         source_name = f'mirna_{s}'
#         destination_name = f'mrna_{d}'
#         print(f"({source_name}, {destination_name})")
#
#
# def create_graphs(matrices):
#     graphs = []
#     for matrix in matrices:
#         # Convert the numpy matrix to a PyTorch tensor
#         matrix = torch.tensor(matrix, dtype=torch.float32)
#
#         # Get the source and destination indices for non-zero entries
#         src, dst = matrix.nonzero(as_tuple=True)
#
#         # Determine the number of nodes based on the highest node ID
#         num_nodes = max(src.max().item(), dst.max().item()) + 1
#
#         # Create a graph using the source and destination node indices
#         g = dgl.graph((src, dst), num_nodes=num_nodes)
#
#         # Add self-loops to the graph
#         g = dgl.add_self_loop(g)
#
#         # Get the updated source and destination indices after adding self-loops
#         src, dst = g.edges()
#         print(src, dst)
#
#         # Ensure weights are only extracted for valid indices
#         valid_indices = (src < matrix.size(0)) & (dst < matrix.size(1))
#         weights = matrix[src[valid_indices], dst[valid_indices]]
#
#         # If weights' size doesn't match the number of edges, assign default weights
#         if weights.size(0) != g.num_edges():
#             weights = torch.ones(g.num_edges())
#
#         # Set the edge weights
#         g.edata['weight'] = weights
#
#         # Add node features: here, we use the degree of each node as a feature
#         node_degrees = g.in_degrees().view(-1, 1).float()  # Convert to float tensor
#         g.ndata['degree'] = node_degrees  # Store as node feature with key 'degree'
#
#         # Assign names to the nodes based on their indices
#         node_names = []
#         for node_id in range(num_nodes):
#             if node_id < matrix.shape[0]:  # Assuming the number of rows is for miRNA
#                 node_names.append(f"mirna_{node_id}")
#             else:
#                 mRNA_index = node_id - matrix.shape[0]  # Offset for mRNA names
#                 node_names.append(f"mRNA_{mRNA_index}")
#
#                 # Store node names as a separate attribute (not in ndata)
#             g.graph_name = node_names  # You can access this as g.graph_name
#             # Print all edges in the graph
#         print_edges(g)
#
#         plot_graph(g)
#
#         graphs.append(g)
#     return graphs
