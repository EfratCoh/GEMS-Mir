import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import os
from pathlib import Path
from tqdm import tqdm
from consts.global_consts import dict_map_label_dataset, FOLDS_PATH, MERGE_DATA, Data_Embedding, number_epoch, ROOT_PATH_PHD_GOAL_ONE, NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS, MATRIX_DATA
from consts.GNN_consts import *
import re
from dgl.data import DGLDataset
from utils.utilsfile import read_csv, to_csv, clean_directory, get_wrapper
import matplotlib.pyplot as plt

# Updated GCNN model
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feature']
        #h = g.in_degrees().view(-1, 1).float()
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        output = self.classify(hg)
        return output, hg

# Helper function to load matrix
def load_graph_matrix(id_interaction):
    dataset_name = '_'.join(id_interaction.split('_')[:-1])
    file_num = id_interaction.split('_')[-1]
    matrix_path = MATRIX_DATA / dataset_name / f"{file_num}.csv"
    return pd.read_csv(matrix_path).values


# represent a one interaction
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

    def create_sequence_dict(self, indices):
        sequence_dict = {}
        for file_path in self.data_dir_sequences.glob("*.csv"):
            # Extract number from filename
            match = re.search(r'\d+', file_path.stem)
            if match:
                file_number = int(match.group())
                if file_number in indices:
                    miRNA_sequence, fragment_x = self.extract_data_from_csv(file_path)
                    sequence_dict[file_number] = {
                        'miRNA_sequence': miRNA_sequence,
                        'fragment': fragment_x
                    }
        return sequence_dict


    def get_data(self, indices):
        matrices, labels, original_filenames = [], [], []
        files = sorted(self.data_dir.glob("*.csv"))
        # Create a map from file stem (number string) to Path object
        file_map = {f.stem: f for f in files}

        for fname in indices:
            fname_str = str(fname)  # ensure string
            if fname_str not in file_map:
               print(f"!!!!!!!!!!!!!!!!!!!!!!!!! File with name '{fname_str}' not found in {self.data_dir}")
               continue

            file_path = file_map[fname_str]
            df = pd.read_csv(file_path)
            matrix = df.iloc[1:, 1:].values.astype(np.float64)
            standardized_matrix = self.padding_matrix(matrix)
            matrices.append(standardized_matrix)

            dataset_name = self.data_dir.name
            if dataset_name in dict_map_label_dataset:
                labels.append(dict_map_label_dataset[dataset_name])
            else:
                raise ValueError(f"Label for dataset '{dataset_name}' not found in mapping.")
            fname = str(dataset_name).replace("_dataset", "") + '_' + fname_str
            original_filenames.append(fname)

        return matrices, labels, self.create_sequence_dict(indices), original_filenames


# Wrap graphs and labels in a custom DGL Dataset if needed
# Dataset class
class GraphDataset(DGLDataset):
    def __init__(self, graphs, labels, filenames):
        self.graphs = graphs
        self.labels = labels
        self.filenames = filenames
        super().__init__(name='interaction_graphs')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.filenames[idx]

# Custom collate function to batch graphs and labels
def collate_fn(batch):
    graphs, labels, filenames = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), filenames

def get_number_by_ID(ID_interaction):
    match = re.search(r'\d+', ID_interaction)
    file_number = int(match.group())
    return int(file_number)

# Example function to create the graph and add features
def create_graph_with_features(matrices, sequence_dict, weight_scale_factor=1e12):
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

        # Convert to undirected graph
        g = dgl.to_bidirected(g, copy_ndata=True)

        # Add self-loops with a small weight (optional)
        g = dgl.add_self_loop(g, fill_data=0.1)

        # Add node features for each node
        # g = add_features_to_graphs_random(g, sequence_dict[num_graph], num_mirna_nodes, num_mrna_nodes, feature_dim)
        num_graph += 1
        features = torch.randn(num_mirna_nodes+num_mrna_nodes,dim_vector_in)  # or actual 64-dim features
        g.ndata['feature'] = features
        graphs.append(g)

    return graphs

def generate_embedding(save_embedding_dir):
    # Main cross-validation evaluation loop
    positive_name = "darnell_human_dataset"
    negative_name = "NPS_darnell_human_dataset"
    pos_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / positive_name
    neg_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / negative_name
    pos_seq = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / positive_name / "duplex_step"
    neg_seq = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / negative_name / "duplex_step"

    all_pos_files = sorted(pos_dir.glob("*.csv"))
    all_neg_files = sorted(neg_dir.glob("*.csv"))

    for fold_dir in sorted(FOLDS_PATH.glob("fold*")):
        print(f"Processing {fold_dir.name}")
        train_df = read_csv(fold_dir / f"{fold_dir.name}_train.csv")
        test_df = read_csv(fold_dir / f"{fold_dir.name}_test.csv")
        train_df['number_interaction'] = train_df.apply(func=get_wrapper(get_number_by_ID,"ID_interaction"), axis=1)
        test_df['number_interaction'] = test_df.apply(func=get_wrapper(get_number_by_ID,"ID_interaction"), axis=1)

        train_pos_idx = train_df[train_df['label'] == 1]['number_interaction'].tolist()
        train_neg_idx = train_df[train_df['label'] == 0]['number_interaction'].tolist()
        test_pos_idx = test_df[test_df['label'] == 1]['number_interaction'].tolist()
        test_neg_idx = test_df[test_df['label'] == 0]['number_interaction'].tolist()

        pos_dataset = InteractionDataset(pos_dir, pos_seq, 25, 75)
        neg_dataset = InteractionDataset(neg_dir, neg_seq, 25, 75)

        train_pos_matrices, train_pos_labels, pos_seq_dict, train_pos_names = pos_dataset.get_data(train_pos_idx)
        train_neg_matrices, train_neg_labels, neg_seq_dict, train_neg_names = neg_dataset.get_data(train_neg_idx)
        test_pos_matrices, test_pos_labels, _, test_pos_names = pos_dataset.get_data(test_pos_idx)
        test_neg_matrices, test_neg_labels, _, test_neg_names = neg_dataset.get_data(test_neg_idx)

        train_pos_graphs = create_graph_with_features(train_pos_matrices ,pos_seq_dict)
        train_neg_graphs = create_graph_with_features(train_neg_matrices, neg_seq_dict)

        test_pos_graphs = create_graph_with_features(test_pos_matrices, pos_seq_dict)
        test_neg_graphs = create_graph_with_features(test_neg_matrices, neg_seq_dict)

        # Concatenate for final training/testing sets
        train_graphs = train_pos_graphs + train_neg_graphs
        test_graphs = test_pos_graphs + test_neg_graphs
        train_labels = train_pos_labels + train_neg_labels
        test_labels = test_pos_labels + test_neg_labels
        train_filenames = train_pos_names + train_neg_names
        test_filenames = test_pos_names + test_neg_names

        train_dataset = GraphDataset(train_graphs, train_labels, train_filenames)
        test_dataset = GraphDataset(test_graphs, test_labels,test_filenames)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle =True, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        device = torch.device("cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Classifier(in_dim=dim_vector_in, hidden_dim=dim_vector_out, n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # loss_fn = nn.BCEWithLogitsLoss() # function mutch to n_classes = 1
        loss_fn = nn.CrossEntropyLoss()

        # save_embedding_dir = Data_Embedding

        fold_embedding_dir = save_embedding_dir / fold_dir.name
        os.makedirs(fold_embedding_dir, exist_ok=True)
        clean_directory(fold_embedding_dir)

        train_losses = []
        test_losses = []
        test_accuracies = []

        for epoch in range(number_epoch):
            running_loss = 0.0
            model.train()

            for batched_graph, labels, _ in train_loader:
                batched_graph = batched_graph.to(device)
                # labels = labels.to(device).float() # match to n_classes=1
                labels = labels.to(device).long()
                optimizer.zero_grad()
                output, _ = model(batched_graph)
                # loss = loss_fn(output.squeeze(), labels) # match to n_classes=1
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()

            def extract_embeddings_and_loss_accuracy(loader):
                embeddings, labels_all, names = [], [], []
                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                with torch.no_grad():
                    for batched_graph, label, filename_batch in loader:
                        batched_graph = batched_graph.to(device)
                        label = label.to(device).long()  # 🔁 multiclass → long

                        output, emb = model(batched_graph)  # output.shape = [batch_size, n_classes]
                        loss = loss_fn(output, label)  # 🔁 no squeeze

                        preds = torch.argmax(output, dim=1)  # 🔁 get predicted class

                        total_loss += loss.item()
                        total_correct += (preds == label).sum().item()
                        total_samples += len(label)

                        embeddings.append(emb.cpu().numpy())
                        labels_all.extend(label.cpu().numpy())
                        names.extend(filename_batch)

                avg_loss = total_loss / len(loader)
                accuracy = total_correct / total_samples
                return np.concatenate(embeddings), labels_all, names, avg_loss, accuracy

            train_embeddings, train_labels_epoch, train_names_epoch, _, _ = extract_embeddings_and_loss_accuracy(
                train_loader)
            test_embeddings, test_labels_epoch, test_names_epoch, test_loss, test_acc = extract_embeddings_and_loss_accuracy(
                test_loader)

            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print(
                f"Epoch {epoch + 1}/{number_epoch} - Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

            # === Save to CSV format per epoch === #
            def save_embeddings_to_csv(embeddings, names, save_dir, set_type):
                epoch_dir = os.path.join(fold_embedding_dir, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)

                df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
                df.insert(0, "ID_interaction", names)
                df.to_csv(os.path.join(epoch_dir, f"{set_type}_embeddings.csv"), index=False)


            save_embeddings_to_csv(train_embeddings, train_names_epoch, fold_embedding_dir, "train")
            save_embeddings_to_csv(test_embeddings, test_names_epoch, fold_embedding_dir, "test")

        # === גרף === #
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, number_epoch + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, number_epoch + 1), test_losses, label="Test Loss", marker='o')
        plt.plot(range(1, number_epoch + 1), test_accuracies, label="Test Accuracy", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_embedding_dir, "training_plot.png"))
        plt.show()


print("[INFO] Running embedding generation...")
generate_embedding(save_embedding_dir="")