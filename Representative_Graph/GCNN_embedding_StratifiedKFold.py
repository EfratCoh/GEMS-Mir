# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# from dgl.nn.pytorch import GraphConv
# import os
# from pathlib import Path
# from consts.global_consts import dict_map_label_dataset, FOLDS_PATH, MERGE_DATA, Data_Embedding, ROOT_PATH_PHD_GOAL_ONE, NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS, MATRIX_DATA
#
# import re
# from dgl.data import DGLDataset
# from utils.utilsfile import read_csv, to_csv, clean_directory, get_wrapper, import_consts
# import matplotlib.pyplot as plt
# from Representative_Graph.Features_Graph import create_graph_with_features
#
#
#
# # Updated GCNN model
# class Classifier(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes):
#         super(Classifier, self).__init__()
#         self.conv1 = GraphConv(in_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, hidden_dim)
#         self.conv3 = GraphConv(hidden_dim, hidden_dim)
#         self.conv4 = GraphConv(hidden_dim, hidden_dim)
#         self.classify = nn.Linear(hidden_dim, n_classes)
#
#
#     def forward(self, g):
#         h = g.ndata['feature']
#         h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
#         h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
#         h = F.relu(self.conv3(g, h, edge_weight=g.edata['weight']))
#         h = F.relu(self.conv4(g, h, edge_weight=g.edata['weight']))
#         g.ndata['h'] = h
#         hg = dgl.mean_nodes(g, 'h')
#         output = self.classify(hg)
#         return output, hg
#
# # Helper function to load matrix
# def load_graph_matrix(id_interaction):
#     dataset_name = '_'.join(id_interaction.split('_')[:-1])
#     file_num = id_interaction.split('_')[-1]
#     matrix_path = MATRIX_DATA / dataset_name / f"{file_num}.csv"
#     return pd.read_csv(matrix_path).values
#
#
# # represent a one interaction
# class InteractionDataset:
#     def __init__(self, data_dir, data_dir_sequences, target_size_miRNA, target_size_mRNA):
#         self.data_dir = Path(data_dir)
#         self.data_dir_sequences = Path(data_dir_sequences)
#         self.labels = []
#         self.matrices = []
#         self.sequence_mirna = []
#         self.sequence_mrna = []
#         self.target_size_miRNA = target_size_miRNA
#         self.target_size_mRNA = target_size_mRNA
#
#     def padding_matrix(self, matrix):
#         matrix = np.array(matrix, dtype=np.float64)
#         new_matrix = np.zeros((self.target_size_miRNA, self.target_size_mRNA), dtype=np.float64)
#         new_matrix[:min(matrix.shape[0], self.target_size_miRNA),
#         :min(matrix.shape[1], self.target_size_mRNA)] = matrix[:min(matrix.shape[0], self.target_size_miRNA),
#                                                         :min(matrix.shape[1], self.target_size_mRNA)]
#         return new_matrix
#
#     def extract_data_from_csv(self,file_path):
#         # Assuming the CSV has columns 'miRNA_sequence' and 'fragment_x'
#         df = pd.read_csv(file_path)
#         # Extracting values from specific columns
#         miRNA_sequence = df['miRNA_sequence'].iloc[1] if 'miRNA_sequence' in df.columns else None
#         fragment_x = df['fragment_x'].iloc[1] if 'fragment_x' in df.columns else None
#         return miRNA_sequence, fragment_x
#
#     def create_sequence_dict(self, indices):
#         sequence_dict = {}
#         for file_path in self.data_dir_sequences.glob("*.csv"):
#             # Extract number from filename
#             match = re.search(r'\d+', file_path.stem)
#             if match:
#                 file_number = int(match.group())
#                 if file_number in indices:
#                     miRNA_sequence, fragment_x = self.extract_data_from_csv(file_path)
#                     sequence_dict[file_number] = {
#                         'miRNA_sequence': miRNA_sequence,
#                         'fragment': fragment_x
#                     }
#         return sequence_dict
#
#
#     def get_data(self, indices):
#         matrices, labels, original_filenames = [], [], []
#         files = sorted(self.data_dir.glob("*.csv"))
#         # Create a map from file stem (number string) to Path object
#         file_map = {f.stem: f for f in files}
#
#         for fname in indices:
#             fname_str = str(fname)  # ensure string
#             if fname_str not in file_map:
#                print(f"!!!!!!!!!!!!!!!!!!!!!!!!! File with name '{fname_str}' not found in {self.data_dir}")
#                continue
#
#             file_path = file_map[fname_str]
#             df = pd.read_csv(file_path)
#             matrix = df.iloc[1:, 1:].values.astype(np.float64)
#             standardized_matrix = self.padding_matrix(matrix)
#             matrices.append(standardized_matrix)
#
#             dataset_name = self.data_dir.name
#             if dataset_name in dict_map_label_dataset:
#                 labels.append(dict_map_label_dataset[dataset_name])
#             else:
#                 raise ValueError(f"Label for dataset '{dataset_name}' not found in mapping.")
#             fname = str(dataset_name).replace("_dataset", "") + '_' + fname_str
#             original_filenames.append(fname)
#
#         return matrices, labels, self.create_sequence_dict(indices), original_filenames
#
#
# # Wrap graphs and labels in a custom DGL Dataset if needed
# # Dataset class
# class GraphDataset(DGLDataset):
#     def __init__(self, graphs, labels, filenames):
#         self.graphs = graphs
#         self.labels = labels
#         self.filenames = filenames
#         super().__init__(name='interaction_graphs')
#
#     def __len__(self):
#         return len(self.graphs)
#
#     def __getitem__(self, idx):
#         return self.graphs[idx], self.labels[idx], self.filenames[idx]
#
# # Custom collate function to batch graphs and labels
# def collate_fn(batch):
#     graphs, labels, filenames = map(list, zip(*batch))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels), filenames
#
# def get_number_by_ID(ID_interaction):
#     match = re.search(r'\d+', ID_interaction)
#     file_number = int(match.group())
#     return int(file_number)
#
#
# def generate_embedding(exp_id, save_embedding_dir):
#     cfg = import_consts(exp_id)
#     # Main cross-validation evaluation loop
#     positive_name = "darnell_human_dataset"
#     negative_name = "NPS_darnell_human_dataset"
#     pos_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / positive_name
#     neg_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / negative_name
#     pos_seq = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / positive_name / "duplex_step"
#     neg_seq = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / negative_name / "duplex_step"
#
#     all_pos_files = sorted(pos_dir.glob("*.csv"))
#     all_neg_files = sorted(neg_dir.glob("*.csv"))
#
#     for fold_dir in sorted(FOLDS_PATH.glob("fold*")):
#         print(f"Processing {fold_dir.name}")
#         train_df = read_csv(fold_dir / f"{fold_dir.name}_train.csv")
#         test_df = read_csv(fold_dir / f"{fold_dir.name}_test.csv")
#         train_df['number_interaction'] = train_df.apply(func=get_wrapper(get_number_by_ID,"ID_interaction"), axis=1)
#         test_df['number_interaction'] = test_df.apply(func=get_wrapper(get_number_by_ID,"ID_interaction"), axis=1)
#
#         train_pos_idx = train_df[train_df['label'] == 1]['number_interaction'].tolist()
#         train_neg_idx = train_df[train_df['label'] == 0]['number_interaction'].tolist()
#         test_pos_idx = test_df[test_df['label'] == 1]['number_interaction'].tolist()
#         test_neg_idx = test_df[test_df['label'] == 0]['number_interaction'].tolist()
#
#         pos_dataset = InteractionDataset(pos_dir, pos_seq, 25, 75)
#         neg_dataset = InteractionDataset(neg_dir, neg_seq, 25, 75)
#
#         train_pos_matrices, train_pos_labels, train_pos_seq_dict, train_pos_names = pos_dataset.get_data(train_pos_idx)
#         train_neg_matrices, train_neg_labels, train_neg_seq_dict, train_neg_names = neg_dataset.get_data(train_neg_idx)
#         test_pos_matrices, test_pos_labels, test_pos_seq_dict, test_pos_names = pos_dataset.get_data(test_pos_idx)
#         test_neg_matrices, test_neg_labels, test_neg_seq_dict, test_neg_names = neg_dataset.get_data(test_neg_idx)
#
#         train_pos_graphs = create_graph_with_features(cfg, train_pos_matrices ,train_pos_seq_dict, train_pos_idx)
#         train_neg_graphs = create_graph_with_features(cfg, train_neg_matrices, train_neg_seq_dict, train_neg_idx)
#
#         test_pos_graphs = create_graph_with_features(cfg, test_pos_matrices, test_pos_seq_dict, test_pos_idx)
#         test_neg_graphs = create_graph_with_features(cfg, test_neg_matrices, test_neg_seq_dict, test_neg_idx)
#
#         # Concatenate for final training/testing sets
#         train_graphs = train_pos_graphs + train_neg_graphs
#         test_graphs = test_pos_graphs + test_neg_graphs
#         train_labels = train_pos_labels + train_neg_labels
#         test_labels = test_pos_labels + test_neg_labels
#         train_filenames = train_pos_names + train_neg_names
#         test_filenames = test_pos_names + test_neg_names
#
#         train_dataset = GraphDataset(train_graphs, train_labels, train_filenames)
#         test_dataset = GraphDataset(test_graphs, test_labels,test_filenames)
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle =True, collate_fn=collate_fn)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
#
#         device = torch.device("cpu")
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         model = Classifier(in_dim=cfg.dim_vector_in, hidden_dim=cfg.dim_vector_out, n_classes=cfg.n_classes).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
#         loss_fn = nn.CrossEntropyLoss()
#
#         # save_embedding_dir = Data_Embedding
#
#         fold_embedding_dir = save_embedding_dir / fold_dir.name
#         os.makedirs(fold_embedding_dir, exist_ok=True)
#         clean_directory(fold_embedding_dir)
#
#         train_losses = []
#         test_losses = []
#         test_accuracies = []
#
#         for epoch in range(cfg.number_epoch):
#             running_loss = 0.0
#             model.train()
#
#             for batched_graph, labels, _ in train_loader:
#                 batched_graph = batched_graph.to(device)
#                 # labels = labels.to(device).float() # match to n_classes=1
#                 labels = labels.to(device).long()
#                 optimizer.zero_grad()
#                 output, _ = model(batched_graph)
#                 # loss = loss_fn(output.squeeze(), labels) # match to n_classes=1
#                 loss = loss_fn(output, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#
#             avg_train_loss = running_loss / len(train_loader)
#             train_losses.append(avg_train_loss)
#
#             model.eval()
#
#             def extract_embeddings_and_loss_accuracy(loader):
#                 embeddings, labels_all, names = [], [], []
#                 probabilities = []  # <-- רשימה חדשה להסתברויות
#                 total_loss = 0.0
#                 total_correct = 0
#                 total_samples = 0
#
#                 with torch.no_grad():
#                     for batched_graph, label, filename_batch in loader:
#                         batched_graph = batched_graph.to(device)
#                         label = label.to(device).long()
#
#                         output, emb = model(batched_graph)
#                         loss = loss_fn(output, label)
#
#                         # <-- חישוב הסתברויות בעזרת Softmax
#                         probs = F.softmax(output, dim=1)
#                         # לוקחים את ההסתברות של המחלקה החיובית (אינדקס 1)
#                         prob_positive = probs[:, 1].cpu().numpy()
#                         probabilities.extend(prob_positive)
#
#                         preds = torch.argmax(output, dim=1)
#
#                         total_loss += loss.item()
#                         total_correct += (preds == label).sum().item()
#                         total_samples += len(label)
#
#                         embeddings.append(emb.cpu().numpy())
#                         labels_all.extend(label.cpu().numpy())
#                         names.extend(filename_batch)
#
#                 avg_loss = total_loss / len(loader)
#                 accuracy = total_correct / total_samples
#
#                 # <-- עכשיו אנחנו מחזירים גם את probabilities
#                 return np.concatenate(embeddings), labels_all, names, avg_loss, accuracy, probabilities
#
#             train_embeddings, train_labels_epoch, train_names_epoch, _, _, train_probs = extract_embeddings_and_loss_accuracy(
#                 train_loader)
#             test_embeddings, test_labels_epoch, test_names_epoch, test_loss, test_acc, test_probs = extract_embeddings_and_loss_accuracy(
#                 test_loader)
#
#             test_losses.append(test_loss)
#             test_accuracies.append(test_acc)
#
#             print(
#                 f"Epoch {epoch + 1}/{cfg.number_epoch} - Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
#
#             # === Save to CSV format per epoch === #
#             def save_embeddings_to_csv(embeddings, names, labels, probs, save_dir, set_type):
#                 epoch_dir = os.path.join(fold_embedding_dir, f"epoch_{epoch}")
#                 os.makedirs(epoch_dir, exist_ok=True)
#
#                 df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
#                 df.insert(0, "ID_interaction", names)
#                 df.insert(1, "True_Label", labels)
#                 df.insert(2, "GCN_Prob_Positive", probs)
#                 df.to_csv(os.path.join(epoch_dir, f"{set_type}_embeddings.csv"), index=False)
#
#             # קריאה לפונקציית השמירה המעודכנת
#             save_embeddings_to_csv(train_embeddings, train_names_epoch, train_labels_epoch, train_probs,
#                                    fold_embedding_dir, "train")
#             save_embeddings_to_csv(test_embeddings, test_names_epoch, test_labels_epoch, test_probs,
#                                    fold_embedding_dir, "test")
#
#
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, cfg.number_epoch + 1), train_losses, label="Train Loss", marker='o')
#         plt.plot(range(1, cfg.number_epoch + 1), test_losses, label="Test Loss", marker='o')
#         plt.plot(range(1, cfg.number_epoch + 1), test_accuracies, label="Test Accuracy", marker='o')
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss / Accuracy")
#         plt.title("Training Progress")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(fold_embedding_dir, "training_plot.png"))
#         plt.show()
#
#         metrics_df = pd.DataFrame({
#             "epoch": list(range(1, cfg.number_epoch + 1)),
#             "train_loss": train_losses,
#             "test_loss": test_losses,
#             "test_accuracy": test_accuracies
#         })
#         metrics_df.to_csv(os.path.join(fold_embedding_dir, "training_metrics.csv"), index=False)

# # print("[INFO] Running embedding generation...")
# # csv_dir = Path('/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0')
# # generate_embedding(26, save_embedding_dir=csv_dir)
#



######################################################################################
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import os
from pathlib import Path
from consts.global_consts import dict_map_label_dataset, FOLDS_PATH, MERGE_DATA, Data_Embedding, ROOT_PATH_PHD_GOAL_ONE, NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS, MATRIX_DATA
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import re
from dgl.data import DGLDataset
from utils.utilsfile import read_csv, to_csv, clean_directory, get_wrapper, import_consts
import matplotlib.pyplot as plt
from Representative_Graph.Features_Graph import create_graph_with_features


#
# # Updated GCNN model
# class Classifier(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes):
#         super(Classifier, self).__init__()
#         self.conv1 = GraphConv(in_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, hidden_dim)
#         self.conv3 = GraphConv(hidden_dim, hidden_dim)
#         self.conv4 = GraphConv(hidden_dim, hidden_dim)
#         self.classify = nn.Linear(hidden_dim, n_classes)
#
#
#     def forward(self, g):
#         h = g.ndata['feature']
#         h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
#         h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
#         h = F.relu(self.conv3(g, h, edge_weight=g.edata['weight']))
#         h = F.relu(self.conv4(g, h, edge_weight=g.edata['weight']))
#         g.ndata['h'] = h
#         hg = dgl.mean_nodes(g, 'h')
#         output = self.classify(hg)
#         return output, hg
#
# # Helper function to load matrix
# def load_graph_matrix(id_interaction):
#     dataset_name = '_'.join(id_interaction.split('_')[:-1])
#     file_num = id_interaction.split('_')[-1]
#     matrix_path = MATRIX_DATA / dataset_name / f"{file_num}.csv"
#     return pd.read_csv(matrix_path).values
#
#
# # represent a one interaction
# class InteractionDataset:
#     def __init__(self, data_dir, data_dir_sequences, target_size_miRNA, target_size_mRNA):
#         self.data_dir = Path(data_dir)
#         self.data_dir_sequences = Path(data_dir_sequences)
#         self.labels = []
#         self.matrices = []
#         self.sequence_mirna = []
#         self.sequence_mrna = []
#         self.target_size_miRNA = target_size_miRNA
#         self.target_size_mRNA = target_size_mRNA
#
#     def padding_matrix(self, matrix):
#         matrix = np.array(matrix, dtype=np.float64)
#         new_matrix = np.zeros((self.target_size_miRNA, self.target_size_mRNA), dtype=np.float64)
#         new_matrix[:min(matrix.shape[0], self.target_size_miRNA),
#         :min(matrix.shape[1], self.target_size_mRNA)] = matrix[:min(matrix.shape[0], self.target_size_miRNA),
#                                                         :min(matrix.shape[1], self.target_size_mRNA)]
#         return new_matrix
#
#     def extract_data_from_csv(self,file_path):
#         # Assuming the CSV has columns 'miRNA_sequence' and 'fragment_x'
#         df = pd.read_csv(file_path)
#         # Extracting values from specific columns
#         miRNA_sequence = df['miRNA_sequence'].iloc[1] if 'miRNA_sequence' in df.columns else None
#         fragment_x = df['fragment_x'].iloc[1] if 'fragment_x' in df.columns else None
#         return miRNA_sequence, fragment_x
#
#     def create_sequence_dict(self, indices):
#         sequence_dict = {}
#         for file_path in self.data_dir_sequences.glob("*.csv"):
#             # Extract number from filename
#             match = re.search(r'\d+', file_path.stem)
#             if match:
#                 file_number = int(match.group())
#                 if file_number in indices:
#                     miRNA_sequence, fragment_x = self.extract_data_from_csv(file_path)
#                     sequence_dict[file_number] = {
#                         'miRNA_sequence': miRNA_sequence,
#                         'fragment': fragment_x
#                     }
#         return sequence_dict
#
#
#     def get_data(self, indices):
#         matrices, labels, original_filenames = [], [], []
#         files = sorted(self.data_dir.glob("*.csv"))
#         # Create a map from file stem (number string) to Path object
#         file_map = {f.stem: f for f in files}
#
#         for fname in indices:
#             fname_str = str(fname)  # ensure string
#             if fname_str not in file_map:
#                print(f"!!!!!!!!!!!!!!!!!!!!!!!!! File with name '{fname_str}' not found in {self.data_dir}")
#                continue
#
#             file_path = file_map[fname_str]
#             df = pd.read_csv(file_path)
#             matrix = df.iloc[1:, 1:].values.astype(np.float64)
#             standardized_matrix = self.padding_matrix(matrix)
#             matrices.append(standardized_matrix)
#
#             dataset_name = self.data_dir.name
#             if dataset_name in dict_map_label_dataset:
#                 labels.append(dict_map_label_dataset[dataset_name])
#             else:
#                 raise ValueError(f"Label for dataset '{dataset_name}' not found in mapping.")
#             fname = str(dataset_name).replace("_dataset", "") + '_' + fname_str
#             original_filenames.append(fname)
#
#         return matrices, labels, self.create_sequence_dict(indices), original_filenames
#
#
# # Wrap graphs and labels in a custom DGL Dataset if needed
# # Dataset class
# class GraphDataset(DGLDataset):
#     def __init__(self, graphs, labels, filenames):
#         self.graphs = graphs
#         self.labels = labels
#         self.filenames = filenames
#         super().__init__(name='interaction_graphs')
#
#     def __len__(self):
#         return len(self.graphs)
#
#     def __getitem__(self, idx):
#         return self.graphs[idx], self.labels[idx], self.filenames[idx]
#
# # Custom collate function to batch graphs and labels
# def collate_fn(batch):
#     graphs, labels, filenames = map(list, zip(*batch))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels), filenames
#
# def get_number_by_ID(ID_interaction):
#     match = re.search(r'\d+', ID_interaction)
#     file_number = int(match.group())
#     return int(file_number)
#
#
# def generate_embedding(exp_id, save_embedding_dir):
#     cfg = import_consts(exp_id)
#     # Main cross-validation evaluation loop
#     positive_name = "darnell_human_dataset"
#     negative_name = "NPS_darnell_human_dataset"
#     pos_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / positive_name
#     neg_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / negative_name
#     pos_seq = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / positive_name / "duplex_step"
#     neg_seq = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex" / negative_name / "duplex_step"
#
#     all_pos_files = sorted(pos_dir.glob("*.csv"))
#     all_neg_files = sorted(neg_dir.glob("*.csv"))
#
#     for fold_dir in sorted(FOLDS_PATH.glob("fold*")):
#         print(f"Processing {fold_dir.name}")
#         train_df = read_csv(fold_dir / f"{fold_dir.name}_train.csv")
#         test_df = read_csv(fold_dir / f"{fold_dir.name}_test.csv")
#         train_df['number_interaction'] = train_df.apply(func=get_wrapper(get_number_by_ID,"ID_interaction"), axis=1)
#         test_df['number_interaction'] = test_df.apply(func=get_wrapper(get_number_by_ID,"ID_interaction"), axis=1)
#
#         train_pos_idx = train_df[train_df['label'] == 1]['number_interaction'].tolist()
#         train_neg_idx = train_df[train_df['label'] == 0]['number_interaction'].tolist()
#         test_pos_idx = test_df[test_df['label'] == 1]['number_interaction'].tolist()
#         test_neg_idx = test_df[test_df['label'] == 0]['number_interaction'].tolist()
#
#         pos_dataset = InteractionDataset(pos_dir, pos_seq, 25, 75)
#         neg_dataset = InteractionDataset(neg_dir, neg_seq, 25, 75)
#
#         train_pos_matrices, train_pos_labels, train_pos_seq_dict, train_pos_names = pos_dataset.get_data(train_pos_idx)
#         train_neg_matrices, train_neg_labels, train_neg_seq_dict, train_neg_names = neg_dataset.get_data(train_neg_idx)
#         test_pos_matrices, test_pos_labels, test_pos_seq_dict, test_pos_names = pos_dataset.get_data(test_pos_idx)
#         test_neg_matrices, test_neg_labels, test_neg_seq_dict, test_neg_names = neg_dataset.get_data(test_neg_idx)
#
#         train_pos_graphs = create_graph_with_features(cfg, train_pos_matrices ,train_pos_seq_dict, train_pos_idx)
#         train_neg_graphs = create_graph_with_features(cfg, train_neg_matrices, train_neg_seq_dict, train_neg_idx)
#
#         test_pos_graphs = create_graph_with_features(cfg, test_pos_matrices, test_pos_seq_dict, test_pos_idx)
#         test_neg_graphs = create_graph_with_features(cfg, test_neg_matrices, test_neg_seq_dict, test_neg_idx)
#
#         # Concatenate for final training/testing sets
#         train_graphs = train_pos_graphs + train_neg_graphs
#         test_graphs = test_pos_graphs + test_neg_graphs
#         train_labels = train_pos_labels + train_neg_labels
#         test_labels = test_pos_labels + test_neg_labels
#         train_filenames = train_pos_names + train_neg_names
#         test_filenames = test_pos_names + test_neg_names
#
#         train_dataset = GraphDataset(train_graphs, train_labels, train_filenames)
#         test_dataset = GraphDataset(test_graphs, test_labels,test_filenames)
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle =True, collate_fn=collate_fn)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
#
#         device = torch.device("cpu")
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         model = Classifier(in_dim=cfg.dim_vector_in, hidden_dim=cfg.dim_vector_out, n_classes=cfg.n_classes).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
#         loss_fn = nn.CrossEntropyLoss()
#
#         fold_embedding_dir = save_embedding_dir / fold_dir.name
#         os.makedirs(fold_embedding_dir, exist_ok=True)
#         clean_directory(fold_embedding_dir)
#
#         # משתנה למעקב אחרי ה-AUC הגבוה ביותר לשמירת ה-Checkpoint
#         best_test_auc = 0.0
#         best_model_state = None
#
#         # רשימות למעקב אחרי כל המדדים באפוקים
#         train_losses = []
#         test_losses = []
#         test_accuracies = []
#         test_aucs = []
#         test_prcs = []
#         test_f1s = []
#         test_precisions = []
#         test_recalls = []
#
#         for epoch in range(cfg.number_epoch):
#             running_loss = 0.0
#             model.train()
#
#             for batched_graph, labels, _ in train_loader:
#                 batched_graph = batched_graph.to(device)
#                 labels = labels.to(device).long()
#                 optimizer.zero_grad()
#                 output, _ = model(batched_graph)
#                 loss = loss_fn(output, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#
#             avg_train_loss = running_loss / len(train_loader)
#             train_losses.append(avg_train_loss)
#
#             model.eval()
#
#             def extract_embeddings_and_loss_accuracy(loader):
#                 embeddings, labels_all, names, probabilities, preds_all = [], [], [], [], []
#                 total_loss = 0.0
#
#                 with torch.no_grad():
#                     for batched_graph, label, filename_batch in loader:
#                         batched_graph = batched_graph.to(device)
#                         label = label.to(device).long()
#
#                         output, emb = model(batched_graph)
#                         loss = loss_fn(output, label)
#                         total_loss += loss.item()
#
#                         probs = F.softmax(output, dim=1)
#                         prob_positive = probs[:, 1].cpu().numpy()
#                         probabilities.extend(prob_positive)
#
#                         preds = torch.argmax(output, dim=1).cpu().numpy()
#                         preds_all.extend(preds)
#
#                         embeddings.append(emb.cpu().numpy())
#                         labels_all.extend(label.cpu().numpy())
#                         names.extend(filename_batch)
#
#                 avg_loss = total_loss / len(loader)
#
#                 # המרת רשימות למערכים לחישוב נוח
#                 y_true = np.array(labels_all)
#                 y_pred = np.array(preds_all)
#                 y_prob = np.array(probabilities)
#
#                 # חישוב כל המדדים החשובים
#                 acc = accuracy_score(y_true, y_pred)
#                 try:
#                     auc = roc_auc_score(y_true, y_prob)
#                     prc = average_precision_score(y_true, y_prob)
#                     prec = precision_score(y_true, y_pred, zero_division=0)
#                     rec = recall_score(y_true, y_pred, zero_division=0)
#                     f1 = f1_score(y_true, y_pred, zero_division=0)
#                 except ValueError:
#                     # הגנה למקרה שיש רק מחלקה אחת בבאצ' (נדיר בטסט אבל יכול לקרות בדאטא קטן)
#                     auc, prc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
#
#                 return np.concatenate(embeddings), y_true, names, avg_loss, acc, y_prob, auc, prc, prec, rec, f1
#
#             # הרצת בדיקה על סט האימון
#             train_embeddings, train_labels_epoch, train_names_epoch, _, _, train_probs, _, _, _, _, _ = extract_embeddings_and_loss_accuracy(
#                 train_loader)
#
#             # הרצת בדיקה על סט האימות/טסט וחילוץ כל המדדים
#             test_embeddings, test_labels_epoch, test_names_epoch, test_loss, test_acc, test_probs, test_auc, test_prc, test_prec, test_rec, test_f1 = extract_embeddings_and_loss_accuracy(
#                 test_loader)
#
#             # שמירת המדדים לרשימות
#             test_losses.append(test_loss)
#             test_accuracies.append(test_acc)
#             test_aucs.append(test_auc)
#             test_prcs.append(test_prc)
#             test_f1s.append(test_f1)
#             test_precisions.append(test_prec)
#             test_recalls.append(test_rec)
#
#             print(
#                 f"Epoch {epoch + 1}/{cfg.number_epoch} - Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | AUC: {test_auc:.4f} | AUPRC: {test_prc:.4f} | F1: {test_f1:.4f}")
#
#             # Checkpoint: שמירת המודל רק אם ה-AUC השתפר
#             if test_auc > best_test_auc:
#                 best_test_auc = test_auc
#                 best_model_state = model.state_dict()
#                 torch.save(best_model_state, os.path.join(fold_embedding_dir, f"best_model_fold_{fold_dir.name}.pth"))
#
#             # שמירת ה-Embeddings ל-CSV לפי אפוק
#             def save_embeddings_to_csv(embeddings, names, labels, probs, save_dir, set_type):
#                 epoch_dir = os.path.join(fold_embedding_dir, f"epoch_{epoch}")
#                 os.makedirs(epoch_dir, exist_ok=True)
#
#                 df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
#                 df.insert(0, "ID_interaction", names)
#                 df.insert(1, "True_Label", labels)
#                 df.insert(2, "GCN_Prob_Positive", probs)
#                 df.to_csv(os.path.join(epoch_dir, f"{set_type}_embeddings.csv"), index=False)
#
#             save_embeddings_to_csv(train_embeddings, train_names_epoch, train_labels_epoch, train_probs,
#                                    fold_embedding_dir, "train")
#             save_embeddings_to_csv(test_embeddings, test_names_epoch, test_labels_epoch, test_probs, fold_embedding_dir,
#                                    "test")
#
#         # שמירת קובץ ה-CSV המורחב עם כל המדדים לדיווח במאמר
#         metrics_df = pd.DataFrame({
#             "fold": [fold_dir.name] * cfg.number_epoch,
#             "epoch": list(range(1, cfg.number_epoch + 1)),
#             "train_loss": train_losses,
#             "test_loss": test_losses,
#             "test_accuracy": test_accuracies,
#             "test_auc": test_aucs,
#             "test_auprc": test_prcs,
#             "test_precision": test_precisions,
#             "test_recall": test_recalls,
#             "test_f1": test_f1s
#         })
#         metrics_df.to_csv(os.path.join(fold_embedding_dir, f"training_metrics_comprehensive_{fold_dir.name}.csv"),
#                           index=False)
#
#         # שחרור זיכרון לשרת בין פולד לפולד
#         del model, optimizer, train_loader, test_loader
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import RelGraphConv
import os
from pathlib import Path
from consts.global_consts import dict_map_label_dataset, FOLDS_PATH, MERGE_DATA, Data_Embedding, ROOT_PATH_PHD_GOAL_ONE, \
    NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS, MATRIX_DATA
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split  # <-- ייבוא הפונקציה לפיצול
import re
from dgl.data import DGLDataset
from utils.utilsfile import read_csv, to_csv, clean_directory, get_wrapper, import_consts
import matplotlib.pyplot as plt
from Representative_Graph.Features_Graph import create_graph_with_features

#
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
        h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv3(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv4(g, h, edge_weight=g.edata['weight']))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        output = self.classify(hg)
        return output, hg

# class Classifier(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, num_rels=3):
#         super(Classifier, self).__init__()
#         # משתמשים ב-RelGraphConv במקום GraphConv.
#         # num_rels=3 אומר לרשת שיש 3 סוגי קשרים (זיווג, שלד, לולאה עצמית)
#         self.conv1 = RelGraphConv(in_dim, hidden_dim, num_rels, regularizer='basis', num_bases=2)
#         self.conv2 = RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer='basis', num_bases=2)
#         self.conv3 = RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer='basis', num_bases=2)
#         self.conv4 = RelGraphConv(hidden_dim, hidden_dim, num_rels, regularizer='basis', num_bases=2)
#         self.classify = nn.Linear(hidden_dim, n_classes)
#
#     def forward(self, g):
#
#         h = g.ndata['feature']
#         etypes = g.edata['edge_type']
#
#         # 🌟 התיקון כאן: הופכים את המימד מ-(E,) ל-(E, 1)
#         norm = g.edata['weight'].unsqueeze(1)
#
#         h = F.relu(self.conv1(g, h, etypes, norm))
#         h = F.relu(self.conv2(g, h, etypes, norm))
#         h = F.relu(self.conv3(g, h, etypes, norm))
#         h = F.relu(self.conv4(g, h, etypes, norm))
#
#         g.ndata['h'] = h
#         hg = dgl.mean_nodes(g, 'h')
#         logits = self.classify(hg)
#
#         return logits, hg

def load_graph_matrix(id_interaction):
    dataset_name = '_'.join(id_interaction.split('_')[:-1])
    file_num = id_interaction.split('_')[-1]
    matrix_path = MATRIX_DATA / dataset_name / f"{file_num}.csv"
    return pd.read_csv(matrix_path).values


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

    def extract_data_from_csv(self, file_path):
        df = pd.read_csv(file_path)
        miRNA_sequence = df['miRNA_sequence'].iloc[1] if 'miRNA_sequence' in df.columns else None
        fragment_x = df['fragment_x'].iloc[1] if 'fragment_x' in df.columns else None
        return miRNA_sequence, fragment_x

    def create_sequence_dict(self, indices):
        sequence_dict = {}
        for file_path in self.data_dir_sequences.glob("*.csv"):
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
        file_map = {f.stem: f for f in files}

        for fname in indices:
            fname_str = str(fname)
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


def collate_fn(batch):
    graphs, labels, filenames = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), filenames


def get_number_by_ID(ID_interaction):
    match = re.search(r'\d+', ID_interaction)
    file_number = int(match.group())
    return int(file_number)


def generate_embedding(exp_id, save_embedding_dir):
    cfg = import_consts(exp_id)
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

        # 1. קריאת קבצי ה-Train וה-Test המקוריים
        full_train_df = read_csv(fold_dir / f"{fold_dir.name}_train.csv")
        test_df = read_csv(fold_dir / f"{fold_dir.name}_test.csv")

        # 2. הוספת עמודת המספר המזהה
        full_train_df['number_interaction'] = full_train_df.apply(func=get_wrapper(get_number_by_ID, "ID_interaction"),
                                                                  axis=1)
        test_df['number_interaction'] = test_df.apply(func=get_wrapper(get_number_by_ID, "ID_interaction"), axis=1)

        # 3. פיצול ה-Train המלא ל-Train אמיתי (90%) ו-Validation (10%) 'On the fly'
        # משתמשים ב-stratify כדי לשמור על היחס בין חיוביים לשליליים
        train_df, val_df = train_test_split(
            full_train_df,
            test_size=0.10,
            stratify=full_train_df['label'],
            random_state=42
        )

        print(f"  --> Split Info: Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

        # 4. חילוץ האינדקסים עבור 3 הסטים
        train_pos_idx = train_df[train_df['label'] == 1]['number_interaction'].tolist()
        train_neg_idx = train_df[train_df['label'] == 0]['number_interaction'].tolist()

        val_pos_idx = val_df[val_df['label'] == 1]['number_interaction'].tolist()
        val_neg_idx = val_df[val_df['label'] == 0]['number_interaction'].tolist()

        test_pos_idx = test_df[test_df['label'] == 1]['number_interaction'].tolist()
        test_neg_idx = test_df[test_df['label'] == 0]['number_interaction'].tolist()

        pos_dataset = InteractionDataset(pos_dir, pos_seq, 25, 75)
        neg_dataset = InteractionDataset(neg_dir, neg_seq, 25, 75)

        # 5. הבאת הנתונים ל-3 הסטים
        train_pos_matrices, train_pos_labels, train_pos_seq_dict, train_pos_names = pos_dataset.get_data(train_pos_idx)
        train_neg_matrices, train_neg_labels, train_neg_seq_dict, train_neg_names = neg_dataset.get_data(train_neg_idx)

        val_pos_matrices, val_pos_labels, val_pos_seq_dict, val_pos_names = pos_dataset.get_data(val_pos_idx)
        val_neg_matrices, val_neg_labels, val_neg_seq_dict, val_neg_names = neg_dataset.get_data(val_neg_idx)

        test_pos_matrices, test_pos_labels, test_pos_seq_dict, test_pos_names = pos_dataset.get_data(test_pos_idx)
        test_neg_matrices, test_neg_labels, test_neg_seq_dict, test_neg_names = neg_dataset.get_data(test_neg_idx)

        # 6. יצירת הגרפים ל-3 הסטים
        train_pos_graphs = create_graph_with_features(cfg, train_pos_matrices, train_pos_seq_dict, train_pos_idx)
        train_neg_graphs = create_graph_with_features(cfg, train_neg_matrices, train_neg_seq_dict, train_neg_idx)

        val_pos_graphs = create_graph_with_features(cfg, val_pos_matrices, val_pos_seq_dict, val_pos_idx)
        val_neg_graphs = create_graph_with_features(cfg, val_neg_matrices, val_neg_seq_dict, val_neg_idx)

        test_pos_graphs = create_graph_with_features(cfg, test_pos_matrices, test_pos_seq_dict, test_pos_idx)
        test_neg_graphs = create_graph_with_features(cfg, test_neg_matrices, test_neg_seq_dict, test_neg_idx)

        # 7. איחוד לסטים סופיים
        train_graphs = train_pos_graphs + train_neg_graphs
        val_graphs = val_pos_graphs + val_neg_graphs
        test_graphs = test_pos_graphs + test_neg_graphs

        train_labels = train_pos_labels + train_neg_labels
        val_labels = val_pos_labels + val_neg_labels
        test_labels = test_pos_labels + test_neg_labels

        train_filenames = train_pos_names + train_neg_names
        val_filenames = val_pos_names + val_neg_names
        test_filenames = test_pos_names + test_neg_names

        train_dataset = GraphDataset(train_graphs, train_labels, train_filenames)
        val_dataset = GraphDataset(val_graphs, val_labels, val_filenames)
        test_dataset = GraphDataset(test_graphs, test_labels, test_filenames)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                 collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                  collate_fn=collate_fn)

        device = torch.device("cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Classifier(in_dim=cfg.dim_vector_in, hidden_dim=cfg.dim_vector_out, n_classes=cfg.n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        fold_embedding_dir = save_embedding_dir / fold_dir.name
        os.makedirs(fold_embedding_dir, exist_ok=True)
        clean_directory(fold_embedding_dir)

        # שמירת המודל מבוססת כעת על ה-Validation AUC המקסימלי
        best_val_auc = 0.0
        best_model_state = None

        # רשימות מעקב
        train_losses = []
        val_losses = []
        val_aucs = []
        test_losses = []
        test_accuracies = []
        test_aucs = []
        test_prcs = []
        test_f1s = []
        test_precisions = []
        test_recalls = []

        for epoch in range(cfg.number_epoch):
            running_loss = 0.0
            model.train()

            for batched_graph, labels, _ in train_loader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()
                output, _ = model(batched_graph)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()

            def extract_embeddings_and_loss_accuracy(loader):
                embeddings, labels_all, names, probabilities, preds_all = [], [], [], [], []
                total_loss = 0.0

                with torch.no_grad():
                    for batched_graph, label, filename_batch in loader:
                        batched_graph = batched_graph.to(device)
                        label = label.to(device).long()

                        output, emb = model(batched_graph)
                        loss = loss_fn(output, label)
                        total_loss += loss.item()

                        probs = F.softmax(output, dim=1)
                        prob_positive = probs[:, 1].cpu().numpy()
                        probabilities.extend(prob_positive)

                        preds = torch.argmax(output, dim=1).cpu().numpy()
                        preds_all.extend(preds)

                        embeddings.append(emb.cpu().numpy())
                        labels_all.extend(label.cpu().numpy())
                        names.extend(filename_batch)

                avg_loss = total_loss / len(loader)

                y_true = np.array(labels_all)
                y_pred = np.array(preds_all)
                y_prob = np.array(probabilities)

                acc = accuracy_score(y_true, y_pred)
                try:
                    auc = roc_auc_score(y_true, y_prob)
                    prc = average_precision_score(y_true, y_prob)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                except ValueError:
                    auc, prc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0, 0.0

                return np.concatenate(embeddings), y_true, names, avg_loss, acc, y_prob, auc, prc, prec, rec, f1

            # הרצה על כל 3 הסטים
            train_embeddings, train_labels_epoch, train_names_epoch, _, _, train_probs, _, _, _, _, _ = extract_embeddings_and_loss_accuracy(
                train_loader)
            val_embeddings, val_labels_epoch, val_names_epoch, val_loss, val_acc, val_probs, val_auc, val_prc, val_prec, val_rec, val_f1 = extract_embeddings_and_loss_accuracy(
                val_loader)
            test_embeddings, test_labels_epoch, test_names_epoch, test_loss, test_acc, test_probs, test_auc, test_prc, test_prec, test_rec, test_f1 = extract_embeddings_and_loss_accuracy(
                test_loader)

            # שמירת מדדים
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            test_aucs.append(test_auc)
            test_prcs.append(test_prc)
            test_f1s.append(test_f1)
            test_precisions.append(test_prec)
            test_recalls.append(test_rec)

            print(
                f"Epoch {epoch + 1}/{cfg.number_epoch} - Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | Test AUPRC: {test_prc:.4f}")

            # Checkpoint: שמירת המודל על בסיס סט ה-Validation בלבד!
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()
                torch.save(best_model_state, os.path.join(fold_embedding_dir, f"best_model_fold_{fold_dir.name}.pth"))

            # שמירת ה-Embeddings ל-CSV לכל סוגי הסטים
            def save_embeddings_to_csv(embeddings, names, labels, probs, save_dir, set_type):
                epoch_dir = os.path.join(fold_embedding_dir, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)

                df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
                df.insert(0, "ID_interaction", names)
                df.insert(1, "True_Label", labels)
                df.insert(2, "GCN_Prob_Positive", probs)
                df.to_csv(os.path.join(epoch_dir, f"{set_type}_embeddings.csv"), index=False)

            save_embeddings_to_csv(train_embeddings, train_names_epoch, train_labels_epoch, train_probs,
                                   fold_embedding_dir, "train")
            save_embeddings_to_csv(val_embeddings, val_names_epoch, val_labels_epoch, val_probs, fold_embedding_dir,
                                   "val")
            save_embeddings_to_csv(test_embeddings, test_names_epoch, test_labels_epoch, test_probs, fold_embedding_dir,
                                   "test")

        # שמירת קובץ התוצאות המסכם
        metrics_df = pd.DataFrame({
            "fold": [fold_dir.name] * cfg.number_epoch,
            "epoch": list(range(1, cfg.number_epoch + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_auc": val_aucs,
            "test_loss": test_losses,
            "test_accuracy": test_accuracies,
            "test_auc": test_aucs,
            "test_auprc": test_prcs,
            "test_precision": test_precisions,
            "test_recall": test_recalls,
            "test_f1": test_f1s
        })
        metrics_df.to_csv(os.path.join(fold_embedding_dir, f"training_metrics_comprehensive_{fold_dir.name}.csv"),
                          index=False)

        # שחרור זיכרון
        del model, optimizer, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


