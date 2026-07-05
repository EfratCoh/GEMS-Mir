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
import networkx as nx
import matplotlib.pyplot as plt


ROOT_PATH_PHD_GOAL_ONE = Path("/sise/home/efrco/PHD/Goal_One/")
dict_map_label_dataset = {"darnell_human_dataset":0, "NPS_darnell_human_dataset":1}


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

    def forward(self, g, embedding):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h=g.in_degrees()
        h = g.in_degrees().view(-1, 1).float()
        h.data = embedding
        #h = embedding.tousertensor()
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
            # Load the matrix from CSV, skipping the first row and column
            df = pd.read_csv(file_path)
            matrix = df.iloc[1:, 1:].values.astype(np.float64)  # Convert to float
            standardized_matrix = self.padding_matrix(matrix)

            # Append the matrix to the list
            self.matrices.append(standardized_matrix)

            # Map the dataset directory name to its label
            dataset_name = self.data_dir.name
            if dataset_name in dict_map_label_dataset:
                self.labels.append(dict_map_label_dataset[dataset_name])
            else:
                raise ValueError(f"Label for dataset '{dataset_name}' not found in mapping.")


    def padding_matrix(self, matrix):
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

def matrix_to_graph(matrix, row_label="miRNA", col_label="fragment"):
    matrix = np.array(matrix, dtype=np.float64)  # Ensure the matrix is of type float
    G = nx.DiGraph()

    num_rows, num_cols = matrix.shape

    # Create row nodes (e.g., miRNA)
    row_nodes = [f"{i}_{row_label}" for i in range(num_rows)]

    # Create column nodes (e.g., fragments)
    col_nodes = [f"{j}_{col_label}" for j in range(num_cols)]

    # Add edges between row and column nodes for non-zero matrix entries
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i, j] != 0:
                G.add_edge(f"{i}_{row_label}", f"{j}_{col_label}", weight=matrix[i, j])

    c = dgl.from_networkx(G, edge_attrs=['weight'])
    return c



def graphClassification():
    dataset_name = "darnell_human_dataset"
    data_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_Graph_Matrix" / dataset_name
    dataset = InteractionDataset(data_dir, 32, 75)
    matrices, labels = dataset.get_data()
    for i, matrix in enumerate(matrices):
        print(f"Matrix {i} shape: {matrix.shape}")
    # Convert matrices to graphs
    graphs = [matrix_to_graph(matrix) for matrix in matrices]

    dict_nodes_embedding = get_node_embedding(dir_nodes_embedding)
    graphs= retrive_graphs(dict_nodes_embedding)
    loo = LeaveOneOut()
    # lambda x: x[0] for x in
    splits = loo.split(graphs)

    for train_index, test_index in splits:
        print(test_index[0])
        train_index = np.insert(train_index, 0,test_index[0])
        # train_index = np.append(train_index,test_index[0])
        train_set=itemgetter(*train_index)(graphs)
        test_set=[itemgetter(*test_index)(graphs)]

        #test_X, test_Y, emb_test = map(list, zip(*test_set))

        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    # Create model
        for i in range(0, batch_size-1):
            c = dgl.DGLGraph()
            c.add_nodes(1)
            test_set.append((c,0,torch.FloatTensor(torch.zeros(1,64))))
        #test_X, test_Y, embedding_test = map(list, zip(*test_set))
        num_classes =17
        model = Classifier(node_embedding_dim, output_embedding_size, num_classes)
        # weights =[99/3, 99/8, 99/99, 99/9, 99/4,99/24,99/69,99/3,99/2,99/6]
        # class_weights = torch.FloatTensor(weights).to(device='cpu')
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.006)
        model.train()
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            counter =0
            d = "Embedding_" +task +"ver1/" +"embedding_"+task + str(epoch)
            if not os.path.exists(d):
                os.mkdir(d)
            # os.mkdir(d)
            for iter, (bg, label, embedding) in enumerate(data_loader):
                embedding_array=[]
                embedding_array=initiate_feat(embedding)
                prediction,hg = model(bg, embedding_array)
                # if write == True:
                # if epoch == epochs -1:
                # if epoch_loss < np.mean(epoch_losses) < 0.4:
                for hidden in hg.detach().numpy():
                    dataset_hidden_name=files_name[train_index[counter]].split(".")[0]
                    hidden.tofile(d+"/"+ dataset_hidden_name + ".csv",sep=',')
                    counter = counter + 1
                print("finish writing")

                #label = torch.stack(label)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)
            end = time.time() - start
            run_time_path = os.path.join(os.getcwd(), running_time_dir, task, str(epoch) + ".txt")
            # if not os.path.exists(os.getcwd()+"/"+running_time_dir+"/"+task):
            # os.mkdir(os.getcwd()+running_time_dir+"/"+task)
            f = open(run_time_path, "w")
            total = time.strftime("%H:%M:%S", time.gmtime(end))
            f.write(total)
            f.close()

        model.eval()
            # Convert a list of tuples to two lists
        test_X, test_Y,embedding_test = map(list, zip(*test_set))
        test_bg = dgl.batch(test_X)
        true_label=test_Y[0]
        dataset_test_name =files_name[test_index[0]].split(".")[0]
        print(dataset_test_name)
        test_Y = torch.tensor(test_Y).float().view(-1, 1)
        probs_Y,hidden_layer = model(test_bg,initiate_feat(embedding_test))
        probs_Y = torch.softmax(probs_Y, 1)
        print(probs_Y.detach().numpy()[0])
        #hidden_layer.detach().numpy()[0].tofile("representation/"+dataset_test_name+".csv",sep=',')
        print(probs_Y)
        print("True label:")
        print(true_label)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
        print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
        break
graphClassification()
