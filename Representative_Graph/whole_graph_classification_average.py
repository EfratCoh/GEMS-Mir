import torch
import dgl
import os
import json
from operator import itemgetter
from sklearn.model_selection import LeaveOneOut
# from PCA import main_program as mp
import networkx as nx
from torch.utils.data import DataLoader
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

task ='average'
output_embedding_size=300
label_file_name="training_with_hypers/graphs_labels_average.txt"
current_datasets_dir='nodeEmbeddings/'
# מאיזה תיקיה קוראים את המאפיניים של הקודקודים
dir_nodes_embedding=current_datasets_dir
dir = 'EdgeLists/'
running_time_dir ="Run_time_gcnn"
epochs = 80


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
        h = g.in_degrees().view(-1, 1).float()
        h.data = embedding
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

def get_labels(labels_path):
    f = open(labels_path)
    line = f.readline()
    dict_labels={}
    while line:
        splits=line.split(";")
        dataset_name = splits[0]
        dataset_label=int(splits[1].split("\n")[0])
        dict_labels[dataset_name]=dataset_label
        line= f.readline()
    f.close()
    return dict_labels

# לכל גרף שולף לכל נוד את האימבדינג
def get_node_embedding(embedding_dir):
    print('get graph from ' + embedding_dir)
    num_embeddings = sum([len(files) for root, dirs, files in os.walk(embedding_dir)])
    graphs = []
    graphs_dic={}
    for file in sorted(os.listdir(embedding_dir), key=lambda s: s.lower()):
        f = open(embedding_dir+"/"+ file, "r")
        line = f.readline()
        head = line.strip().split(" ")
        node_num = head[0]
        node_dim = head[1]
        line = f.readline()
        vector_dict = {}
        while line:
            curline = line.strip().split(" ")
            vector_dict[int(curline[0])] = [float(s) for s in curline[1:]]
            line = f.readline()
        f.close()

        vector_dict = dict(sorted(vector_dict.items(), key=lambda e: e[0]))
        graphs.append(np.array(list(vector_dict.values())))
        graphs_dic[file.split(".")[0]]=np.array(list(vector_dict.values()))
    graphs = np.asarray(graphs)
    return graphs_dic


def collate(samples):
    graphs, labels, embedding= map(list,zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), embedding

#יוצר טאפל לכל גרף , שומר את הגרף, הלייבל של הגרף, לוקח את האימבדינג של כל הקודקודים ושם בטאפך
def retrive_graphs(dict_nodes_embedding):
    #Graph_list = mp()
    Graph_list=[]
    for file in sorted(os.listdir(current_datasets_dir),key=lambda s: s.lower()):
        file = file.split('\\')[-1].split('.')[0]
        G = nx.read_edgelist(dir+file+".file", create_using=nx.Graph, nodetype=int, data=(('weight',float),))
        G = nx.Graph(G)
        Graph_list.append((file.split(".")[0], G))
    datasets = []
    # dict_labels = get_labels("training/graphs_labels"+index+".txt")
    dict_labels = get_labels(label_file_name)
    for (dataset_name,graph) in Graph_list:
        c = dgl.DGLGraph()
        c.from_networkx(graph, edge_attrs=['weight'])
        c.set_n_initializer(dgl.init.zero_initializer)
        datasets.append((c, dict_labels[dataset_name],torch.FloatTensor(dict_nodes_embedding[dataset_name])))

    return datasets

# מאתחל את האימבדינג ברמת הגרף
# איניט לנוג אימבדינג של הקודקודים
batch_size = 20
node_embedding_dim = 64
def initiate_feat(embedding):
    embedding_array=[]
    if len(embedding) == 1:
        embedding_array = torch.Tensor(embedding[0])
    else:
        for j in range(1, len(embedding)):
            if j == 1:
                embedding_array = torch.FloatTensor(np.concatenate((embedding[j - 1], embedding[j])))
            else:
                embedding_array = torch.FloatTensor(np.concatenate((embedding_array, embedding[j])))
    # embedding_array=embedding_array
    # embedding_array = torch.from_numpy(embedding_array).double()
    # dgl.base_initializer([batch_size, node_embedding_dim], float, F.cpu(), )
    return embedding_array



def graphClassification():
    start = time.time()
    files_name= [file for file in sorted(os.listdir(current_datasets_dir),key=lambda s: s.lower())]
    print(len(files_name))
    dict_nodes_embedding = get_node_embedding(dir_nodes_embedding)
    graphs = retrive_graphs(dict_nodes_embedding)

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

        # pay attention is for binary
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