import pickle
import numpy as np
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch_geometric
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def sampling(src_nodes, sample_num, neighbour_table):
    results = []  
    for sid in src_nodes:
        res = np.random.choice(neighbour_table[sid], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten() 

def multihop_sampling(src_nodes, sample_nums, neighbour_table):
    sampling_result = [src_nodes]  
    for k, hopk_num in enumerate(sample_nums):  
        hopk_result = sampling(sampling_result[k], hopk_num, neighbour_table)
        sampling_result.append(hopk_result)
    return sampling_result

class NeighbourAggregator(nn.Module):  
    def __init__(self, input_dim, output_dim, use_bias=False, aggregation_func="mean"):
        super(NeighbourAggregator, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggregation_func = aggregation_func
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbour_feature):
        if self.aggregation_func == "mean":
            aggr_neighbour = neighbour_feature.mean(dim=1)  
        elif self.aggregation_func == "pool_max":
            aggr_neighbour, _ = neighbour_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, pool_max but got {}".format(self.aggregation_func))

        neighbour_hidden = torch.matmul(aggr_neighbour, self.weight)
        if self.use_bias:
            neighbour_hidden += self.bias
        return neighbour_hidden  

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggregation_func={}'.format(self.input_dim, self.output_dim, self.aggregation_func)

class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, aggregation_func="mean"):
        super(SageGCN, self).__init__()

        assert aggregation_func in ["mean", "pool_max"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation_func = aggregation_func
        self.activation = activation
        self.aggregator = NeighbourAggregator(input_dim, hidden_dim, aggregation_func=aggregation_func)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbour_node_features):
        neighbour_hidden = self.aggregator(neighbour_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
  
        hidden = torch.cat([self_hidden, neighbour_hidden], dim=1)

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim * 2
        return 'in_features={}, out_features={}'.format(self.input_dim, output_dim)

class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbours_list, aggregation_func="mean"):

        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbours_list = num_neighbours_list
        self.num_layers = len(num_neighbours_list)

        self.aggregation_func = aggregation_func

        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0], aggregation_func=aggregation_func))
        for index in range(0, len(hidden_dim) - 2):
            hidden_dim[index] *= 2
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1], aggregation_func=aggregation_func))
        
        hidden_dim[-2] *= 2
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None, aggregation_func=aggregation_func))  

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbour_node_features = hidden[hop + 1].view((src_node_num, self.num_neighbours_list[hop], -1))
                h = gcn(src_node_features, neighbour_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]  

    def extra_repr(self):
        return 'in_features={}, num_neighbours_list={}'.format(self.input_dim, self.num_neighbours_list)

num_layers = int(sys.argv[1])
assert num_layers in [2, 3]

INPUT_DIM = 3703  
if num_layers == 2:
    HIDDEN_DIM = [256, 6]  
    NUM_NEIGHBOURS_LIST = [10, 10]  
else:
    HIDDEN_DIM = [256, 128, 6]  
    NUM_NEIGHBOURS_LIST = [10, 5, 5]  

assert len(HIDDEN_DIM) == len(NUM_NEIGHBOURS_LIST)
BATCH_SIZE = 16  
EPOCHS = 10
NUM_BATCH_PER_EPOCH = 20  
LEARNING_RATE = 0.1 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Planetoid(root="CiteSeer", name= "CiteSeer")

data = dataset.data
print(data)
x = data.x

out = pickle.load(open("./CiteSeer/CiteSeer/raw/ind.citeseer.graph", "rb"), encoding="latin1") 
graph = out.toarray() if hasattr(out, "toarray") else out 

train_index = np.where(data.train_mask)[0]
train_label = data.y
val_index = np.where(data.val_mask)[0]
test_index = np.where(data.test_mask)[0]

model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_neighbours_list=NUM_NEIGHBOURS_LIST, aggregation_func=str(sys.argv[2])).to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

def train():
    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []

    model.train()  
    for e in range(EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        if e % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1

        for batch in range(NUM_BATCH_PER_EPOCH):  
            batch_src_index = np.random.choice(train_index, size=(BATCH_SIZE,))
            batch_src_label = (train_label[batch_src_index]).long().to(DEVICE)
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBOURS_LIST, graph)
            batch_sampling_x = [(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]

            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            _, pred = torch.max(batch_train_logits, dim=1)
            correct = (pred == batch_src_label).sum().item()
            acc = correct / BATCH_SIZE
            train_acc += acc

            validate_loss, validate_acc = validate()
            val_loss += validate_loss
            val_acc += validate_acc

            print(
                "Epoch {:03d} Batch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}".format
                (e, batch, loss.item(), acc, validate_loss, validate_acc))

        train_losses.append(train_loss / NUM_BATCH_PER_EPOCH)
        train_acces.append(train_acc / NUM_BATCH_PER_EPOCH)
        val_losses.append(val_loss / NUM_BATCH_PER_EPOCH)
        val_acces.append(val_acc / NUM_BATCH_PER_EPOCH)

        test()

def validate():
    model.eval() 
    with torch.no_grad(): 
        val_sampling_result = multihop_sampling(val_index, NUM_NEIGHBOURS_LIST, graph)
        val_x = [(x[idx]).float().to(DEVICE) for idx in val_sampling_result]
        val_logits = model(val_x)
        val_label = (data.y[val_index]).long().to(DEVICE)
        loss = criterion(val_logits, val_label)
        predict_y = val_logits.max(1)[1]
        accuracy = torch.eq(predict_y, val_label).float().mean().item()

        return loss.item(), accuracy

def test():
    model.eval()  
    with torch.no_grad():  
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBOURS_LIST, graph)
        test_x = [(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = (data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuracy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuracy)

train()