import networkx as nx
import torch
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

class GNN(torch.nn.Module):

    def __init__(self, input_features):
        super().__init__()
        
        self.conv_1 = GraphConvolutionLayer(input_features, 32)
        self.conv_2 = GraphConvolutionLayer(32, 32)
        self.fc_1 = torch.nn.Linear(32, 16)
        self.output_layer = torch.nn.Linear(16, 2)
    
    def forward(self, X, A,batch_mat):
        x = self.conv_1(X, A).clamp(0)
        x = self.conv_2(x, A).clamp(0)
        output = global_sum_pool(x)
        output = self.fc_1(output)
        output = self.output_layer(output)
        return F.softmax(output, dim=1)

class GraphConvolutionLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.W1 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.W2 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32)) 
        self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float32))
    
    def forward(self, X, A):
        potential_messages = torch.mm(X, self.W2)
        propagated_messages = torch.mm(A, potential_messages)
        root_update = torch.mm(X, self.W1)
        output = propagated_messages + root_update + self.bias
        return output

def global_sum_pool(X):
    return torch.sum(X, dim=0).unsqueeze(0)