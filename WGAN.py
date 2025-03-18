import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.conv1 = GCNConv(input_dim, 2 * output_dim)
        self.conv2 = GCNConv(2 * output_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        return self.conv2(x, edge_index)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x