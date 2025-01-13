import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random

class InductiveGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InductiveGCN, self).__init__()
        hidden_dim = 64  # 可以根据经验或需求调整隐藏层维度
        num_layers = 2  # 固定层数为2，也可以根据实际情况调整
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_channels))
        self.neighbor_sampling_sizes = [100, 50]  # 可以根据数据集大小和计算资源调整采样大小

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            sampled_edge_index = self.sample_neighbors(edge_index, self.neighbor_sampling_sizes[i])
            x = conv(x, sampled_edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.normalize(x, p=2, dim=-1)
        return x

    def sample_neighbors(self, edge_index, num_samples):
        num_nodes = edge_index.max().item() + 1
        sampled_edge_index = torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        for node in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == node]
            if len(neighbors) > num_samples:
                sampled_neighbors = random.sample(neighbors.tolist(), num_samples)
            else:
                sampled_neighbors = neighbors.tolist()
            new_edge_index = torch.tensor([[node] * len(sampled_neighbors), sampled_neighbors], dtype=torch.long, device=edge_index.device)
            sampled_edge_index = torch.cat([sampled_edge_index, new_edge_index], dim=1)
        return sampled_edge_index