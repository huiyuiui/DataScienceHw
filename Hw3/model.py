import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from torch_geometric.nn import GCNConv

# GCN
class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
          
        return h

# GAT
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_size, hid_size, num_heads=num_heads, activation=F.elu)
        self.layer2 = GATConv(hid_size * num_heads, out_size, num_heads=1, activation=None)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        h = self.layer1(g, h)
        h = self.dropout(h)
        h = h.reshape(h.shape[0], -1)  # Flatten multiple heads
        h = self.layer2(g, h)
        h = h.reshape(h.shape[0], h.shape[2])

        return h
    
def drop_edge(graph, drop_prob):
    E = graph.number_of_edges()
    keep_mask = torch.rand(E) > drop_prob
    src, dst = graph.edges()
    edge_index = torch.stack([src[keep_mask], dst[keep_mask]], dim=0)

    return edge_index

    
# SSP
class CRD(nn.Module):
    def __init__(self, in_size, out_size, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(in_size, out_size * 2, cached=True) 
        self.conv2 = GCNConv(out_size * 2, out_size, cached=True)
        self.conv3 = GCNConv(out_size * 2, out_size, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.dropout(x, p=self.p, training=self.training)

        return x
    
class CLS(nn.Module):
    def __init__(self, in_size, out_size):
        super(CLS, self).__init__()
        self.conv = GCNConv(in_size, out_size, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x
    
class SSP(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size, drop_out):
        super(SSP, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)
        self.p = drop_out

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, x):
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.log_softmax(x, dim=1)

        return x
