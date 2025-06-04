
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, TopKPooling, SAGPooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, ModuleList

from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
    DirGNNConv,
    GINConv,
)

from .layers.Graph_Encoder import Graph_Encoder, Graph_Encoder_Norm, Graph_Encoder_Norm_Pooling
from layers.Graph_Decoder import Graph_Decoder_Norm_Unpooling
from layers.GNN_layers import Unpool, Pool

class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, act, dropout):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = 0.5
        heads = 1
        edge_dim = 3

        # Initialize GATConv layers with edge_dim parameter
        self.conv_src_to_dst = GATConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim,  # Specify edge feature dimensionality,
            dropout=dropout,
        )
        self.conv_dst_to_src = GATConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim  # Specify edge feature dimensionality
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr):
        # Transpose edge_index for reverse direction
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0).to(x.device)

        # Apply GATConv layers with edge attributes
        out_src_to_dst = self.conv_src_to_dst(x, edge_index, edge_attr)
        out_dst_to_src = self.conv_dst_to_src(x, edge_index_t, edge_attr)

        # Combine outputs from both directions
        return (1 - self.alpha) * out_src_to_dst + self.alpha * out_dst_to_src

class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h

from torch_geometric.utils import degree

def norm_g(g, num_nodes):
    # Normalize adjacency matrix g based on node degrees
    # g is assumed to be the adjacency matrix in COO format (edge_index)
    row, col = g
    degrees = degree(row, num_nodes=num_nodes, dtype=torch.float)
    degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
    # Normalize the adjacency matrix by degree
    normed_g = g / degrees[row].unsqueeze(1)
    return normed_g

class Decoder(torch.nn.Module):
    '''
    gcn-based decoder for graph data with torch_geometric support
    '''
    def __init__(self, ks, dim, act, drop_p) -> None:
        super(Decoder, self).__init__()
        self.inp_LNs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.LNs = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.inp_LNs.append(nn.LayerNorm(dim))
            self.unpools.append(Unpool())
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.LNs.append(nn.LayerNorm(dim))

        self.out_ln = nn.LayerNorm(dim)

    def forward(self, h, ori_h, down_outs, adj_ms, indices_list, num_nodes):
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]

            # Normalize adjacency matrix for each graph in batch
            g = norm_g(g, num_nodes)  # normalize each graph's adjacency

            # Perform unpooling
            g, h = self.unpools[i](g, h, idx)

            # Layer normalization and Graph Convolution
            h1 = self.inp_LNs[i](down_outs[up_idx] + h)
            g = norm_g(g, num_nodes)  # Re-normalize if needed
            h = self.up_gcns[i](g, h1)
            h = self.LNs[i](h + h1)
        
        h = self.out_ln(h + ori_h)
        return h

class GraphUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, pool_ratios=0.5, act=F.relu, dropout=0.0):
        super(GraphUNet, self).__init__()

        self.depth = depth
        GCN = GCN

        if isinstance(pool_ratios, float):
            pool_ratios = [pool_ratios] * depth

        # Initial GCN layer
        self.gcn_in = GCN(in_dim, hidden_dim, act, dropout)

        # Downsampling layers
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_pool_indices = []

        cur_dim = hidden_dim
        for i in range(depth):
            self.pools.append(Pool(pool_ratios[i], cur_dim, dropout))
            self.down_gcns.append(GCN(cur_dim, hidden_dim, act, dropout))
            cur_dim = hidden_dim

        # Bottleneck GCN
        self.bn_gcn = GCN(cur_dim, hidden_dim, act, dropout)

        # Upsampling layers
        self.up_gcns = nn.ModuleList()
        self.unpools = nn.ModuleList()

        for i in range(depth):
            self.unpools.append(Unpool())
            self.up_gcns.append(GCN(hidden_dim, hidden_dim, act, dropout))

        # Output GCN
        self.gcn_out = GCN(hidden_dim, out_dim, act, dropout)

        self.reduce1 = nn.Linear(hidden_dim, hidden_dim)
        self.reduce2 = nn.Linear(hidden_dim, hidden_dim)