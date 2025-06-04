from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, ModuleList

from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    GATv2Conv,
    JumpingKnowledge,
    DirGNNConv,
    GINConv,
)

class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha, edge_dim):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        # Initialize GATConv layers with edge_dim parameter
        self.conv_src_to_dst = GATConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim,  # Specify edge feature dimensionality
            add_self_loops=True
        )
        self.conv_dst_to_src = GATConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim,  # Specify edge feature dimensionality
            add_self_loops=True
        )

    def forward(self, x, edge_index, edge_attr):
        # Transpose edge_index for reverse direction
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0).to(x.device)

        # Apply GATConv layers with edge attributes
        out_src_to_dst = self.conv_src_to_dst(x, edge_index, edge_attr)
        out_dst_to_src = self.conv_dst_to_src(x, edge_index_t, edge_attr)

        # Combine outputs from both directions
        return (1 - self.alpha) * out_src_to_dst + self.alpha * out_dst_to_src

class DirGATv2Conv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha, edge_dim, get_att_weights=False):
        super(DirGATv2Conv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        # Initialize GATConv layers with edge_dim parameter
        self.conv_src_to_dst = GATv2Conv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim,  # Specify edge feature dimensionality
            add_self_loops=False
        )
        self.conv_dst_to_src = GATv2Conv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim,  # Specify edge feature dimensionality
            add_self_loops=False
        )

    def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
        # Transpose edge_index for reverse direction
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0).to(x.device)

        # Apply GATConv layers with edge attributes
        out_src_to_dst, () = self.conv_src_to_dst(x, edge_index, edge_attr, return_attention_weights=return_attention_weights)
        out_dst_to_src = self.conv_dst_to_src(x, edge_index_t, edge_attr, return_attention_weights=return_attention_weights)

        # Combine outputs from both directions
        return (1 - self.alpha) * out_src_to_dst + self.alpha * out_dst_to_src
    
class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h, edge_index):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        
        # Perform top_k graph pooling
        return top_k_graph(scores, g, h, edge_index, self.k)

def top_k_graph(scores, g, h, edge_index, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)

    # Use the edge_index format and select edges corresponding to the top-k nodes
    edge_mask = torch.isin(edge_index[0], idx) & torch.isin(edge_index[1], idx)
    edge_index = edge_index[:, edge_mask]

    return edge_index, new_h, idx


class Unpool(nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    def forward(self, g, h, idx, edge_index):
        # Create a zero tensor to store unpooled features
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h

        # Create the unpooled edge index
        edge_index = self.expand_edge_index(edge_index, idx, g.shape[0])

        return edge_index, new_h

    def expand_edge_index(self, edge_index, idx, num_nodes):
        # Create a mapping from the pooled indices back to the original graph
        unpooled_edge_index = torch.zeros_like(edge_index)
        unpooled_edge_index[0] = torch.index_select(idx, 0, edge_index[0])
        unpooled_edge_index[1] = torch.index_select(idx, 0, edge_index[1])

        # Reorganize the edge_index for the unpooled graph
        return unpooled_edge_index