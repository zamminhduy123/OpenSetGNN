
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
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

from .layers.Graph_Encoder import Graph_Encoder
    
class GAE_Projection_2(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4):
        super().__init__()

        self.layer_type = layer_type
        self.encoder = Graph_Encoder(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_features)
        )

        # Projection head
        self.projection_head = nn.Sequential(
            Linear(embedding_size*2, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.1),
            Linear(embedding_size, projection_emb)
        )

    def adj_decode(self, x, batch_index):

        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]  # Node embeddings for this graph
            
            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.sigmoid(torch.mm(graph_emb, graph_emb.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    
    def forward(self, data, get_pooling = False):    
        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = True)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = True)

        # global pooling
        global_pooling = torch.cat([gmp(x, data.batch), gap(x, data.batch)], dim=1)

        reconstructed_X = self.decoder(h)
        adj_reconstructed = self.adj_decode(x, data.batch)

        if (get_pooling):
            return global_pooling, reconstructed_X, adj_reconstructed
 
        return self.projection_head(global_pooling), reconstructed_X, adj_reconstructed