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
from torch_geometric.nn.norm import GraphNorm

from .GNN_layers import DirSageConv, DirGATConv, DirGATv2Conv
from torch_geometric.nn import TopKPooling

class Graph_Encoder(nn.Module):
    def __init__(self, num_features, embedding_size=64, layer_type="gat", num_layers=4, activate = "leakyReLU", directed=False, num_heads=1, edge_dim=0, remove_random=True):
        # Init parent
        super(Graph_Encoder, self).__init__()

        self.bn = torch.nn.BatchNorm1d(num_features)
        self.input = torch.nn.Linear(num_features, embedding_size)

        self.layer_type = layer_type
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            if (not directed):
                return GCNConv(input_dim, output_dim) if (layer_type == "gcn") else SAGEConv(input_dim, output_dim) if (layer_type == "sage") else GATConv(input_dim, output_dim, heads=heads, edge_dim=edge_dim, add_self_loops=False)
            return GCNConv(input_dim, output_dim) if (layer_type == "gcn") else DirSageConv(input_dim, output_dim, alpha) if (layer_type == "sage") else DirGATConv(input_dim, output_dim, heads, alpha, edge_dim=edge_dim)
        
        self.convs = ModuleList([GCNConvType(embedding_size, embedding_size, heads=num_heads, alpha=0.5)])

        for _ in range(num_layers - 1):
            self.convs.append(GCNConvType(embedding_size * num_heads if ("gat" in layer_type) else embedding_size, embedding_size, heads=num_heads, alpha=0.5))

        self.activate_function = nn.ELU() if (activate == "elu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ReLU()

        self.directed = directed
        self.remove_random = remove_random

    def forward(self, x, edge_index, edge_attribute=None, acummulate = False):

        self.bn = self.bn.to(x.device)
        self.input = self.input.to(x.device)
        self.activate_function = self.activate_function.to(x.device)
        self.convs = self.convs.to(x.device)
        
        # encode feature matrix
        if (not self.remove_random):
            x = self.bn(x)
        x = self.input(x)
        x = self.activate_function(x)

        # encode feature matrix
        xs = []
        h = None
        for i, conv in enumerate(self.convs):
            if ("gat" in self.layer_type):
                x = conv(x, edge_index, edge_attribute)
            else:
                x = conv(x, edge_index)

            x = self.activate_function(x)
            x = F.dropout(x, p=0.3, training=self.training)

            layer_flatten = x.view(-1)
            if i >= 1:
                layer_flatten = F.pad(layer_flatten, (0, xs[0].size(0) - layer_flatten.size(0)))
            xs.append(layer_flatten)

            if (h is None):
                h = x * 0.5
            else:
                if (x.size(1) > h.size(1)):
                    h = F.pad(h, (0, x.size(1) - h.size(1)))
                h += x * 0.5

        xs = torch.stack(xs, dim=1)

        if (acummulate):
            return x, h, xs
        return x, x, xs
    

class Graph_Encoder_Norm(nn.Module):
    def __init__(self, num_features, embedding_size=64, layer_type="gat", num_layers=4, activate = "leakyReLU", directed=False, num_heads=1, edge_dim=0, get_att_weights=False):
        # Init parent
        super(Graph_Encoder_Norm, self).__init__()

        self.bn = torch.nn.BatchNorm1d(num_features) 
        # self.bn = nn.LayerNorm(num_features)
        # self.bn = torch.nn.GroupNorm(1, num_features)
        # self.input = torch.nn.Linear(num_features, embedding_size)

        self.layer_type = layer_type
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            if (not directed):
                return GCNConv(input_dim, output_dim) if (layer_type == "gcn") else SAGEConv(input_dim, output_dim) if (layer_type == "sage") else GATConv(input_dim, output_dim, heads=heads, edge_dim=edge_dim, add_self_loops=False) if (layer_type == "gat") else GATv2Conv(input_dim, output_dim, heads=heads, edge_dim=edge_dim, add_self_loops=False)
            return GCNConv(input_dim, output_dim) if (layer_type == "gcn") else DirSageConv(input_dim, output_dim, alpha) if (layer_type == "sage") else DirGATConv(input_dim, output_dim, heads, alpha, edge_dim=edge_dim) if ('gat' in layer_type) else DirGATv2Conv(input_dim, output_dim, heads, alpha, edge_dim=edge_dim)
        
        self.convs = ModuleList([GCNConvType(num_features, embedding_size, heads=num_heads, alpha=0.5)])
        # self.norms = nn.ModuleList([nn.LayerNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size)])

        self.norms = nn.ModuleList([GraphNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size)])

        for _ in range(num_layers - 1):
            self.convs.append(GCNConvType(embedding_size * num_heads if ("gat" in layer_type) else embedding_size, embedding_size, heads=num_heads, alpha=0.5))
            self.norms.append(GraphNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size))

        self.activate_function = nn.ELU() if (activate == "elu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.GELU() if (activate == "gelu") else nn.ReLU()
        

        self.directed = directed

    def forward(self, x, edge_index, edge_attribute=None, acummulate = False, remove_random=False, return_one = False, batch=None):
        # self.bn = self.bn.to(x.device)
        # self.input = self.input.to(x.device)
        self.activate_function = self.activate_function.to(x.device)
        self.convs = self.convs.to(x.device)
        self.norms = self.norms.to(x.device)
        
        # encode feature matrix
        if (not remove_random):
            x = self.bn(x)
        # x = self.input(x)
        # x = self.activate_function(x)

        # encode feature matrix
        xs = []
        h = None
        for i, conv in enumerate(self.convs):
            if ("gat" in self.layer_type):
                x = conv(x, edge_index, edge_attribute)
            else:
                x = conv(x, edge_index)

            # Normalize each layer output
            x = self.norms[i](x, batch=batch)

            x = self.activate_function(x)
            x = F.dropout(x, p=0.3, training=self.training)

            layer_flatten = x.view(-1)
            if i >= 1:
                layer_flatten = F.pad(layer_flatten, (0, xs[0].size(0) - layer_flatten.size(0)))
            xs.append(layer_flatten)

            if (h is None):
                h = x * 0.5
            else:
                if (x.size(1) > h.size(1)):
                    h = F.pad(h, (0, x.size(1) - h.size(1)))
                h += x * 0.5

        xs = torch.stack(xs, dim=1)


        if (acummulate):
            if (return_one):
                return h
            else:
                return x, h, xs
        if (return_one):
            return x
        return x, x, xs

class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleAutoEncoder, self).__init__()
        
        # Encoder: 2-layer MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            # nn.ReLU()
        )
        
        # Decoder: 2-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()  # or nn.Identity() if input is not normalized
        )
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x):
        return self.decode(self.encode(x))

class Graph_Encoder_Norm_Pooling(nn.Module):
    def __init__(self, num_features, embedding_size=64, layer_type="gat", num_layers=4, activate = "leakyReLU", directed=False, num_heads=1, edge_dim=0):
        # Init parent
        super(Graph_Encoder_Norm_Pooling, self).__init__()

        self.input = torch.nn.Linear(num_features, embedding_size)

        self.layer_type = layer_type
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            if (not directed):
                return GCNConv(input_dim, output_dim) if (layer_type == "gcn") else SAGEConv(input_dim, output_dim) if (layer_type == "sage") else GATConv(input_dim, output_dim, heads=heads, edge_dim=edge_dim, add_self_loops=False)
            return GCNConv(input_dim, output_dim) if (layer_type == "gcn") else DirSageConv(input_dim, output_dim, alpha) if (layer_type == "sage") else DirGATConv(input_dim, output_dim, heads, alpha, edge_dim=edge_dim)
        
        self.convs = ModuleList([GCNConvType(embedding_size, embedding_size, heads=num_heads, alpha=0.5)])
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size)])
        # Pooling layers
        # self.poolings = nn.ModuleList([TopKPooling(embedding_size * num_heads if ("gat" in layer_type) else embedding_size, ratio=0.1)])

        emb_dim = embedding_size * num_heads if ("gat" in layer_type) else embedding_size
        for _ in range(num_layers - 1):
            self.convs.append(GCNConvType(emb_dim, emb_dim, heads=num_heads, alpha=0.5))
            self.norms.append(nn.LayerNorm(emb_dim))
            # self.poolings.append(TopKPooling(emb_dim, ratio=0.1))
            emb_dim = emb_dim * num_heads if ("gat" in layer_type) else emb_dim

        self.activate_function = nn.ELU() if (activate == "elu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.GELU() if (activate == "gelu") else nn.ReLU()


        
        self.directed = directed

    def forward(self, x, edge_index, edge_attribute=None, batch=None, acummulate = False):
        self.input = self.input.to(x.device)
        self.activate_function = self.activate_function.to(x.device)
        self.convs = self.convs.to(x.device)
        
        x = self.input(x)
        x = self.activate_function(x)

        # encode feature matrix
        layer_data = [(x, edge_index, edge_attribute, batch, None, x.shape[0])]
        total_node = x.size(0)
        new_edge_index, new_edge_attr, new_batch = edge_index, edge_attribute, batch
        for i, (conv, pool) in enumerate(zip(self.convs, self.poolings)):
            if ("gat" in self.layer_type):
                x = conv(x, new_edge_index, new_edge_attr)
            else:
                x = conv(x, new_edge_index)

            # Normalize each layer output
            x = self.norms[i](x)

            x = self.activate_function(x)
            x = F.dropout(x, p=0.3, training=self.training)

            # Apply pooling (down-sampling)
            x, new_edge_index, new_edge_attr, new_batch, perm, _ = pool(x, new_edge_index, batch=new_batch)  # 
            layer_data.append((x, new_edge_index, new_edge_attr, new_batch, perm, total_node))
            total_node = x.size(0)

        return x, x, layer_data, new_edge_index, new_batch
    