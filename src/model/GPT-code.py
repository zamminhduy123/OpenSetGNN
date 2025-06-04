

from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, TopKPooling, SAGPooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, ModuleList

from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
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
            edge_dim=edge_dim  # Specify edge feature dimensionality
        )
        self.conv_dst_to_src = GATConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            edge_dim=edge_dim  # Specify edge feature dimensionality
        )

    def forward(self, x, edge_index, edge_attr):
        # Transpose edge_index for reverse direction
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0).to(x.device)

        # Apply GATConv layers with edge attributes
        out_src_to_dst = self.conv_src_to_dst(x, edge_index, edge_attr)
        out_dst_to_src = self.conv_dst_to_src(x, edge_index_t, edge_attr)

        # Combine outputs from both directions
        return (1 - self.alpha) * out_src_to_dst + self.alpha * out_dst_to_src

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
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_size * num_heads if (layer_type == "gat") else embedding_size)])
        # Pooling layers
        self.poolings = nn.ModuleList([TopKPooling(embedding_size * num_heads if (layer_type == "gat") else embedding_size, ratio=0.8)])

        emb_dim = embedding_size * num_heads if (layer_type == "gat") else embedding_size
        for _ in range(num_layers - 1):
            self.convs.append(GCNConvType(emb_dim, emb_dim, heads=num_heads, alpha=0.5))
            self.norms.append(nn.LayerNorm(emb_dim))
            self.poolings.append(TopKPooling(emb_dim, ratio=0.8))
            emb_dim = emb_dim * num_heads if (layer_type == "gat") else emb_dim

        self.activate_function = nn.ELU() if (activate == "elu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.GELU() if (activate == "gelu") else nn.ReLU()


        
        self.directed = directed

    def forward(self, x, edge_index, edge_attribute=None, batch=None, acummulate = False):
        self.input = self.input.to(x.device)
        self.activate_function = self.activate_function.to(x.device)
        self.convs = self.convs.to(x.device)
        
        x = self.input(x)
        x = self.activate_function(x)

        # encode feature matrix
        layer_data = []
        new_edge_index, new_edge_attr, new_batch = edge_index, edge_attribute, batch
        for i, (conv, pool) in enumerate(zip(self.convs, self.poolings)):
            if (self.layer_type == "gat"):
                x = conv(x, new_edge_index, new_edge_attr)
            else:
                x = conv(x, new_edge_index)

            # Normalize each layer output
            x = self.norms[i](x)

            x = self.activate_function(x)
            x = F.dropout(x, p=0.3, training=self.training)

            # Apply pooling (down-sampling)
            x, new_edge_index, new_edge_attr, new_batch, _, _ = pool(x, new_edge_index, batch=new_batch)  # 
            layer_data.append((x, new_edge_index, new_edge_attr, new_batch))

        layer_data = torch.stack(layer_data, dim=1)

        return x, x, layer_data, new_edge_index, new_batch
    
class Unpool(nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    def forward(self, data, idx, orig_num_nodes):
        """
        Unpool a graph to restore its original size using indices from pooling.

        Args:
            data (Data): PyTorch Geometric Data object with x (node features), edge_index, batch.
            idx (torch.Tensor): Indices of nodes kept during pooling (from top_k_graph).
            orig_num_nodes (int or torch.Tensor): Original number of nodes per graph (scalar or per graph in batch).

        Returns:
            Data: Updated Data object with unpooled node features and original edge_index.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Handle batched graphs
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initialize new feature matrix with zeros
        max_nodes = orig_num_nodes.max() if torch.is_tensor(orig_num_nodes) else orig_num_nodes
        new_x = torch.zeros((batch.size(0), x.size(1)), dtype=x.dtype, device=x.device)

        # Map pooled features back to original positions
        batch_offsets = torch.zeros(batch.max() + 1, dtype=torch.long, device=x.device)
        if batch.max() > 0:
            batch_offsets[1:] = torch.cumsum(orig_num_nodes, dim=0)[:-1]

        # Adjust indices for batched graphs
        global_idx = idx + batch_offsets[batch[idx]]

        # Place pooled features at correct indices
        new_x[global_idx] = x

        # Return updated Data object (edge_index remains unchanged or can be recomputed)
        return Data(x=new_x, edge_index=edge_index, batch=batch)
    
class Graph_Decoder_Norm(nn.Module):
    def __init__(self, embedding_size, num_features, layer_type="gat", num_layers=4, directed=False, num_heads=1):
        super(Graph_Decoder_Norm, self).__init__()
        
        self.layer_type = layer_type
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            if not directed:
                return GCNConv(input_dim, output_dim) if layer_type == "gcn" else SAGEConv(input_dim, output_dim) if layer_type == "sage" else GATConv(input_dim, output_dim, heads=heads, edge_dim=1, add_self_loops=True)
            # Add directed versions if needed (e.g., DirSageConv, DirGATConv)
            return GCNConv(input_dim, output_dim) if layer_type == "gcn" else SAGEConv(input_dim, output_dim) if layer_type == "sage" else GATConv(input_dim, output_dim, heads=heads)
        
        self.convs = ModuleList()
        # First layer takes encoder output
        self.convs.append(GCNConvType(embedding_size, embedding_size, heads=num_heads, alpha=0.5))
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_size)])

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvType(embedding_size * num_heads, embedding_size, heads=num_heads, alpha=0.5))
            self.norms.append(nn.LayerNorm(embedding_size * num_heads if (layer_type == "gat") else embedding_size))
        
        self.activate_function = nn.GELU()

    def forward(self, z, layer_data):
        for i, conv in enumerate(self.convs):
            if self.layer_type == "gat":
                x = conv(x, edge_index, edge_attribute)
            else:
                x = conv(x, edge_index)
            x = self.norms[i](x)
            x = self.activate_function(x)
            x = F.dropout(x, p=0.1)
        
        return x