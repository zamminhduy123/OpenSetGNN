from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, ModuleList
from torch_geometric.data import Data

from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
    DirGNNConv,
    GINConv,
)

from .GNN_layers import DirSageConv, DirGATConv
from torch_geometric.nn import TopKPooling

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

class Unpooling(nn.Module):
    def __init__(self):
        super(Unpooling, self).__init__()

    def forward(self, x, idx):
        """
        Unpooling layer: Restore the embeddings of pooled nodes back to the full graph.
        Args:
            x (Tensor): Pooled node embeddings of shape [num_pooled_nodes, embed_dim]
            idx (Tensor): Indices of the pooled nodes from the encoder (shape: [num_pooled_nodes])
        
        Returns:
            new_h (Tensor): Unpooled node embeddings of shape [original_num_nodes, embed_dim]
        """
        new_h = torch.zeros(x.size(0), x.size(1), device=x.device)
        new_h[idx] = x  # Place the pooled embeddings back at the specified indices
        return new_h 


class Graph_Decoder_Norm_Unpooling_2(nn.Module):
    def __init__(self, embedding_size, num_features, layer_type="gat", num_layers=4, directed=False, num_heads=1):
        super(Graph_Decoder_Norm_Unpooling_2, self).__init__()
        
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
        for _ in range(num_layers - 1):
            self.convs.append(GCNConvType(embedding_size * num_heads, embedding_size, heads=num_heads, alpha=0.5))
            self.norms.append(nn.LayerNorm(embedding_size * num_heads if (layer_type == "gat") else embedding_size))
        
        self.activate_function = nn.GELU()
        self.output = Linear(embedding_size, num_features)  # Final layer to map to original feature space

    def forward(self, z, layer_data):
        """
        Forward pass for graph decoder with correct unpooling back to original graph size.
        
        Args:
            z: Latent representation from encoder
            layer_data: List containing pooled data from each encoder layer
                    (x, edge_index, edge_attr, batch, perm)
            
        Returns:
            x: Reconstructed node features
        """
        x = z  # Start with the latent representation
        current_edge_index = None
        current_edge_attr = None
        
        # Process layers in reverse order (from deepest to shallowest)
        for i in range(len(self.convs)):
            # Get the corresponding encoder layer index (in reverse)
            enc_idx = len(layer_data) - 1 - i
            
            if enc_idx >= 1:
                # Get data from the current encoder layer
                curr_x, curr_edge_index, curr_edge_attr, curr_batch, curr_perm, prev_node = layer_data[enc_idx]
                prev_x, prev_edge_index, prev_edge_attr, _, _, _ = layer_data[enc_idx-1]
                target_size = prev_node

                # Unpool the current layer's output
                unpooled_x = torch.zeros(
                    target_size,
                    x.size(1),
                    dtype=x.dtype,
                    device=x.device
                )
                
                # Place the pooled node features back using the permutation indices
                unpooled_x[curr_perm] = x
                  
                # Apply convolution (before unpooling)
                if self.layer_type == "gat":
                    x = self.convs[i](unpooled_x, prev_edge_index, prev_edge_attr)
                else:
                    x = self.convs[i](unpooled_x, prev_edge_index)
                
                # Apply normalization, activation, and dropout
                if i < len(self.norms):
                    x = self.norms[i](x)
                x = self.activate_function(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x

class Graph_Decoder_Norm_Unpooling_1(nn.Module):
    def __init__(self, num_features, embedding_size=64, layer_type="gat", num_layers=4, activate="leakyReLU", directed=False, num_heads=1, edge_dim=0):
        super(Graph_Decoder_Norm_Unpooling, self).__init__()

        self.layer_type = layer_type
        
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            if not directed:
                return GCNConv(input_dim, output_dim) if layer_type == "gcn" else SAGEConv(input_dim, output_dim) if layer_type == "sage" else GATConv(input_dim, output_dim, heads=heads, edge_dim=edge_dim, add_self_loops=False)
            return GCNConv(input_dim, output_dim) if layer_type == "gcn" else DirSageConv(input_dim, output_dim, alpha) if layer_type == "sage" else DirGATConv(input_dim, output_dim, heads, alpha, edge_dim=edge_dim)

        self.convs = nn.ModuleList([])
        self.norms = nn.ModuleList([])

        # Unpooling layers
        self.unpoolings = nn.ModuleList([Unpool()])
        for _ in range(num_layers):
            self.convs.append(GCNConvType(embedding_size * num_heads if layer_type == "gat" else embedding_size, embedding_size, heads=num_heads, alpha=0.5))
            self.norms.append(nn.LayerNorm(embedding_size * num_heads if layer_type == "gat" else embedding_size))
            self.unpoolings.append(Unpool())

        # Activation function
        self.activate_function = nn.ELU() if activate == "elu" else nn.LeakyReLU() if activate == "leakyReLU" else nn.GELU() if activate == "gelu" else nn.ReLU()

    def forward(self, x, edge_index, edge_attribute=None, batch=None):
        xs = []  # To store intermediate outputs
        h = None  # To store the feature output of the last layer

        # Loop over layers
        for i, (conv, unpool) in enumerate(zip(self.convs, self.unpoolings)):
            # Apply unpooling (reversing pooling)
            x = unpool(x, edge_index)
            
            # Apply GCN layer (same as encoder)
            if self.layer_type == "gat":
                x = conv(x, edge_index, edge_attribute)
            else:
                x = conv(x, edge_index)

            # Normalize each layer output
            x = self.norms[i](x)
            x = self.activate_function(x)
            x = F.dropout(x, p=0.3, training=self.training)

            # Store intermediate output
            xs.append(x)

            # If this is the first layer, initialize `h`
            if h is None:
                h = x * 0.5
            else:
                if x.size(1) > h.size(1):
                    h = F.pad(h, (0, x.size(1) - h.size(1)))
                h += x * 0.5

        # Final output: concatenate the intermediate outputs
        xs = torch.stack(xs, dim=1)
        return h  # Returning the final node features after reconstruction