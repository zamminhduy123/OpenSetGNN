
from torch_geometric.nn import global_mean_pool, global_add_pool as gap, TopKPooling, SAGPooling, global_max_pool as gMp, Set2Set
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
from torch_geometric.utils import to_dense_batch
from .layers.Graph_Encoder import Graph_Encoder, Graph_Encoder_Norm, Graph_Encoder_Norm_Pooling, SimpleAutoEncoder
from .layers.Graph_Decoder import Graph_Decoder_Norm_Unpooling_2
from .layers.GNN_layers import DirSageConv, DirGATConv, Unpool, Pool
from torch_geometric.nn.norm import GraphNorm

from utils.infomax import global_global_loss_, local_global_loss_
class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_features, num_heads=2):
        super().__init__()
        # Simple multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_features)
        )

    def forward(self, x):
        # x shape: [num_nodes, embed_dim] => for MHA we might need [num_nodes, batch, embed_dim]
        x_t = x.unsqueeze(1)  # [num_nodes, 1, embed_dim]
        # Self-attention
        attn_output, _ = self.attn(x_t, x_t, x_t)
        attn_output = attn_output.squeeze(1)  # back to [num_nodes, embed_dim]
        return self.mlp(attn_output)
    
class MLPDecoder(nn.Module):
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_features)
        )

    def forward(self, x):
        return self.mlp(x)
    
class EdgeDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()  # optional if you want [0,1] scores
        )

    def forward(self, node_embeddings):
        """
        node_embeddings: [N, D] for a single graph
        returns: [N, N] reconstructed adjacency matrix
        """
        N = node_embeddings.size(0)
        i_idx, j_idx = torch.meshgrid(
            torch.arange(N, device=node_embeddings.device),
            torch.arange(N, device=node_embeddings.device),
            indexing='ij'
        )
        i_idx = i_idx.reshape(-1)
        j_idx = j_idx.reshape(-1)

        edge_feat = torch.cat([
            node_embeddings[i_idx],
            node_embeddings[j_idx]
        ], dim=-1)  # [N*N, 2D]

        edge_scores = self.mlp(edge_feat).view(N, N)
        return edge_scores  # [N, N]

class NeighborhoodAwareDecoder(nn.Module):
    def __init__(self, embed_dim, num_features, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.gcn = GCNConv(embed_dim, embed_dim) # Add GCN layer
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_features)
        )

    def forward(self, x, edge_index): #add edge_index as an input
        x_t = x.unsqueeze(1)
        attn_output, _ = self.attn(x_t, x_t, x_t)
        attn_output = attn_output.squeeze(1)
        gcn_output = self.gcn(attn_output, edge_index) #neighborhood aggregation
        return self.mlp(gcn_output)

class Graph_Decoder(nn.Module):
    def __init__(self, embedding_size, num_features, layer_type="gat", num_layers=4, directed=False, num_heads=1):
        super(Graph_Decoder, self).__init__()
        
        self.layer_type = layer_type
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            if not directed:
                return GCNConv(input_dim, output_dim) if layer_type == "gcn" else SAGEConv(input_dim, output_dim) if layer_type == "sage" else GATConv(input_dim, output_dim, heads=heads, edge_dim=1, add_self_loops=True)
            # Add directed versions if needed (e.g., DirSageConv, DirGATConv)
            return GCNConv(input_dim, output_dim) if layer_type == "gcn" else SAGEConv(input_dim, output_dim) if layer_type == "sage" else GATConv(input_dim, output_dim, heads=heads)
        
        self.convs = ModuleList()
        # First layer takes encoder output
        self.convs.append(GCNConvType(embedding_size, embedding_size, heads=num_heads, alpha=0.5))
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvType(embedding_size * num_heads, embedding_size, heads=num_heads, alpha=0.5))
        # Last GNN layer before output
        self.convs.append(GCNConvType(embedding_size * num_heads, embedding_size, heads=num_heads, alpha=0.5))
        
        self.output = Linear(embedding_size, num_features)  # Final reconstruction layer
        self.activate_function = nn.ELU()

    def forward(self, x, edge_index, edge_attribute=None):
        for i, conv in enumerate(self.convs):
            if self.layer_type == "gat":
                x = conv(x, edge_index, edge_attribute)
            else:
                x = conv(x, edge_index)
            x = self.activate_function(x)
            x = F.dropout(x, p=0.1)
        
        x_reconstructed = self.output(x)  # Linear output to match input range
        return x_reconstructed

class GAE_Projection_Att(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4):
        super().__init__()

        self.layer_type = layer_type
        self.encoder = Graph_Encoder(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers)
        self.decoder = AttentionDecoder(embedding_size, num_features)

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

from .layers.Id_Encoder import BinaryEncoder, IdEmbedding

class GAE_Projection_Att_WithIDEncoder(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0):
        super().__init__()

        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)

        self.id_encoder = BinaryEncoder(input_dim=29, embedding_dim=id_dim)
        # self.id_encoder = nn.Embedding(num_embeddings=1000, embedding_dim=id_dim)

        HEADS = 2 

        self.layer_type = layer_type
        self.encoder = Graph_Encoder(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=3)
        self.decoder = AttentionDecoder(embedding_size * HEADS, num_features)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        self.ae = GraphAutoEncoder(input_dim=embedding_size, hidden_dim=int(embedding_size/2), output_dim=embedding_size)

        self.linear = nn.Sequential(
            Linear(embedding_size*2*HEADS, embedding_size),
        )

        # Projection head
        self.projection_head = nn.Sequential(
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.1),
            Linear(embedding_size, projection_emb if (num_classes == 0) else num_classes),
        )

    def adj_decode(self, x, batch_index):

        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]  # Node embeddings for this graph
            
            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.relu(torch.mm(graph_emb, graph_emb.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, get_pooling = False, with_ae = False, acummulate = True):    
        # binary_id = data.binary_id.to(dtype=torch.float32, device=device)
        # dec_id = data.dec_id

        # ids = []
        # for id in dec_id:
        #     ids.extend(id)
        # ids = torch.tensor(ids).to(dtype=torch.int64, device=device)

        # Encode binary ID
        # Assuming data.binary_id has shape (2507, 29)
        # Convert binary to decimal if required
        # tensor_id = torch.matmul(binary_id, 2**torch.arange(29).flip(0).to(device, dtype=torch.float32))

        # tensor_id = tensor_id.to(dtype=torch.int64, device=device)
        # id_embedding = self.id_encoder(tensor_id).to(device)

        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.projection_head = self.projection_head.to(device)
        self.encoder = self.encoder.to(device)
        self.ae = self.ae.to(device)


        # remove first feature from x because it is the id
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        # id_embedding = self.id_encoder(binary_id).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)



        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate)

        # global pooling
        global_pooling = torch.cat([gmp(x, data.batch), gap(x, data.batch)], dim=1)

        linear = self.linear(global_pooling)

        reconstructed_X = self.decoder(h)
        adj_reconstructed = self.adj_decode(x, data.batch)

        if (get_pooling):
            return linear, reconstructed_X, adj_reconstructed
        
        if (with_ae):
            graph_rec = self.ae(linear)
            return self.projection_head(linear), reconstructed_X, adj_reconstructed, global_pooling, graph_rec 
 
        return self.projection_head(linear), reconstructed_X, adj_reconstructed 
    
class GAE_Projection_Att_WithIDEncoder_2(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, remove_random=False):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 2 

        self.layer_type = layer_type
        self.encoder = Graph_Encoder(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, remove_random=remove_random)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        self.ae = GraphAutoEncoder(input_dim=embedding_size, hidden_dim=int(embedding_size/2), output_dim=embedding_size)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
        )

        node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = AttentionDecoder(node_dim, num_features)

    def adj_decode(self, x, batch_index):

        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]  # Node embeddings for this graph
            
            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.relu(torch.mm(graph_emb, graph_emb.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, acummulate = True):    
        # binary_id = data.binary_id.to(dtype=torch.float32, device=device)
        # dec_id = data.dec_id

        # ids = []
        # for id in dec_id:
        #     ids.extend(id)
        # ids = torch.tensor(ids).to(dtype=torch.int64, device=device)

        # Encode binary ID
        # Assuming data.binary_id has shape (2507, 29)
        # Convert binary to decimal if required
        # tensor_id = torch.matmul(binary_id, 2**torch.arange(29).flip(0).to(device, dtype=torch.float32))

        # tensor_id = tensor_id.to(dtype=torch.int64, device=device)
        # id_embedding = self.id_encoder(tensor_id).to(device)

        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.ae = self.ae.to(device)

        if (self.id_dim > 0):
            id_tensor = data.x[:, 0].long()
            data.x = data.x[:, 1:]
            id_embedding = self.id_embedding(id_tensor).to(device)
            data.x = torch.cat([data.x, id_embedding], dim=1)

        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate)

        global_pooling = torch.cat([gmp(x, data.batch), gap(x, data.batch)], dim=1)

        linear = self.linear(global_pooling)

        reconstructed_X = self.decoder(h)
        adj_reconstructed = self.adj_decode(x, data.batch)
 
        return linear, reconstructed_X, adj_reconstructed 
    
class GAE_Projection_Att_WithIDEncoder_3(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        self.ae = GraphAutoEncoder(input_dim=embedding_size, hidden_dim=int(embedding_size/2), output_dim=embedding_size)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.8)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = AttentionDecoder(node_dim, num_features)

    def adj_decode(self, x, batch_index):

        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]  # Node embeddings for this graph
            
            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.relu(torch.mm(graph_emb, graph_emb.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, acummulate = True, remove_random=False):    
        # binary_id = data.binary_id.to(dtype=torch.float32, device=device)
        # dec_id = data.dec_id

        # ids = []
        # for id in dec_id:
        #     ids.extend(id)
        # ids = torch.tensor(ids).to(dtype=torch.int64, device=device)

        # Encode binary ID
        # Assuming data.binary_id has shape (2507, 29)
        # Convert binary to decimal if required
        # tensor_id = torch.matmul(binary_id, 2**torch.arange(29).flip(0).to(device, dtype=torch.float32))

        # tensor_id = tensor_id.to(dtype=torch.int64, device=device)
        # id_embedding = self.id_encoder(tensor_id).to(device)

        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.ae = self.ae.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            id_tensor = data.x[:, 0].long()
            data.x = data.x[:, 1:]
            id_embedding = self.id_embedding(id_tensor).to(device)
            data.x = torch.cat([data.x, id_embedding], dim=1)

        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(x, data.edge_index, batch=data.batch)
        # x2, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)

        linear = self.linear(global_pooling)

        reconstructed_X = self.decoder(h)
        adj_reconstructed = self.adj_decode(x, data.batch)
 
        return linear, reconstructed_X, adj_reconstructed 

class EdgeFrequencyDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output is predicted frequency
        )

    def forward(self, z, edge_index):
        src = z[edge_index[0]]  # shape: [num_edges, hidden_dim]
        dst = z[edge_index[1]]
        edge_feat = torch.cat([src, dst], dim=1)  # [num_edges, 2 * hidden_dim]
        pred_freq = self.mlp(edge_feat)  # [num_edges, 1]
        return pred_freq

class GAE_CLS_EdgeFreq_NODE(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(embedding_size, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = AttentionDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.edge_decoder = EdgeFrequencyDecoder(embedding_size)
        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def edge_freq_decode(self, z, edge_index):
        return self.edge_decoder(z, edge_index)
    
    def adj_decode(self, z, batch_index):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.sigmoid(torch.mm(node_emb_1, node_emb_2.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X
    
    def graph_logits(self, z, edge_index, batch):
        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, get_att_weights=False):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, get_att_weights=get_att_weights)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = AttentionDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def adj_decode(self, z, batch_index):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # 2. Dense embeddings + mask
            _, mask = to_dense_batch(z, batch_index)

            # 3. Predict full matrix
            # adj_pred = torch.einsum('b i d, b j d -> b i j', node_emb_1, node_emb_2)
            # adj_pred = torch.relu(adj_pred)

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.relu(torch.mm(node_emb_1, node_emb_2.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X
    
    def graph_logits(self, z, edge_index, batch):
        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 4 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        # self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, node_dim)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 1):
            data = self.id_encode(data, device)
        elif (self.id_dim == 0):
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_logits(self, z, edge_index, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        max_pooled = manual_global_max_pool(z, batch)
        global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine_SupCon(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        # self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 128),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(128, projection_emb),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, embedding_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)
        else:
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_embedding(self, z, edge_index, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        max_pooled = manual_global_max_pool(z, batch)
        global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine_CAC(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        # self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 128),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(128, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, embedding_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)
        else:
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_embedding(self, z, edge_index, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        max_pooled = manual_global_max_pool(z, batch)
        global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine_SupCon_Edge(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        # self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 128),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(128, projection_emb),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, embedding_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.edge_decoder = EdgeFrequencyDecoder(embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)
        else:
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def edge_decode(self, z, edge_index):
        return self.edge_decoder(z, edge_index)

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_embedding(self, z, edge_index, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        max_pooled = manual_global_max_pool(z, batch)
        global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine_Dual(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        # self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, embedding_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        data.x_ae = data.x_ae[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        data.x_ae = torch.cat([data.x_ae, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)
        else:
            data.x = data.x[:, 1:]
            data.x_ae = data.x_ae[:, 1:]

        if ("gat" in self.layer_type):
            _, x, _ = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
            _, x_ae, _ = self.encoder(data.x_ae, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            _, x, _ = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)
            _, x_ae, _ = self.encoder(data.x_ae, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, x_ae
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_logits(self, z, edge_index, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        max_pooled = manual_global_max_pool(z, batch)
        global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear


class GAE_CLS_Link_NODE_AE(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)

        latent_dim = 16
        self.ae = SimpleAutoEncoder(input_dim=embedding_size, hidden_dim=32, latent_dim=latent_dim)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size

        # self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = latent_dim * 2 * HEADS if ("gat" in layer_type) else latent_dim * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(latent_dim, latent_dim)
            self.node_out_2 = nn.Linear(latent_dim, latent_dim)

        self.node_proj_1 = nn.Linear(num_features, latent_dim)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.ae = self.ae.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)
        else:
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
            x_encoded = self.ae.encode(h)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)
            x_encoded = self.ae.encode(h)

        return x, x_encoded, xs
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        x_decoded = self.ae.decode(z)
        reconstructed_X = self.decoder(x_decoded)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_logits(self, z, edge_index, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        max_pooled = manual_global_max_pool(z, batch)
        global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear


def manual_global_max_pool(x, batch):
    """
    Args:
        x: Tensor of shape [num_nodes, num_features]
        batch: Tensor of shape [num_nodes], with graph index for each node
    Returns:
        max_pooled: Tensor of shape [num_graphs, num_features]
    """
    num_graphs = batch.max().item() + 1
    num_features = x.size(1)
    
    # Initialize output tensor with very low values
    max_pooled = torch.full((num_graphs, num_features), float('-inf'), device=x.device)

    for i in range(num_graphs):
        mask = (batch == i)  # Get all nodes in graph i
        x_i = x[mask]
        if x_i.size(0) > 0:
            max_vals, _ = torch.max(x_i, dim=0)
            max_pooled[i] = max_vals

    return max_pooled

class GAE_CLS_Link_NODECLS_Cosine(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)

        self.node_cls = nn.Linear(embedding_size+num_features, 2)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.3)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, embedding_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return x, h, xs
    
    def adj_decode(self, z, batch_index):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.sigmoid(torch.mm(node_emb_1, node_emb_2.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def node_logits(self, z, x):
        node_out = torch.cat([z, x], dim=1)
        logits = self.node_cls(node_out)
        return logits

    def graph_logits(self, z, edge_index, batch):
        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class AttnReadout(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x, batch):
        weights = torch.sigmoid(self.attn(x))           # shape: [num_nodes, 1]
        x = x * weights                                  # weighted features
        return gap(x, batch)                # aggregated per graph

class GAE_CLS_Link_NODE_Cosine_SupCon_2(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048, attn_head = 1):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)

        HEADS = attn_head

        self.layer_type = layer_type
        # num_layers -= 1
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=1)

        node_dim = embedding_size
        if ("gat" in layer_type):
            node_dim *= HEADS

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, pooling_dim*2),
            # ReLU(inplace=True),
            # nn.GELU(inplac/e=True),
             ReLU(inplace=True) if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU() if (activate == "elu") else nn.GELU(),
            Dropout(0.3),
            Linear(pooling_dim*2, projection_emb),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.read_out = Set2Set(node_dim, processing_steps=3, num_layers=2)
        # self.read_out = AttnReadout(node_dim)
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device='cpu', acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        # self.ff1 = self.ff1.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.read_out = self.read_out.to(device)

        if (self.id_dim > 1):
            data = self.id_encode(data, device)
        elif (self.id_dim == 0):
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            _, h, _ = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            _, h, _ = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return h
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def graph_pooling(self, z, batch):
        return self.read_out(z, batch)
    
    def graph_embedding(self, z, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        # max_pooled = manual_global_max_pool(z, batch)
        # global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        global_pooling = self.graph_pooling(z, batch)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine_Cross_2(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)

        HEADS = 1

        self.layer_type = layer_type
        # num_layers -= 1
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=1)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 128),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU() if (activate == "elu") else nn.GELU(),
            Dropout(0.3),
            Linear(128, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.read_out = Set2Set(embedding_size, processing_steps=3, num_layers=2)
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):  
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        # self.ff1 = self.ff1.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.read_out = self.read_out.to(device)
        # self.shared_gnn_layers = self.shared_gnn_layers.to(device)
        # self.shared_norms = self.shared_norms.to(device)
        # self.pooling1 = self.pooling1.to(device)



        if (self.id_dim > 1):
            data = self.id_encode(data, device)
        elif (self.id_dim == 0):
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            _, h, _ = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            _, h, _ = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return h
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def graph_pooling(self, z, batch):
        return self.read_out(z, batch)
    
    def graph_embedding(self, z, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        # max_pooled = manual_global_max_pool(z, batch)
        # global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        global_pooling = self.graph_pooling(z, batch)
        linear = self.linear(global_pooling)
        return linear

class GAE_CLS_Link_NODE_Cosine_Cross_RPL(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)

        HEADS = 1

        self.layer_type = layer_type
        # num_layers -= 1
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=1)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.projection = nn.Sequential(
            Linear(pooling_dim, pooling_dim*2),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU() if (activate == "elu") else nn.GELU(),
            Dropout(0.5),
            Linear(pooling_dim*2, projection_emb),
        )

        self.classify = nn.Sequential(
            Linear(pooling_dim, pooling_dim*2),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU() if (activate == "elu") else nn.GELU(),
            Dropout(0.5),
            Linear(pooling_dim*2, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.read_out = Set2Set(embedding_size, processing_steps=3, num_layers=2)
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):  
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        # self.ff1 = self.ff1.to(device)
        self.classify = self.classify.to(device)
        self.projection = self.projection.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.read_out = self.read_out.to(device)
        # self.shared_gnn_layers = self.shared_gnn_layers.to(device)
        # self.shared_norms = self.shared_norms.to(device)
        # self.pooling1 = self.pooling1.to(device)



        if (self.id_dim > 1):
            data = self.id_encode(data, device)
        elif (self.id_dim == 0):
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            _, h, _ = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            _, h, _ = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return h
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def graph_pooling(self, z, batch):
        return self.read_out(z, batch)

    def graph_logits(self, z, batch):
        global_pooling = self.graph_pooling(z, batch)
        linear = self.classify(global_pooling)
        return linear
    
    def graph_embedding(self, z, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        # max_pooled = manual_global_max_pool(z, batch)
        # global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        global_pooling = self.graph_pooling(z, batch)
        linear = self.projection(global_pooling)
        return linear


class GVAE_CLS_Link_NODE_Cosine_SupCon_2(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)

        HEADS = 1

        self.layer_type = layer_type
        # num_layers -= 1
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=1)
        self.mu_net = nn.Linear(embedding_size * HEADS, embedding_size * HEADS)
        self.logvar_net = nn.Linear(embedding_size * HEADS, embedding_size * HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU() if (activate == "elu") else nn.GELU(),
            Dropout(0.3),
            Linear(512, projection_emb),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.read_out = Set2Set(embedding_size, processing_steps=3, num_layers=2)
        self.decoder = MLPDecoder(node_dim, num_features)

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        # Move submodules to device if not already done during init.
        # It's generally better to move the whole model to device once: model.to(device)
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.read_out = self.read_out.to(device)
        self.mu_net = self.mu_net.to(device)
        self.logvar_net = self.logvar_net.to(device)
        if self.linear_node:
            self.node_out_1 = self.node_out_1.to(device)
            self.node_out_2 = self.node_out_2.to(device)



        if (self.id_dim > 1):
            data = self.id_encode(data, device)
        elif (self.id_dim == 0):
            data.x = data.x[:, 1:]

        # The variable 'h_intermediate' is what your original encoder produced as 'h'
        _, h_intermediate, _ = self.encoder(data.x, data.edge_index, data.edge_attr if "gat" in self.layer_type else None, acummulate=acummulate, remove_random=remove_random) 

        mu = self.mu_net(h_intermediate)
        logvar = self.logvar_net(h_intermediate)

        return mu, logvar, h_intermediate
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def kl_divergence_loss(self, mu, logvar):
        """
        Calculates KL Divergence between the learned latent distribution and a
        standard Gaussian.
        KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            = 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
        """
        # kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # Sum over latent dimensions
        kld_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)
        return torch.mean(kld_loss) # Average over the batch
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def graph_pooling(self, z, batch):
        return self.read_out(z, batch)
    
    def graph_embedding(self, z, batch):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        # max_pooled = manual_global_max_pool(z, batch)
        # global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        global_pooling = self.graph_pooling(z, batch)
        linear = self.linear(global_pooling)
        return linear


class GAE_CLS_Link_NODE_Cosine_Semi(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, encoder = None, linear_node=False, num_id_embeddings = 2048):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=num_id_embeddings, embedding_dim=id_dim)
        
        HEADS = 1 

        # self.shared_gnn_layers = GATv2Conv(num_features, embedding_size, heads=HEADS, edge_dim=3, add_self_loops=True, dropout=0.5)
        # self.shared_norms = GraphNorm(embedding_size * HEADS)
        # self.shared_activate = ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU()

        self.layer_type = layer_type
        # num_layers -= 1
        self.encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=1)
        self.unsup_encoder = encoder if (encoder) else Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=1)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, projection_emb),
        )
        self.linear_unsup = nn.Sequential(
            Linear(pooling_dim, 512),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            Linear(512, projection_emb),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.read_out = Set2Set(embedding_size, processing_steps=3, num_layers=2)
        self.decoder = MLPDecoder(node_dim, num_features)

        # Bilinear weight for link recon
        self.weight  = nn.Parameter(torch.randn(64))

        self.linear_node = linear_node
        if (linear_node):
            self.node_out_1 = nn.Linear(node_dim, embedding_size)
            self.node_out_2 = nn.Linear(node_dim, embedding_size)

        self.node_proj_1 = nn.Linear(num_features, node_dim)
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def convert_binary_to_decimal(self, binary_input):
        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)

    def id_encode(self, data, device):
        id_tensor = data.x[:, 0].long()
        data.x = data.x[:, 1:]
        id_embedding = self.id_embedding(id_tensor).to(device)
        data.x = torch.cat([data.x, id_embedding], dim=1)
        
        return data

    def forward(self, data, device, acummulate = True, remove_random=True):    
        self.id_embedding = self.id_embedding.to(device)
        # self.ff1 = self.ff1.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.shared_gnn_layers = self.shared_gnn_layers.to(device)
        # self.shared_norms = self.shared_norms.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 1):
            data = self.id_encode(data, device)
        elif (self.id_dim == 0):
            data.x = data.x[:, 1:]

        if ("gat" in self.layer_type):
            _, h, _ = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
            _, h_unsup, _ = self.unsup_encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            _, h, _ = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)
            _, h_unsup, _ = self.unsup_encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        return h, h_unsup
    
    def adj_decode(self, z, batch_index, use_sigmoid=True):
        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = z[graph_nodes]  # Node embeddings for this graph
        
            if (self.linear_node):
                node_emb_1 = self.node_out_1(graph_emb)
                node_emb_2 = self.node_out_2(graph_emb)
            else:
                node_emb_1 = node_emb_2 = graph_emb

            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.mm(node_emb_1, node_emb_2.t())  # [num_nodes, num_nodes]
            if (use_sigmoid):
                graph_adj = torch.sigmoid(graph_adj)

            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)

    def node_recon(self, z):
        reconstructed_X = self.decoder(z)
 
        return reconstructed_X

    def cosine_score(self, x_ego, z_neigh):
        z_ego = self.node_proj_1(x_ego)
        
        z_ego = F.normalize(z_ego, dim=1)
        z_neigh = F.normalize(z_neigh, dim=1)

        # Positive pairs
        pos_score = self.cos(z_ego, z_neigh)

        # Negative pairs (shuffle)
        idx = torch.randperm(x_ego.size(0))
        neg_score = self.cos(z_ego, z_neigh[idx])

        return pos_score, neg_score
    
    def cosine_contrastive_loss(self, x_ego, z_neigh):
        pos_score, neg_score = self.cosine_score(x_ego, z_neigh)
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_pooling(self, z, batch):
        return self.read_out(z, batch)
    
    def graph_embedding(self, z, batch, unsup=False):
        # global pooling
        # x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        # max_pooled = manual_global_max_pool(z, batch)
        # global_pooling = torch.cat([max_pooled, gap(z, batch)], dim=1)
        global_pooling = self.graph_pooling(z, batch)
        linear = self.linear(global_pooling) if (not unsup) else self.linear_unsup(global_pooling)
        return linear


class GAE_2(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=2, directed=False, id_dim = 4, num_classes = 0):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 1 
        self.max_num_nodes = 128

        self.layer_type = layer_type
        self.encoder = Graph_Encoder_Norm_Pooling(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)
        # self.edge_decoder = EdgeDecoder(embedding_size * HEADS)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.8)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = Graph_Decoder_Norm_Unpooling(num_features, layer_type, num_layers, directed, HEADS)

    def adj_decode(self, x, batch_index):
        decoded_adjs = []

        for graph_id in range(batch_index.max().item() + 1):
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]

            # Use MLP edge decoder
            graph_adj = self.edge_decoder(graph_emb)
            decoded_adjs.append(graph_adj)

        # Pad to max number of nodes
        max_nodes = max(adj.size(0) for adj in decoded_adjs)
        padded_adjs = [F.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        return torch.stack(padded_adjs)

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, acummulate = False, remove_random=False):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            id_tensor = data.x[:, 0].long()
            data.x = data.x[:, 1:]
            id_embedding = self.id_embedding(id_tensor).to(device)
            data.x = torch.cat([data.x, id_embedding], dim=1)

        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        x_classify = x
        global_pooling = torch.cat([gmp(x_classify, data.batch), gap(x_classify, data.batch)], dim=1)

        linear = self.linear(global_pooling)

        node_embeddings = x
        reconstructed_X = self.decoder(x, data.edge_index, data.batch, data.edge_attr if data.edge_attr is not None else None)
        adj_reconstructed = self.adj_decode(x, data.batch)
 
        return linear, node_embeddings, reconstructed_X, adj_reconstructed 

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

class GraphUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, pool_ratios=0.5, act=F.relu, dropout=0.0):
        super(GraphUNet, self).__init__()

        self.depth = depth

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

    def forward(self, data):
        adj, x = self._pyg_to_dense(data)
        edge_index = data.edge_index

        # Initial embedding
        h = self.gcn_in(adj, x)

        # Downsampling
        hs = [h]
        adjs = [adj]
        edge_indices = [edge_index]

        for i in range(self.depth):
            edge_index, h, idx = self.pools[i](adj, h, edge_indices[-1])
            self.down_pool_indices.append(idx)
            h = self.down_gcns[i](adj, h)
            hs.append(h)
            adjs.append(adj)
            edge_indices.append(edge_index)

        # Bottleneck
        h = self.bn_gcn(adj, h)

        # Upsampling
        for i in range(self.depth):
            up_idx = self.depth - i - 1
            edge_index, h = self.unpools[i](adjs[up_idx], h, self.down_pool_indices[up_idx], edge_indices[up_idx])
            h = self.up_gcns[i](adjs[up_idx], h[:hs[up_idx].shape[0], :])
            h = h + hs[up_idx]  # Skip connection

        # Output layer
        h = self.gcn_out(adjs[0], h)

        # Clear saved indices for next forward pass
        self.down_pool_indices = []

        data.x = h
        return data

class GAE_3(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0, edge_dim = 1):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 1 

        self.layer_type = layer_type
        self.encoder = Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS, edge_dim=edge_dim)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        # self.pooling1 = TopKPooling(node_dim, ratio=0.8)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.GELU() if (activate == 'gelu') else nn.ELU(),
            Dropout(0.3),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        # self.decoder = Graph_Decoder_Norm_Unpooling_2(embedding_size, num_features, layer_type, num_layers, directed, HEADS)
        self.decoder = MLPDecoder(embedding_size, embedding_size)


        self.feature_recon = nn.Linear(embedding_size, num_features)
        self.reduced_1 = nn.Linear(embedding_size, embedding_size)
        self.reduced_2 = nn.Linear(embedding_size, embedding_size)

    def adj_decode(self, x, batch_index):

        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]  # Node embeddings for this graph

            h1 = self.reduced_1(graph_emb)
            h2 = self.reduced_2(graph_emb)
            
            # Decode adjacency matrix (dot product for simplicity)
            # graph_adj = torch.mm(h1, h2.t())  # [num_nodes, num_nodes]
            # graph_adj = torch.sigmoid(h1 @ h2.T)  # Apply sigmoid to get probabilities
            graph_adj = h1 @ h2.T
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, acummulate = True, remove_random=False):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        # self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            id_tensor = data.x[:, 0].long()
            data.x = data.x[:, 1:]
            id_embedding = self.id_embedding(id_tensor).to(device)
            data.x = torch.cat([data.x, id_embedding], dim=1)
        else:
            data.x = data.x[:, 1:]

        batch = data.batch
        # if (self.layer_type == "gat"):
        #     #x, x, layer_data, new_edge_index, new_batch
        #     x, h, layer_data, new_edge_index, new_batch = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate)
        # else:
        #     x, h, layer_data, new_edge_index, new_batch = self.encoder(data.x, data.edge_index, acummulate = acummulate)
        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)
        # batch = new_batch
        
        x_classify = x
        # x2, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)

        linear = self.linear(global_pooling)

        # reconstructed_X = self.decoder(x, layer_data)
        reconstructed_X = self.decoder(x)

        adj_reconstructed = self.adj_decode(reconstructed_X, data.batch)
        reconstructed_X = self.feature_recon(reconstructed_X)
        reconstructed_X = F.gelu(reconstructed_X)
 
        return linear, x, reconstructed_X, adj_reconstructed 

class GAE_Projection_Att_WithIDEncoder_3_NodeCLS(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 2 

        self.layer_type = layer_type
        self.encoder = Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        self.node_cls = nn.Linear(embedding_size * HEADS + num_features, 1)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        self.ae = GraphAutoEncoder(input_dim=embedding_size, hidden_dim=int(embedding_size/2), output_dim=embedding_size)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.8)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = AttentionDecoder(node_dim, num_features)

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

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, acummulate = True, remove_random=False):    
        # binary_id = data.binary_id.to(dtype=torch.float32, device=device)
        # dec_id = data.dec_id

        # ids = []
        # for id in dec_id:
        #     ids.extend(id)
        # ids = torch.tensor(ids).to(dtype=torch.int64, device=device)

        # Encode binary ID
        # Assuming data.binary_id has shape (2507, 29)
        # Convert binary to decimal if required
        # tensor_id = torch.matmul(binary_id, 2**torch.arange(29).flip(0).to(device, dtype=torch.float32))

        # tensor_id = tensor_id.to(dtype=torch.int64, device=device)
        # id_embedding = self.id_encoder(tensor_id).to(device)

        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.ae = self.ae.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            id_tensor = data.x[:, 0].long()
            data.x = data.x[:, 1:]
            id_embedding = self.id_embedding(id_tensor).to(device)
            data.x = torch.cat([data.x, id_embedding], dim=1)

        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate, remove_random=remove_random)

        node_cls = self.node_cls(torch.cat([data.x, x], dim=1))
        node_cls = torch.sigmoid(node_cls)

        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(x, data.edge_index, batch=data.batch)
        # x2, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)

        linear = self.linear(global_pooling)

        reconstructed_X = self.decoder(h)
        adj_reconstructed = self.adj_decode(x, data.batch)
 
        return linear, reconstructed_X, adj_reconstructed, node_cls
     
class GAE_Projection_Att_WithIDEncoder_4(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64, projection_emb=64, activate = "leakyReLU", layer_type="gat", num_layers=4, directed=False, id_dim = 4, num_classes = 0):
        super().__init__()

        self.id_dim = id_dim
        self.id_embedding = nn.Embedding(num_embeddings=2048, embedding_dim=id_dim)
        
        HEADS = 2 

        self.layer_type = layer_type
        self.encoder = Graph_Encoder_Norm(num_features, embedding_size, activate=activate, layer_type=layer_type, num_layers=num_layers, directed=directed, num_heads=HEADS)
        # self.decoder = NeighborhoodAwareDecoder(embedding_size * HEADS, num_features, num_heads=HEADS)
        # self.decoder = Graph_Decoder(embedding_size, num_features, layer_type, num_layers, directed, HEADS)

        self.ae = GraphAutoEncoder(input_dim=embedding_size, hidden_dim=int(embedding_size/2), output_dim=embedding_size)

        node_dim = embedding_size * HEADS if ("gat" in layer_type) else embedding_size
        self.pooling1 = TopKPooling(node_dim, ratio=0.8)
        # self.pooling2 = SAGPooling(hidden_channels, ratio=k)

        pooling_dim = embedding_size * 2 * HEADS if ("gat" in layer_type) else embedding_size * 2
        self.linear = nn.Sequential(
            Linear(pooling_dim, embedding_size),
            ReLU() if (activate == "relu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ELU(),
            Dropout(0.3),
            nn.Linear(embedding_size, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        # node_dim = embedding_size * 2 if ("gat" in layer_type) else embedding_size
        self.decoder = AttentionDecoder(node_dim, num_features)

    def adj_decode(self, x, batch_index):

        # Decoder: Per-graph adjacency reconstruction
        decoded_adjs = []  # Store adjacency matrices for all graphs
        for graph_id in range(batch_index.max().item() + 1):
            # Get nodes belonging to this graph
            graph_nodes = (batch_index == graph_id).nonzero(as_tuple=True)[0]
            graph_emb = x[graph_nodes]  # Node embeddings for this graph
            
            # Decode adjacency matrix (dot product for simplicity)
            graph_adj = torch.relu(torch.mm(graph_emb, graph_emb.t()))  # [num_nodes, num_nodes]
            decoded_adjs.append(graph_adj)

        # Pad adjacency matrices to the largest size
        max_nodes = max(adj.size(0) for adj in decoded_adjs)  # Largest number of nodes
        padded_adjs = [torch.nn.functional.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(0)))
                    for adj in decoded_adjs]
        batched_adjs = torch.stack(padded_adjs)  # Shape: [64, max_nodes, max_nodes]
        return batched_adjs

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, data, device, acummulate = True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.ae = self.ae.to(device)

        if (self.id_dim > 0):
            id_tensor = data.x[:, 0].long()
            data.x = data.x[:, 1:]
            id_embedding = self.id_embedding(id_tensor).to(device)
            data.x = torch.cat([data.x, id_embedding], dim=1)

        if (self.layer_type == "gat"):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate)
        else:
            x, h, xs = self.encoder(data.x, data.edge_index, acummulate = acummulate)

        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(x, data.edge_index, batch=data.batch)
        # x2, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)

        linear = self.linear(global_pooling)

        reconstructed_X = self.decoder(h)
        adj_reconstructed = self.adj_decode(x, data.batch)
 
        return linear, reconstructed_X, adj_reconstructed 