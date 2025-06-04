import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, ModuleList
from torch_geometric.nn import (
    TopKPooling,
    global_mean_pool as gmp,
    global_add_pool as gap,
    GATConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
)
from torch_geometric.nn.norm import GraphNorm

class Graph_Encoder_Norm(nn.Module):
    def __init__(self, num_features, embedding_size=64, layer_type="gat", num_layers=4, activate = "leakyReLU", directed=False, num_heads=1, edge_dim=0):
        # Init parent
        super(Graph_Encoder_Norm, self).__init__()

        self.bn = torch.nn.BatchNorm1d(num_features) 
        # self.bn = nn.LayerNorm(num_features)
        # self.bn = torch.nn.GroupNorm(1, num_features)
        self.input = torch.nn.Linear(num_features, embedding_size)

        self.layer_type = layer_type
        def GCNConvType(input_dim, output_dim, heads=1, alpha=0.5):
            return GATv2Conv(input_dim, output_dim, heads, alpha, edge_dim=edge_dim, share_weights=True, )
        
        self.convs = ModuleList([GCNConvType(embedding_size, embedding_size, heads=num_heads, alpha=0.5)])
        # self.norms = nn.ModuleList([nn.LayerNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size)])

        self.norms = nn.ModuleList([GraphNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size)])

        for _ in range(num_layers - 1):
            self.convs.append(GCNConvType(embedding_size * num_heads if ("gat" in layer_type) else embedding_size, embedding_size, heads=num_heads, alpha=0.5))
            self.norms.append(GraphNorm(embedding_size * num_heads if ("gat" in layer_type) else embedding_size))

        self.activate_function = nn.ELU() if (activate == "elu") else nn.LeakyReLU() if (activate == "leakyReLU") else nn.ReLU()
        

        self.directed = directed

    def forward(self, x, edge_index, edge_attribute=None, acummulate = False, remove_random=False, return_one = False, batch=None, get_attention_weights=False):
        self.bn = self.bn.to(x.device)
        self.input = self.input.to(x.device)
        self.activate_function = self.activate_function.to(x.device)
        self.convs = self.convs.to(x.device)
        
        # encode feature matrix
        if (not remove_random):
            x = self.bn(x)
        x = self.input(x)
        x = self.activate_function(x)

        # encode feature matrix
        xs = []
        h = None
        for i, conv in enumerate(self.convs):
            if ("gat" in self.layer_type):
                x, (_, attention_weights) = conv(x, edge_index, edge_attribute, return_attention_weights=get_attention_weights)
                xs.append(attention_weights.squeeze(1))
            else:
                x = conv(x, edge_index)

            # Normalize each layer output
            x = self.norms[i](x, batch=batch)

            x = self.activate_function(x)
            x = F.dropout(x, p=0.3, training=self.training)

            layer_flatten = x.view(-1)
            if i >= 1:
                layer_flatten = F.pad(layer_flatten, (0, xs[0].size(0) - layer_flatten.size(0)))
            # xs.extend(layer_flatten)

            if (h is None):
                h = x * 0.5
            else:
                if (x.size(1) > h.size(1)):
                    h = F.pad(h, (0, x.size(1) - h.size(1)))
                h += x * 0.5

            if (not get_attention_weights):
                xs = None


        if (acummulate):
            if (return_one):
                return h
            else:
                return x, h, torch.stack(xs, dim=0) if (xs is not None) else None
        if (return_one):
            return x
        return x, x, torch.stack(xs, dim=0) if (xs is not None) else None

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

class GAE_CLS_Link_NODE_Cosine_AttValue(torch.nn.Module):
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

    def forward(self, data, device, acummulate = True, remove_random=True, get_attention_weights=True):    
        self.id_embedding = self.id_embedding.to(device)
        self.linear = self.linear.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        self.pooling1 = self.pooling1.to(device)

        if (self.id_dim > 0):
            data = self.id_encode(data, device)

        if ("gat" in self.layer_type):
            x, h, xs = self.encoder(data.x, data.edge_index, data.edge_attr, acummulate = acummulate, remove_random=remove_random, get_attention_weights=get_attention_weights)
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

    def cosine_contrastive_loss(self, pos_score, neg_score):
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        loss_pos = self.loss_fn(pos_score, pos_labels)
        loss_neg = self.loss_fn(neg_score, neg_labels)

        loss = loss_pos + loss_neg
        return loss, pos_score, neg_score

    def graph_logits(self, z, edge_index, batch):
        # global pooling
        x_classify, edge_index, _, batch, _, _ = self.pooling1(z, edge_index, batch=batch)
        global_pooling = torch.cat([gmp(x_classify, batch), gap(x_classify, batch)], dim=1)
        linear = self.linear(global_pooling)
        return linear

