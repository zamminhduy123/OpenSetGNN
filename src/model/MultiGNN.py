import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool

# -------------------- Config -------------------- #
HIDDEN    = 64
LATENT    = 32
NUM_LAYERS= 2
K_FRAC    = 0.1        # top-K node pooling
ALPHA     = 1.0        # class-loss weight
BETA      = 0.5        # recon weight
GAMMA     = 0.5        # SVDD weight
LR        = 1e-3
EPOCHS    = 30
# ------------------------------------------------ #

# ----------------- Model ------------------------ #
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, layers=NUM_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden, heads=4, concat=False, add_self_loops=True, edge_dim=3))
        for _ in range(layers-1):
            self.convs.append(GATv2Conv(hidden, hidden, heads=4, concat=False, add_self_loops=True, edge_dim=3))
        self.lin_mu  = nn.Linear(hidden, LATENT)
    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        mu = self.lin_mu(x)
        return mu                     # (N, LATENT)

class MultiTaskGNN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.encoder = Encoder(in_dim)
        # Node decoder  (latent -> original features)
        self.node_dec = nn.Linear(LATENT, in_dim)
        # Graph classifier
        self.cls_head = nn.Linear(LATENT, num_classes)
        # Bilinear weight for link recon
        self.weight   = nn.Parameter(torch.randn(LATENT))
    # ---------- forward blocks ----------
    def encode(self, data):
        z = self.encoder(data.x, data.edge_index, data.edge_attr)
        return z
    def node_recon(self, z):          # (N,F)
        return self.node_dec(z)
    def link_score(self, z, edge):
        # edge: (2,E) indices
        return (z[edge[0]] * z[edge[1]] * self.weight).sum(dim=1)
    def graph_logits(self, z, batch):
        g_emb = global_add_pool(z, batch)
        return self.cls_head(g_emb)