import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LinearClassifier(nn.Module):
    def __init__(self, n_classes, feat_dim, init=False):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim*2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(feat_dim*2, n_classes),
        )
        if init:
            for layer in self.fc:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_normal_(layer.weight)
                    torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        output = self.fc(x)
        return output
    
class PrototypeClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(PrototypeClassifier, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.prototypes = None

    def set_prototypes(self, prototypes):
        """
        Set the prototypes for the classifier.
        prototypes: (num_classes, embedding_dim)
        """
        assert prototypes.shape == (self.num_classes, self.embedding_dim), \
            f"Expected shape {(self.num_classes, self.embedding_dim)}, got {prototypes.shape}"
        
        self.prototypes = nn.Parameter(prototypes)  # learnable

    def forward(self, embeddings):
        """
        embeddings: (B, D)
        Returns:
            distances: (B, num_classes) - lower means more similar
            logits: -negative distances
        """
        return -self.hybrid_distance_loss(embeddings, self.prototypes)
    
    def hybrid_distance_loss(self, z, p, margin=0.5, alpha=0.5):
        """
        z: sample embedding, shape [B, D]
        p: prototypes, shape [C, D]
        margin: margin for pushing apart negative classes
        alpha: weight to combine Euclidean and cosine
        """
        # Euclidean Distance
        euclid = torch.cdist(z, p, p=2)  # shape: [B, C]

        # Cosine Distance (1 - cos similarity)
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(p, dim=1)
        cosine = 1 - torch.matmul(z_norm, p_norm.T)  # shape: [B, C]

        # Combine both
        hybrid_dist = euclid
        return hybrid_dist