import torch
import torch.nn as nn

class BinaryEncoder(nn.Module):
    def __init__(self, input_dim=29, embedding_dim=16):
        super(BinaryEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, binary_input):
        # binary_input shape: [num_nodes, 29]
        x = self.fc(binary_input)
        x = self.relu(x)
        return x  # shape: [num_nodes, embedding_dim]


class IdEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(IdEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def convert_binary_to_decimal(self, binary_input):

        # Convert binary to decimal
        decimal_ids = []
        for binary_id in binary_input:
            decimal_id = int("".join(str(int(bit.item())) for bit in binary_id), 2)
            decimal_ids.append(decimal_id)
        return torch.tensor(decimal_ids)
    
    def forward(self, input_ids):
        # input_ids shape: [num_nodes]
        embedded_ids = self.embedding(self.convert_binary_to_decimal(input_ids))
        return embedded_ids  # shape: [num_nodes, embedding_dim]