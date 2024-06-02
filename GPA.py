import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix mult. for query*keys for each N example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, max_length, vocab_size):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.max_length = max_length

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
             for _ in range(num_layers)]
        )

        # Final linear layer
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.dropout(self.embedding(positions) + self.embedding(x))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out

# Paramètres du modèle Transformer
embed_size = 512
heads = 8
num_layers = 6
forward_expansion = 4
dropout = 0.1
max_length = 10000  # Taille maximale de la séquence
vocab_size = 10000  # Taille du vocabulaire

# Créer une instance du modèle Transformer
model = Transformer(embed_size, heads, num_layers, forward_expansion, dropout, max_length, vocab_size)

# Définir une entrée de test
input_ids = torch.randint(0, vocab_size, (1, 10))  # Séquence de taille 10 avec des identifiants de token

# Créer un masque pour ignorer les padding tokens
mask = (input_ids != 0).float()

# Passer l'entré à travers le modèle Transformer
outputs = model(input_ids, mask)

# Afficher la taille de la sortie
print(outputs.shape)
