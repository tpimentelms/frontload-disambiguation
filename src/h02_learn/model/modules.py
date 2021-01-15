"""
Transformer implementation was based on the one found in:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/76762bb08225014fb3055a9d07f0043aba972d68
"""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, temperature=1, dropout=0.3):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        q_temp = q / self.temperature

        weights = torch.matmul(q_temp, k.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(-2)
            weights = weights.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.softmax(weights, dim=-1))

        output = torch.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    # pylint: disable=arguments-differ,too-many-instance-attributes
    def __init__(self, n_heads, d_input, d_key, d_value, dropout=0.3):
        super().__init__()

        self.n_heads = n_heads
        self.d_input = d_input
        self.d_key = d_key
        self.d_value = d_value
        self.dropout_p = dropout

        self.linear_key = nn.Linear(d_input, n_heads * d_key)
        self.linear_query = nn.Linear(d_input, n_heads * d_key)
        self.linear_value = nn.Linear(d_input, n_heads * d_value)
        self.linear_out = nn.Linear(n_heads * d_value, d_input)

        self.attention = ScaledDotProductAttention(temperature=d_key ** 0.5, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        residual = q
        q = self.layer_norm(q)

        batch_size, len_q, _ = q.shape

        q = self.linear_query(q).reshape(batch_size, len_q, self.n_heads, self.d_key)
        k = self.linear_key(k).reshape(batch_size, k.shape[1], self.n_heads, self.d_key)
        v = self.linear_value(v).reshape(batch_size, v.shape[1], self.n_heads, self.d_key)

        # Transpose for the attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        hidden = self.attention(q, k, v, mask)

        # Transpose to move the head dimension back: bs x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        hidden = hidden.transpose(1, 2).contiguous().reshape(batch_size, len_q, -1)
        output = self.dropout(self.linear_out(hidden))
        output += residual

        return output


class MultiHeadSelfAttention(MultiHeadAttention):
    # pylint: disable=arguments-differ
    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask)


class MlpBottleneck(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, d_input, d_hidden, dropout=0.3):
        super().__init__()

        self.linear_in = nn.Linear(d_input, d_hidden)
        self.linear_out = nn.Linear(d_hidden, d_input)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x):
        residual = x

        x = self.layer_norm(x)
        x = self.linear_in(x)
        x = torch.relu(x)
        x = self.linear_out(x)
        x = self.dropout(x)

        x += residual
        return x


class Transformer(nn.Module):
    # pylint: disable=arguments-differ,too-many-instance-attributes
    def __init__(self, n_layers, n_heads, d_input, d_key, d_value, d_bottleneck, dropout=0.3):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_input = d_input
        self.d_key = d_key
        self.d_value = d_value
        self.d_bottleneck = d_bottleneck
        self.dropout = dropout

        self.attentions = self.build_attentions()

    def build_attentions(self):
        self_attentions = []
        for layer in range(self.n_layers):
            self_attention = MultiHeadSelfAttention(
                self.n_heads, self.d_input, self.d_key, self.d_value,
                dropout=self.dropout)
            mlp_positional = MlpBottleneck(self.d_input, self.d_bottleneck, dropout=self.dropout)

            layer = nn.ModuleList([
                self_attention,
                mlp_positional
            ])
            self_attentions += [layer]

        attentions = nn.ModuleList(self_attentions)
        return attentions

    def forward(self, x, mask):
        for attn, mlp in self.attentions:
            x = attn(x, mask)
            x = mlp(x)

        return x


class PositionalEmbedding(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, n_positions, d_embedding):
        super().__init__()

        self.n_positions = n_positions
        self.d_embedding = d_embedding

        self.embedding = nn.Parameter(torch.Tensor(n_positions, d_embedding))

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        embs = self.embedding[:x.shape[1]]
        embs = embs.unsqueeze(0)
        return embs.repeat(x.shape[0], 1, 1)


class TransformerLM(nn.Module):
    # pylint: disable=arguments-differ,too-many-instance-attributes,too-many-arguments
    def __init__(self, alphabet_size, pad_idx, embedding_size, hidden_size,
                 nlayers, dropout, n_positions=500, tie_weights=True):
        super().__init__()

        self.alphabet_size = alphabet_size
        self.pad_idx = pad_idx
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout
        self.tie_weights = tie_weights

        # Alphabet + <MASK> embeddings
        self.char_embedding = nn.Embedding(alphabet_size, embedding_size)
        self.pos_embedding = PositionalEmbedding(n_positions, embedding_size)
        self.linear_in = nn.Linear(embedding_size * 2, hidden_size)

        self.transformer = Transformer(
            2, 10, hidden_size, int(hidden_size / 10), int(hidden_size / 10),
            int(hidden_size / 2), dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.out = nn.Linear(embedding_size, alphabet_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Tie weights
        self.logit_scale = (embedding_size ** -0.5)
        if tie_weights:
            self.out.weight = self.char_embedding.weight

    def forward(self, x):
        x_emb, mask = self.get_embeddings(x)

        hidden = self.linear_in(x_emb)
        hidden = self.transformer(hidden, mask)

        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden).contiguous()

        hidden = torch.relu(self.linear(hidden))
        logits = self.out(hidden)
        return logits * self.logit_scale

    def get_embeddings(self, instance):
        mask = (instance != self.pad_idx)
        char_emb = self.char_embedding(instance)
        pos_emb = self.pos_embedding(instance)

        emb = torch.cat([char_emb, pos_emb], dim=-1)
        return self.dropout(emb), mask
