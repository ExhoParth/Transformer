import math

import torch
import torch.nn as nn

import copy

class Embeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# TODO: OutputEmbeddings

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  

        # creating a matrix of shape (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        # creating a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply the sin to even positions
        positional_encoding[:, 0::2] = torch.sin(position * denominator)

        # apply the cos to oddd positions
        positional_encoding[:, 1::2] = torch.cos(position * denominator)

        positional_encoding = positional_encoding.unsqueeze(0) # (1, seq_len, d_model) for batch

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class PositionwiseFFN(nn.Module):

    def __init__(self, d_model:int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.l_1 = nn.Linear(d_model, d_ff) # W_1 & b_1
        self.dropout = nn.Dropout(dropout)
        self.l_2 = nn.Linear(d_ff, d_model) # W_2 & b_2

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.l_2(self.dropout(torch.relu(self.l_1(x))))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, f"`d_model` ({d_model}) must be divisible by `h` ({h})."

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Expand mask to (batch_size, 1, seq_len, seq_len) for broadcasting over heads
            mask = mask.unsqueeze(1)  # Adds a dimension at position 1
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().reshape(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)

class SublayerResConnection(nn.Module):
    def __init__(self, size, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, size: int, self_attention: MultiHeadAttention, positionwise_ffn: PositionwiseFFN, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.positionwise_ffn = positionwise_ffn
        self.residual_connections = nn.ModuleList([SublayerResConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.positionwise_ffn)
        return x

class Encoder(nn.Module):

    def __init__(self, layer, N) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, size: int, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, positionwise_ffn: PositionwiseFFN, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.positionwise_ffn = positionwise_ffn
        self.residual_connections = nn.ModuleList([SublayerResConnection(size, dropout) for _ in range(3)])
        self.size = size

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.positionwise_ffn)
        return x

class Decoder(nn.Module):

    def __init__(self, layer, N) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: Embeddings, tgt_embeddings: Embeddings, src_positions: PositionalEncoding, tgt_positions: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_positions = src_positions
        self.tgt_positions = tgt_positions
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embeddings(src)
        src = self.src_positions(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embeddings(tgt)
        tgt = self.tgt_positions(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # embedding layers
    src_embeddings = Embeddings(d_model, src_vocab_size)
    tgt_embeddings = Embeddings(d_model, tgt_vocab_size)

    # positional encoding layers
    src_positions = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positions = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    c = copy.deepcopy

    # attention
    attention = MultiHeadAttention(d_model, h, dropout)
    positionwise_ffn = PositionwiseFFN(d_model, d_ff, dropout)

    transformer = Transformer(
        encoder=Encoder(EncoderLayer(d_model, c(attention), c(positionwise_ffn), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attention), c(attention), c(positionwise_ffn), dropout), N),
        src_embeddings=src_embeddings,
        tgt_embeddings=tgt_embeddings,
        src_positions=src_positions,
        tgt_positions=tgt_positions,
        projection_layer=ProjectionLayer(d_model, tgt_vocab_size)

    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
