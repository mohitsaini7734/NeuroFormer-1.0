# Importing Libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting Device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

def scaled_dot_product_attention(q, k, v, is_causal=True):
    d_k = q.size()[-1] # head_dim
    dot_product = torch.matmul(q, k.transpose(-1, -2)) # q: [B, num_heads, T, head_dim], kᵀ: [B, num_heads, head_dim, T], q @ kᵀ: [B, num_heads, T, T]
    scaled = dot_product / math.sqrt(d_k) # [B, num_heads, T, T]

    if is_causal:
        seq_len = q.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        scaled = scaled.masked_fill(~causal_mask, float('-inf'))

    scaled = scaled.clamp(min=-1e9, max=1e9)
    attention = F.softmax(scaled, dim=-1) # attention: [B, num_heads, T, T]
    values = torch.matmul(attention, v) # v: [B, num_heads, T, head_dim] , Values: [B, num_heads, T, head_dim]

    return attention, values

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model # Dimentions of Input Embeddings
        self.num_heads = num_heads # Number of Heads in Multi-Head Attention
        self.head_dim = d_model // num_heads # Dimentions of Embedding Passed in each Head
        self.qkv_layer = nn.Linear(d_model, 3 * d_model) # Projects Input as Query, Key and Value
        self.linear_layer = nn.Linear(d_model, d_model) # Returns output dimentions same as input embedding
        self.dropout = nn.Dropout(p=drop_prob) # Dropout Layer

    def forward(self, x):
        batch_size, sequence_length, d_model = x.size() # [B, T, d_model]
        qkv = self.qkv_layer(x) # [B, T, 3 * d_model]

        # Reshape and split into Q, K, V
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [B, num_heads, T, 3 * head_dim]

        q, k, v = qkv.chunk(3, dim=-1) # Each: [B, num_heads, T, head_dim]

        # Apply causal attention
        attention, values = scaled_dot_product_attention(q, k, v, is_causal=True)

        # Combine heads
        values = values.transpose(1, 2).contiguous() # [B, T, num_heads, head_dim]
        values = values.reshape(batch_size, sequence_length, d_model) # [B, T, d_model]

        out = self.linear_layer(values)
        out = self.dropout(out)
        return out

class LayerNormalization(nn.Module):

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model)) # Learnable Parameter
        self.beta = nn.Parameter(torch.zeros(d_model)) # Learnable Parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # Mean
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True) # Variance
        std = (var + self.eps).sqrt() # Standard Deviation : eps is added to avoid division by 0

        y = (x - mean) / std # Output
        out = self.gamma * y + self.beta # Applying Learnable Parameters
        return out

class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):

  def __init__(self, sequence_length, d_model, drop_prob=0.1):
    super().__init__()
    self.pos_embedding = nn.Embedding(sequence_length, d_model) # [T, d_model]
    self.dropout = nn.Dropout(p = drop_prob)

  def forward(self, x):

    batch_size, sequence_length, d_model = x.size() # [B, T, d_model]
    positions = torch.arange(sequence_length, device=x.device) # [T]
    positions = positions.unsqueeze(0) # [1, T]
    pos_emb = self.pos_embedding(positions) # [1, T, d_model]

    x = x + pos_emb # [B, T, d_model] Broadcast Addition
    x = self.dropout(x)
    return x

class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, hidden, drop_prob=0.1):
        super().__init__()

        self.norm1 = LayerNormalization(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, drop_prob)

        self.norm2 = LayerNormalization(d_model)
        self.ffn = PositionWiseFeedForward(d_model, hidden, drop_prob)

    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class GPTDecoder(nn.Module):

    def __init__(self, vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(sequence_length, d_model, drop_prob)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Transformer blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, hidden, drop_prob)
            for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.final_norm = LayerNormalization(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):

        self.eval()
        with torch.no_grad():
            for _ in range(max_length):

                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                if top_p is not None and top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if input_ids.size(1) > self.sequence_length:
                    input_ids = input_ids[:, -self.sequence_length:]

        return input_ids

class NeuroFormer(nn.Module):

    def __init__(self, vocab_size, sequence_length, d_model=512, num_heads=8, hidden=2048, num_layers=6, drop_prob=0.1):
        super().__init__()
        self.model = GPTDecoder(vocab_size, sequence_length, d_model, num_heads, hidden, num_layers, drop_prob)

    def forward(self, input_ids):
        return self.model(input_ids)

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9, eos_token_id=None):
        return self.model.generate(input_ids, max_length, temperature, top_k, top_p, eos_token_id)