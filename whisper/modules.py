import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert self.head_dim * n_head == n_embd, "n_embd must be divisible by n_head"
        
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # If kv is provided, use it for keys and values (cross-attention)
        if kv is not None:
            k = v = kv
        else:
            k = v = x
            
        # Linear projections and reshape to (B, nh, T, hs)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.k_proj(k).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T_k, hs)
        v = self.v_proj(v).view(B, -1, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T_k, hs)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T_k)
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Compute the weighted sum over values
        output = attn @ v  # (B, nh, T, hs)
        
        # Reshape and project back to (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        output = self.out_proj(output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                cross_attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x), mask)
        # Feed-forward with residual connection
        x = x + self.ff(self.ln2(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                cross_attn_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.ln1(x), mask)
        
        # Cross-attention
        if cross_attn_kv is not None:
            x = x + self.cross_attn(self.ln2(x), kv=cross_attn_kv)
            
        # Feed-forward
        x = x + self.ff(self.ln3(x))
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, n_embd: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)].expand(x.size(0), -1, -1)
