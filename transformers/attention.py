import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        """
        Multi-Head Attention module
        
        Args:
            d_model: Model dimension
            h: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        
        # We assume d_v = d_k = d_model / h
        self.d_k = d_model // h
        self.h = h
        
        # Linear projections
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def attention(self, query, key, value, mask=None, dropout=None):
        """Scaled Dot Product Attention"""
        d_k = query.size(-1)
        
        # (batch, h, seq_len, d_k) x (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
            
        # (batch, h, seq_len, seq_len) x (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into h heads
        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # Apply attention
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.output_linear(x)
