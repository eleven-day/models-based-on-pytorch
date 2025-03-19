import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Multi-head self-attention module"""
    
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        # Define query, key, value projections for all heads
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        # Shape: (batch_size, seq_length, embed_size)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Split embedding into self.heads pieces
        q = q.reshape(batch_size, seq_length, self.heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.heads, self.head_dim)
        
        # Transpose for attention dot product: (batch_size, heads, seq_length, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Compute attention scores
        # (batch_size, heads, seq_length, seq_length)
        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply mask for autoregressive property (optional)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        # (batch_size, heads, seq_length, head_dim)
        out = torch.matmul(attention, v)
        
        # Reshape back: (batch_size, seq_length, heads, head_dim)
        out = out.permute(0, 2, 1, 3)
        
        # Concat heads: (batch_size, seq_length, embed_size)
        out = out.reshape(batch_size, seq_length, self.embed_size)
        
        # Final linear layer
        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, embed_size, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Transformer decoder block with self-attention"""
    
    def __init__(self, embed_size, heads, dropout, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attention = self.attention(x, mask)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        
        # Feed forward with residual connection
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        x = self.dropout(x)
        
        return x


class GPT(nn.Module):
    """GPT model with transformer decoder blocks"""
    
    def __init__(
        self, 
        vocab_size, 
        embed_size=768,
        max_length=1024, 
        num_layers=12,
        heads=12, 
        ff_dim=3072, 
        dropout=0.1
    ):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        # Add transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, ff_dim) for _ in range(num_layers)]
        )
        
        self.ln_f = nn.LayerNorm(embed_size)  # Final layer norm
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        
    def forward(self, x, mask=None):
        # Get sequence length and create position indices
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(x.device)
        
        # Make sure sequence doesn't exceed maximum length
        assert seq_length <= self.max_length, f"Sequence length ({seq_length}) exceeds maximum length ({self.max_length})"
        
        # Apply token and position embeddings
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        x = self.dropout(token_embed + pos_embed)
        
        # Create attention mask for autoregressive property (optional)
        if mask is None and seq_length > 1:
            # Create a lower triangular mask to prevent attending to future tokens
            mask = torch.tril(torch.ones((seq_length, seq_length))).to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        return logits
