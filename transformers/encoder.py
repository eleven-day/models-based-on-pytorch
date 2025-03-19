import torch.nn as nn
from .attention import MultiHeadAttention
from .utils import PositionwiseFeedForward, LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        """
        Encoder Layer consisting of Multi-Head Attention and Feed Forward
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            h: Number of attention heads
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Sublayer 1: Multi-head attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Sublayer 2: Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, h, dropout=0.1):
        """
        Encoder stack of N layers
        
        Args:
            n_layers: Number of encoder layers
            d_model: Model dimension
            d_ff: Feed-forward dimension
            h: Number of attention heads
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, h, dropout) for _ in range(n_layers)])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
