import torch.nn as nn
from .attention import MultiHeadAttention
from .utils import PositionwiseFeedForward, LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        """
        Decoder Layer consisting of Two Multi-Head Attention layers and Feed Forward
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            h: Number of attention heads
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        # Sublayer 1: Masked multi-head attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Sublayer 2: Cross attention with residual connection and layer normalization
        attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Sublayer 3: Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, h, dropout=0.1):
        """
        Decoder stack of N layers
        
        Args:
            n_layers: Number of decoder layers
            d_model: Model dimension
            d_ff: Feed-forward dimension
            h: Number of attention heads
            dropout: Dropout probability
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, h, dropout) for _ in range(n_layers)])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
