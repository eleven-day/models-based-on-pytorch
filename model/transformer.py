import torch
import torch.nn as nn
import math
from .encoder import Encoder
from .decoder import Decoder
from .utils import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_layers=6, d_ff=2048, h=8, dropout=0.1):
        """
        Full Transformer model as described in "Attention is All You Need"
        
        Args:
            src_vocab: Source vocabulary size
            tgt_vocab: Target vocabulary size
            d_model: Model dimension
            n_layers: Number of encoder/decoder layers
            d_ff: Feed-forward dimension
            h: Number of attention heads
            dropout: Dropout probability
        """
        super(Transformer, self).__init__()
        
        # Embeddings and positional encoding
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Encoder and decoder stacks
        self.encoder = Encoder(n_layers, d_model, d_ff, h, dropout)
        self.decoder = Decoder(n_layers, d_model, d_ff, h, dropout)
        
        # Final linear layer and softmax
        self.generator = nn.Linear(d_model, tgt_vocab)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, src, src_mask):
        """Encode the source sequence"""
        src = self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim)
        src = self.pos_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        """Decode with the encoded source"""
        tgt = self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim)
        tgt = self.pos_encoding(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Forward pass through the entire model"""
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(output)
    
    @staticmethod
    def create_masks(src, tgt=None):
        """Create masks for transformer training"""
        # Source mask (padding mask)
        src_mask = (src != 0).unsqueeze(-2)
        
        if tgt is None:
            return src_mask
        
        # Target mask combines padding mask and look-ahead mask
        tgt_mask = (tgt != 0).unsqueeze(-2)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        if src.is_cuda:
            nopeak_mask = nopeak_mask.cuda()
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
