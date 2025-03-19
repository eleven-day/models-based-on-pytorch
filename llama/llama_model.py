import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with the hidden states.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1] // 2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        xq: Query states tensor of shape (batch_size, seq_len, head_dim)
        xk: Key states tensor of shape (batch_size, seq_len, head_dim)
        freqs_cis: Precomputed frequency tensor of shape (seq_len, head_dim/2)
        
    Returns:
        Tuple of tensors (xq_out, xk_out) with applied rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input tensor
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms * self.weight
        return x_norm


class LLaMAAttention(nn.Module):
    """
    Multi-headed attention module with rotary position embeddings
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Ensure head_dim * num_heads equals hidden_size
        assert head_dim * num_heads == hidden_size, f"head_dim ({head_dim}) * num_heads ({num_heads}) != hidden_size ({hidden_size})"
        
        # Input projection layers
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
    def forward(
        self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Apply rotary embeddings to q and k
        q_rot = q.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        k_rot = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        q_rot, k_rot = apply_rotary_emb(q_rot, k_rot, freqs_cis)
        q = q_rot.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k_rot.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, v)
        
        # Reshape and project back
        # (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        return output


class LLaMAFFN(nn.Module):
    """
    LLaMA Feed-Forward Network
    Uses SwiGLU activation (Swish-Gated Linear Unit) as in the paper
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU to input tensor
        SwiGLU(x) = Swish(xW) ⊗ (xV)
        """
        # Calculate Swish-Gated Linear Unit
        swish = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        
        # Element-wise multiplication and down projection
        return self.down_proj(swish * up)


class LLaMATransformerBlock(nn.Module):
    """
    LLaMA Transformer Block consisting of attention and feed-forward layers
    with pre-normalization
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Pre-normalization layers
        self.attention_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        
        # Self-attention and feed-forward modules
        self.attention = LLaMAAttention(hidden_size, num_heads, head_dim)
        self.feed_forward = LLaMAFFN(hidden_size, intermediate_size)
    
    def forward(
        self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply attention with pre-normalization and residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        
        # Apply feed-forward with pre-normalization and residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


class LLaMAModel(nn.Module):
    """
    LLaMA (Large Language Model Meta AI) architecture
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_size: int = 11008,
        max_seq_len: int = 2048,
        head_dim: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        
        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                LLaMATransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    head_dim=head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final normalization layer
        self.norm = RMSNorm(hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Precompute the frequencies for rotary position embeddings
        self.freqs_cis = precompute_freqs_cis(
            head_dim, 
            max_seq_len * 2,  # Extra buffer
        )
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Make sure input sequence isn't too long
        assert seq_len <= self.max_seq_len, f"Input sequence length ({seq_len}) exceeds maximum length ({self.max_seq_len})"
        
        # Get token embeddings
        h = self.tok_embeddings(input_ids)
        
        # Get the correct part of the pre-computed frequencies
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        
        # Create causal mask if not provided
        if attention_mask is None:
            # Create causal mask (lower triangular)
            mask = torch.full(
                (seq_len, seq_len), 
                float("-inf"), 
                device=h.device,
            )
            mask = torch.triu(mask, diagonal=1) 
        else:
            # Use the provided mask (extended for multi-head attention)
            mask = attention_mask
            
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # Apply final normalization
        h = self.norm(h)
        
        # Project to vocabulary
        logits = self.output(h)
        
        return logits
