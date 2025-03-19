import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from modules import (
    TransformerBlock, 
    CrossAttentionBlock, 
    SinusoidalPositionalEmbedding
)
from config import WhisperConfig

class AudioEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        
        # Convolutional front-end for processing mel spectrogram
        self.conv1 = nn.Conv1d(config.n_mels, config.n_embd, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, stride=2, padding=1)
        
        # Positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(config.n_embd, config.n_audio_ctx)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.n_embd, config.n_head)
            for _ in range(config.n_audio_layer)
        ])
        
        # Layer norm for final output
        self.ln_out = nn.LayerNorm(config.n_embd)
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Batch of mel spectrograms [batch_size, n_mels, time]
        Returns:
            Audio encodings [batch_size, time, n_embd]
        """
        # Apply convolutional front-end
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # [batch_size, time, n_embd]
        
        # Add positional embeddings
        pos_emb = self.positional_embedding(x)
        x = x + pos_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Final layer norm
        x = self.ln_out(x)
        
        return x

class TextDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.n_vocab, config.n_embd)
        
        # Positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(config.n_embd, config.n_text_ctx)
        
        # Transformer blocks with cross-attention
        self.transformer_blocks = nn.ModuleList([
            CrossAttentionBlock(config.n_embd, config.n_head)
            for _ in range(config.n_text_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.n_vocab, bias=False)
        
        # Initialize the output projection with token embeddings for better convergence
        self.output_projection.weight = self.token_embedding.weight
        
    def forward(
        self, 
        tokens: torch.Tensor, 
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tokens: Batch of token sequences [batch_size, seq_len]
            audio_features: Audio encodings from encoder [batch_size, audio_len, n_embd]
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        B, T = tokens.size()
        
        # Create causal mask for decoder self-attention
        # For each position, allow attending only to previous positions
        causal_mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(tokens.device)
        
        # Token embeddings
        token_emb = self.token_embedding(tokens)  # [B, T, n_embd]
        
        # Add positional embeddings
        pos_emb = self.positional_embedding(token_emb)  # [1, T, n_embd]
        x = token_emb + pos_emb
        
        # Apply transformer blocks with cross-attention to audio
        for block in self.transformer_blocks:
            x = block(x, causal_mask, audio_features)
            
        # Final layer norm
        x = self.ln_out(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [B, T, vocab_size]
        
        return logits
        
class WhisperModel(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)
        
    def forward(
        self, 
        mel: torch.Tensor, 
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mel: Batch of mel spectrograms [batch_size, n_mels, time]
            tokens: Batch of token sequences [batch_size, seq_len]
        Returns:
            Logits for token prediction [batch_size, seq_len, vocab_size]
        """
        # Encode audio
        audio_features = self.encoder(mel)
        
        # Decode and get logits
        logits = self.decoder(tokens, audio_features)
        
        return logits
        
    def generate(
        self, 
        mel: torch.Tensor, 
        tokenizer, 
        max_length: int = 448, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate text from audio using greedy decoding.
        
        Args:
            mel: Batch of mel spectrograms [batch_size, n_mels, time]
            tokenizer: Tokenizer for decoding
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token ids and text
        """
        # Get device
        device = mel.device
        
        # Initialize with BOS token
        batch_size = mel.shape[0]
        generated = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        
        # Encode audio (only need to do this once)
        audio_features = self.encoder(mel)
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Get logits for the next token
            logits = self.decoder(generated, audio_features)
            
            # Focus on the last token's prediction
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample or greedy select
            if temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if we generated an EOS token
            if (next_token == tokenizer.special_tokens["<EOS>"]).all():
                break
        
        # Decode the generated tokens
        decoded_text = tokenizer.decode(generated[0].cpu().tolist())
        
        return generated, decoded_text
