# PyTorch Implementation of the Transformer Model

This repository contains a clean implementation of the Transformer architecture from the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Structure

- `model/transformer.py`: Main Transformer model
- `model/encoder.py`: Transformer encoder stack
- `model/decoder.py`: Transformer decoder stack
- `model/attention.py`: Multi-head attention mechanism
- `model/utils.py`: Utility modules (positional encoding, feed-forward networks)
- `main.py`: Example usage

## Model Architecture

The Transformer model consists of:

- **Encoder**: Stack of N encoder layers, each containing:
  - Multi-Head Attention
  - Position-wise Feed Forward Network
  - Layer Normalization and Residual Connections

- **Decoder**: Stack of N decoder layers, each containing:
  - Masked Multi-Head Attention
  - Multi-Head Attention over encoder output
  - Position-wise Feed Forward Network
  - Layer Normalization and Residual Connections

- **Additional Components**:
  - Positional Encoding
  - Input/Output Embeddings
  - Final Linear layer and Softmax

## Usage

Basic usage example:

```python
from model import Transformer

# Create model
model = Transformer(
    src_vocab=10000,  # Size of source vocabulary
    tgt_vocab=10000,  # Size of target vocabulary
    d_model=512,      # Model dimension
    n_layers=6,       # Number of encoder/decoder layers
    d_ff=2048,        # Feed-forward dimension
    h=8,              # Number of attention heads
    dropout=0.1       # Dropout rate
)

# Forward pass (with proper masking)
output = model(src_tokens, tgt_tokens, src_mask, tgt_mask)
```

See `main.py` for a more detailed example.

## Requirements

- PyTorch 1.0+
- Python 3.6+
