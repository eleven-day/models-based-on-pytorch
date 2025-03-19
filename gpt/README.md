# PyTorch GPT Implementation

This project is a PyTorch implementation of a GPT (Generative Pre-trained Transformer) architecture, which is a decoder-only transformer model.

## Overview

The implementation consists of the following components:

1. **Multi-Head Self-Attention**: Allows the model to focus on different parts of the input sequence
2. **Position-wise Feed-Forward Networks**: Non-linear transformations applied to each position
3. **Positional Encodings**: Adds information about token positions in the sequence
4. **Transformer Blocks**: Combines attention and feed-forward layers with residual connections and layer normalization
5. **Complete GPT Model**: Stacks multiple transformer blocks with token and positional embeddings

## Files

- `model.py`: Contains the implementation of the GPT model architecture
- `train.py`: Example script for training the model and generating text

## Usage

### Training a model

```bash
python train.py --batch_size 32 --epochs 10 --embed_size 256 --num_layers 6 --heads 8 --block_size 64
```

### Model Parameters

- `vocab_size`: Size of the vocabulary
- `embed_size`: Dimension of the embeddings
- `max_length`: Maximum sequence length
- `num_layers`: Number of transformer blocks
- `heads`: Number of attention heads
- `ff_dim`: Dimension of the feed-forward network
- `dropout`: Dropout rate for regularization

## Architecture Details

1. **Token and Positional Embeddings**: Converts input tokens to vectors and adds positional information
2. **Transformer Blocks**: Each block has:
   - Multi-head self-attention mechanism
   - Layer normalization
   - Position-wise feed-forward network
3. **Final Layer Norm and Output Projection**: Projects to vocabulary size for token prediction

## Requirements

- PyTorch 1.7+
- NumPy
- Python 3.6+

## Notes

This implementation follows the original GPT architecture but can be customized by adjusting hyperparameters. For training on real data, you would need to replace the random data with a proper dataset and tokenizer.
