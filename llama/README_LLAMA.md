# LLaMA Model Implementation in PyTorch

This is an implementation of the LLaMA (Large Language Model Meta AI) architecture using PyTorch. LLaMA is an efficient transformer-based language model designed by Meta AI.

## Architecture Details

The LLaMA architecture includes several key modifications to the standard transformer architecture:

1. **Pre-normalization**: Using RMSNorm before each sub-layer instead of post-normalization
2. **Rotary Positional Embeddings (RoPE)**: Applied to each layer's queries and keys
3. **SwiGLU Activation**: In the feed-forward network instead of standard ReLU
4. **No biases**: Linear layers do not use bias terms

## Files

- `llama_model.py`: Core implementation of the LLaMA model architecture
- `llama_train.py`: Script for distributed training of the model
- `llama_generate.py`: Utility for generating text with a trained model

## Key Components

### RMSNorm
Root Mean Square Layer Normalization, a simplified and more efficient alternative to LayerNorm.

### Rotary Position Embeddings (RoPE)
Enables better relative position modeling by encoding relative positions through a rotation matrix that is multiplied with the query and key vectors.

### SwiGLU Activation
A variant of the GLU (Gated Linear Unit) that uses the SiLU (Sigmoid Linear Unit, also known as Swish) activation function, providing better performance than ReLU.

## Usage

### Training the Model

```bash
python llama_train.py \
    --vocab_size 32000 \
    --hidden_size 1024 \
    --num_layers 12 \
    --num_heads 16 \
    --intermediate_size 2730 \
    --max_seq_len 512 \
    --batch_size 8 \
    --epochs 3 \
    --world_size 2 \
    --output_dir ./checkpoints
```

This will train a LLaMA model using distributed data parallelism across 2 GPUs. The model parameters in this example are smaller than the full LLaMA model for easier experimentation.

### Generating Text

```bash
python llama_generate.py \
    --checkpoint ./checkpoints/llama_model_epoch_3.pt \
    --vocab_size 32000 \
    --hidden_size 1024 \
    --num_layers 12 \
    --num_heads 16 \
    --intermediate_size 2730 \
    --max_seq_len 512 \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 40
```

This will generate up to 100 new tokens using the specified checkpoint.

## Model Configurations

The LLaMA paper introduced several model sizes. Here are the parameters for reference:

| Model | Parameters | Hidden Size | Layers | Heads | Head Dim | Intermediate Size |
|-------|------------|-------------|--------|-------|---------|-------------------|
| LLaMA-7B | 6.7B | 4096 | 32 | 32 | 128 | 11008 |
| LLaMA-13B | 13.0B | 5120 | 40 | 40 | 128 | 13824 |
| LLaMA-33B | 32.5B | 6656 | 60 | 52 | 128 | 17920 |
| LLaMA-65B | 65.2B | 8192 | 80 | 64 | 128 | 22016 |

## Implementation Notes

1. The implementation uses PyTorch's distributed data parallelism (DDP) for efficient multi-GPU training
2. Mixed precision training is enabled with automatic mixed precision (AMP)
3. Gradient accumulation is supported to simulate larger batch sizes
4. For text generation, top-k and nucleus (top-p) sampling are implemented for better text quality

## Requirements

- PyTorch 1.10+
- CUDA-capable GPU for efficient training
- 8GB+ GPU memory for smaller models (more for larger configurations)
- Python 3.8+

## Limitations

- This implementation does not include a tokenizer; in a real-world scenario, you would need to integrate a tokenizer
- The provided sample data is randomly generated; for actual training, you should replace it with proper tokenized text
- The hyperparameters need to be tuned for specific tasks and datasets

## References

1. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
2. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
3. Shazeer, N. (2020). "GLU Variants Improve Transformer." [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
