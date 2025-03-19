# Transformer Models Implementation with PyTorch

This repository contains PyTorch reimplementations of popular transformer-based models including:

- **GPT** (Generative Pre-trained Transformer)
- **LLAMA** (Large Language Model Meta AI)
- **Whisper** (Automatic Speech Recognition)
- **Transformers** (Base architecture)
- **Mistral-MoE** (Mixture of Experts)
- **LLaVA** (Large Language and Vision Assistant)


## Installation

```bash
git clone https://github.com/eleven-day/transformer-based-on-pytorch.git
cd transformer-based-on-pytorch
pip install -r requirements.txt
```

## Models

### 1. GPT (Generative Pre-trained Transformer)
Implementation of the GPT architecture based on the "Improving Language Understanding by Generative Pre-Training" paper by Radford et al.

### 2. LLAMA (Large Language Model Meta AI)
Implementation of the LLAMA architecture based on the "LLaMA: Open and Efficient Foundation Language Models" paper by Meta AI.

### 3. Whisper
Implementation of the Whisper model for automatic speech recognition based on the "Robust Speech Recognition via Large-Scale Weak Supervision" paper by OpenAI.

### 4. Transformers
Base implementation of the Transformer architecture based on the "Attention Is All You Need" paper by Vaswani et al.

### 5. Mistral-MoE (Mixture of Experts)
Implementation of the Mistral Mixture of Experts architecture for efficient large language models.

### 6. LLaVA (Large Language and Vision Assistant)
Implementation of the LLaVA architecture, a large language and vision model that connects vision encoders with language models.

## Usage Examples

Each model has a corresponding example file in the `examples/` directory. Here's a quick example:

```python
from models.gpt import GPT
from configs.model_configs import GPTConfig

# Initialize the model
config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
model = GPT(config)

# Generate text
output = model.generate("Once upon a time", max_length=50)
print(output)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.0+
- NumPy
- tqdm

See `requirements.txt` for complete dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

## Citations

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training"
- Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision"
- Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
