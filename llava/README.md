# LLaVA Implementation with PyTorch

This repository contains a PyTorch implementation of the LLaVA (Language-and-Vision Assistant) model, which combines vision and language capabilities to create a multimodal AI assistant.

## Architecture

The LLaVA model consists of three main components:
1. Vision Encoder: Uses CLIP ViT to encode images into feature representations
2. Projector: Maps visual features to the language model's embedding space
3. Language Model: Pretrained LLM (e.g., Llama-2) for text generation

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py \
  --vision-model openai/clip-vit-large-patch14 \
  --language-model meta-llama/Llama-2-7b-hf \
  --data-dir path/to/dataset \
  --save-path checkpoints \
  --batch-size 4 \
  --epochs 3
```

## Inference

```bash
python inference.py \
  --checkpoint checkpoints/llava-checkpoint-3.pt \
  --image path/to/image.jpg \
  --prompt "What can you see in this image?"
```

## Implementation Details

- The vision encoder uses a pretrained CLIP ViT model
- The language model uses a pretrained Llama-2 model with LoRA fine-tuning
- The projector maps the visual features to the language model's embedding space
