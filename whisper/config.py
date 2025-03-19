import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class WhisperConfig:
    # Model dimensions
    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_vocab: int = 51864
    n_text_ctx: int = 448

    # Architecture dimensions
    n_embd: int = 512
    n_head: int = 8
    n_audio_layer: int = 4
    n_text_layer: int = 4

    @classmethod
    def get_model_config(cls, size: str = "base") -> "WhisperConfig":
        """Get model configuration based on size."""
        configs = {
            "tiny": WhisperConfig(n_embd=384, n_head=6, n_audio_layer=4, n_text_layer=4),
            "base": WhisperConfig(n_embd=512, n_head=8, n_audio_layer=6, n_text_layer=6),
            "small": WhisperConfig(n_embd=768, n_head=12, n_audio_layer=12, n_text_layer=12),
            "medium": WhisperConfig(n_embd=1024, n_head=16, n_audio_layer=24, n_text_layer=24),
            "large": WhisperConfig(n_embd=1280, n_head=20, n_audio_layer=32, n_text_layer=32),
        }
        return configs.get(size, configs["base"])
