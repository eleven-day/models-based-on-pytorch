import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionEncoder
from .projector import VisionProjector
from .llm import LanguageModel

class LLaVA(nn.Module):
    def __init__(self, 
                 vision_tower="openai/clip-vit-large-patch14", 
                 llm="meta-llama/Llama-2-7b-hf",
                 freeze_vision=True,
                 freeze_llm=False,
                 use_lora=True,
                 lora_rank=8,
                 projection_dim=768):
        super().__init__()
        
        # Initialize vision encoder
        self.vision_encoder = VisionEncoder(vision_tower, freeze=freeze_vision)
        
        # Initialize language model
        self.llm = LanguageModel(llm, lora_rank=lora_rank, use_lora=use_lora)
        if freeze_llm and not use_lora:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Initialize projector
        vision_hidden_size = self.vision_encoder.get_hidden_size()
        text_hidden_size = self.llm.get_hidden_size()
        self.projector = VisionProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            projection_dim=projection_dim
        )
    
    def forward(self, images=None, input_ids=None, labels=None, attention_mask=None):
        image_features = None
        
        # Process images if provided
        if images is not None:
            # Get image features from vision encoder
            image_features = self.vision_encoder(images)
            # Project image features to language space
            image_features = self.projector(image_features)
        
        # Prepare inputs for language model
        outputs = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=image_features,
            labels=labels,
            return_dict=True
        )
        
        return outputs
