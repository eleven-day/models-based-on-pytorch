import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

class VisionEncoder(nn.Module):
    def __init__(self, vision_tower="openai/clip-vit-large-patch14", freeze=True):
        super().__init__()
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        self.vision_hidden_size = self.vision_tower.config.hidden_size
        
        # Freeze vision tower if specified
        if freeze:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: [batch_size, num_channels, height, width]
        Returns:
            image_features: [batch_size, num_patches, hidden_size]
        """
        vision_outputs = self.vision_tower(images)
        image_features = vision_outputs.last_hidden_state
        return image_features
    
    def get_hidden_size(self):
        return self.vision_hidden_size
