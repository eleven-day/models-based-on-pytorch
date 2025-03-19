import torch
import torch.nn as nn

class VisionProjector(nn.Module):
    """Projects image features to the language model's embedding space."""
    def __init__(self, vision_hidden_size, text_hidden_size, projection_dim=768):
        super().__init__()
        self.linear1 = nn.Linear(vision_hidden_size, projection_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(projection_dim, text_hidden_size)
        self.layer_norm = nn.LayerNorm(text_hidden_size)
    
    def forward(self, image_features):
        """
        Args:
            image_features: [batch_size, num_patches, vision_hidden_size]
        Returns:
            projected_features: [batch_size, num_patches, text_hidden_size]
        """
        hidden = self.linear1(image_features)
        hidden = self.act(hidden)
        hidden = self.linear2(hidden)
        projected_features = self.layer_norm(hidden)
        return projected_features
