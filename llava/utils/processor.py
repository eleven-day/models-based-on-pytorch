import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

class LLaVAProcessor:
    def __init__(self, vision_tower="openai/clip-vit-large-patch14", tokenizer="meta-llama/Llama-2-7b-hf"):
        self.image_processor = AutoProcessor.from_pretrained(vision_tower)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, 
            padding_side="right",
            use_fast=False,
        )
        # Add image token for multimodal inputs
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image_token>"]})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image_token>")
    
    def process_text(self, text, max_length=512):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
    
    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]
    
    def process_conversation(self, conversation, max_length=512):
        prompt = conversation.get_prompt()
        images = conversation.extract_images()
        
        # Process text
        text_inputs = self.process_text(prompt, max_length=max_length)
        
        # Process images if any
        image_tensor = None
        if images:
            image_tensor = self.process_image(images[0])  # Using first image for simplicity
            
        return {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "images": image_tensor
        }
