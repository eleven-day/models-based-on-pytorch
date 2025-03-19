import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm

from llava import LLaVA
from utils import LLaVAProcessor, Conversation

# Simple image-text dataset class
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=512):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.data = self._load_data()
    
    def _load_data(self):
        # This is a placeholder - in a real scenario, you'd load your dataset
        # For example, reading from a JSON file that contains image paths and instructions
        # Return a list of (image_path, instruction, response) tuples
        return [("path/to/image1.jpg", "What's in this image?", "A dog playing in the park.")]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, instruction, response = self.data[idx]
        
        # Create conversation
        conversation = Conversation()
        conversation.add_message("user", f"<image> {instruction}", image=image_path)
        conversation.add_message("assistant", response)
        
        # Process conversation
        processed = self.processor.process_conversation(conversation, max_length=self.max_length)
        
        # Create labels for training
        labels = processed["input_ids"].clone()
        
        # Set labels for non-response tokens to -100 (ignored in loss calculation)
        response_tokens = self.processor.process_text(response)["input_ids"][0]
        response_length = len(response_tokens)
        
        # Find where the response starts
        labels[0, :-response_length] = -100
        
        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "images": processed["images"],
            "labels": labels
        }

def train(args):
    # Initialize processor
    processor = LLaVAProcessor(args.vision_model, args.language_model)
    
    # Initialize model
    model = LLaVA(
        vision_tower=args.vision_model,
        llm=args.language_model,
        freeze_vision=True,
        use_lora=True,
        lora_rank=args.lora_rank
    ).to(args.device)
    
    # Create dataset and dataloader
    dataset = MultimodalDataset(args.data_dir, processor, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    num_training_steps = len(dataloader) * args.epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            images = batch["images"].to(args.device) if batch["images"] is not None else None
            labels = batch["labels"].to(args.device)
            
            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            lr_scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Log epoch stats
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            model_path = os.path.join(args.save_path, f"llava-checkpoint-{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            print(f"Checkpoint saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaVA model")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--language-model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with training data")
    parser.add_argument("--save-path", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    train(args)
