import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from typing import Dict, List, Optional, Tuple

from mistral_moe import MistralMoEConfig, MistralMoEForCausalLM
from tokenizer import MistralTokenizer
from dataset import load_dataset_from_files, CausalLanguageModelingCollator
from trainer import MoETrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mistral-8x7B MoE model")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--intermediate_size", type=int, default=14336, help="Intermediate size for MLP")
    parser.add_argument("--num_hidden_layers", type=int, default=32, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=8, help="Number of key/value heads (for GQA)")
    parser.add_argument("--max_position_embeddings", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts in MoE")
    parser.add_argument("--num_experts_per_token", type=int, default=2, help="Number of experts per token")
    parser.add_argument("--sliding_window", type=int, default=4096, help="Sliding window attention size")
    
    # Training configuration
    parser.add_argument("--data_path", type=str, required=True, help="Path to text files")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer files")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    
    return parser.parse_args()

def create_model(args):
    """Create Mistral MoE model from args"""
    config = MistralMoEConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        sliding_window=args.sliding_window,
    )
    
    model = MistralMoEForCausalLM(config)
    return model

def create_optimizer_and_scheduler(model, args, num_training_steps):
    """Create optimizer and learning rate scheduler"""
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Create scheduler with linear warmup
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - args.warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MistralTokenizer.from_pretrained(args.tokenizer_path)
    
    # Create data collator
    data_collator = CausalLanguageModelingCollator(tokenizer, max_length=args.max_length)
    
    # Load datasets
    print(f"Loading datasets from {args.data_path}...")
    data_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) 
                  if os.path.isfile(os.path.join(args.data_path, f)) and f.endswith('.txt')]
    
    train_dataset, eval_dataset = load_dataset_from_files(data_files, tokenizer, max_length=args.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4
    )
    
    # Calculate number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.num_epochs * num_update_steps_per_epoch
    
    # Create model
    print("Creating model...")
    model = create_model(args)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, num_training_steps)
    
    # Create trainer
    trainer = MoETrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.load_model(args.resume_from_checkpoint)
    
    # Train model
    print("Starting training...")
    training_results = trainer.train()
    
    print(f"Training completed. Results: {training_results}")
    
if __name__ == "__main__":
    main()
