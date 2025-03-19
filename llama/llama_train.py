import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import os
from typing import Optional
from llama_model import LLaMAModel


class SimpleTokenDataset(Dataset):
    """Simple dataset for training with tokenized data"""
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]  # Shifted by 1 for next token prediction
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def setup_distributed(rank: int, world_size: int):
    """Set up distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], 
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            
            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Perform optimization step after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimize
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_batches += 1
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
    
    return total_loss / total_batches


def train(rank, world_size, args):
    """Main training function"""
    # Set up distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = LLaMAModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
        head_dim=args.hidden_size // args.num_heads,
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create dummy data for testing (replace with real data in production)
    data = np.random.randint(0, args.vocab_size, size=(args.data_size,))
    
    # Create dataset and dataloader
    dataset = SimpleTokenDataset(data, args.max_seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(dataloader) * args.epochs
    )
    
    # Create gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Main training loop
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        # Train one epoch
        avg_loss = train_epoch(
            model, 
            dataloader, 
            optimizer, 
            scheduler, 
            scaler, 
            device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        print(f"Rank {rank}, Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint (only on rank 0)
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"llama_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Clean up distributed
    dist.destroy_process_group()


def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description="LLaMA Model Training")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=2730, help="Intermediate size in FFN")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--data_size", type=int, default=100000, help="Size of dummy data for testing")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Start distributed training
    torch.multiprocessing.spawn(train, args=(args.world_size, args), nprocs=args.world_size)


if __name__ == "__main__":
    main()
