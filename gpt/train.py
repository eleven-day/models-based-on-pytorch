import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse

class SimpleTextDataset(Dataset):
    """A simple dataset for demonstration purposes"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a block of tokens
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
    
    return total_loss / len(train_loader)

def generate_text(model, start_tokens, vocab_size, max_length=100, temperature=1.0, device='cuda'):
    model.eval()
    current_tokens = start_tokens.to(device)
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            outputs = model(current_tokens)
            
            # Get the next token prediction (from the last position)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with the growing sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
    return current_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=64)
    args = parser.parse_args()
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # For demonstration, let's create some random data
    vocab_size = 10000
    data = np.random.randint(0, vocab_size, size=100000)
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(data, args.block_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize the model
    model = GPT(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        heads=args.heads,
        max_length=args.block_size,
        ff_dim=4 * args.embed_size,
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        avg_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss}")
    
    # Generate some text
    start_sequence = torch.tensor([[42, 43, 44, 45]], dtype=torch.long)  # Random start tokens
    generated = generate_text(model, start_sequence, vocab_size, device=device)
    print("Generated sequence:")
    print(generated)
    
    # Save the model
    torch.save(model.state_dict(), "gpt_model.pt")
    print("Model saved to gpt_model.pt")

if __name__ == "__main__":
    main()
