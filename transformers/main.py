import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer

def example_usage():
    # Hyperparameters
    d_model = 512
    n_layers = 6
    d_ff = 2048
    h = 8
    dropout = 0.1
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    
    # Create model
    model = Transformer(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_ff=d_ff,
        h=h,
        dropout=dropout
    )
    
    # Example input
    src = torch.randint(1, src_vocab_size, (2, 10))  # (batch_size, seq_len)
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))   # (batch_size, seq_len)
    
    # Create masks
    src_mask, tgt_mask = Transformer.create_masks(src, tgt[:, :-1])
    
    # Forward pass
    output = model(src, tgt[:, :-1], src_mask, tgt_mask)
    
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt[:, :-1].shape}")
    print(f"Output shape: {output.shape}")
    
    # Model training would involve:
    # 1. Define loss function (CrossEntropyLoss for sequence generation)
    # 2. Define optimizer
    # 3. Training loop with forward/backward passes
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Example of a single training step
    optimizer.zero_grad()
    output = model(src, tgt[:, :-1], src_mask, tgt_mask)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    example_usage()
