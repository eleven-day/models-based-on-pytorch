import torch
import argparse
import os
from llama_model import LLaMAModel


def generate_text(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
    top_k=40,
    device="cuda"
):
    """
    Generate text using a trained LLaMA model
    
    Args:
        model: The LLaMA model
        input_ids: Input token IDs (prompt)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter
        device: Device to run generation on
        
    Returns:
        Generated token IDs
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    # Start with the input_ids (prompt)
    generated = input_ids
    past_key_values = None
    
    # Generate tokens auto-regressively
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Only use the last token as input for efficiency
            outputs = model(generated[:, -min(model.max_seq_len, generated.shape[1]):])
            
            # Get the next token predictions
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top_k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_values)
            
            # Apply top_p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')
            
            # Get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            generated = torch.cat([generated, next_token], dim=1)
            
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    
    # Model parameters (should match the checkpoint)
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=2730, help="Intermediate size in FFN")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the model
    model = LLaMAModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
        head_dim=args.hidden_size // args.num_heads,
    ).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from: {args.checkpoint}")
    
    # Convert prompt to dummy token IDs (in a real app, use a proper tokenizer)
    dummy_tokens = [1, 2, 3, 4, 5]  # Replace with actual tokenization of the prompt
    input_ids = torch.tensor([dummy_tokens], dtype=torch.long).to(device)
    
    print(f"Generating text with prompt: {args.prompt}")
    
    # Generate text
    output_ids = generate_text(
        model,
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
    )
    
    # Print the result (in a real app, use a proper detokenizer)
    print("\nGenerated token IDs:")
    print(output_ids.tolist()[0])
    print("\n(In a real application, these would be converted back to text using a tokenizer)")


if __name__ == "__main__":
    main()
