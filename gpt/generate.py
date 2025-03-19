import torch
import argparse
from model import GPT

def generate(
    model,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    device='cuda'
):
    """
    Generate text using a trained GPT model
    
    Args:
        model: The GPT model
        prompt: List of tokens to begin generation with
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_k: If set, limits sampling to the top k most likely tokens
        device: Device to run the model on ('cuda' or 'cpu')
    
    Returns:
        Generated token sequence
    """
    model.eval()
    prompt_tensor = torch.tensor([prompt], dtype=torch.long).to(device)
    
    # Start with the prompt
    generated = prompt_tensor
    
    # Generate max_new_tokens new tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass to get next token predictions
            # Only use the context that fits within model's max sequence length
            context = generated[:, -model.max_length:]
            outputs = model(context)
            
            # Get the next token logits from the last position
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated.tolist()[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter')
    parser.add_argument('--vocab_size', type=int, required=True, help='Size of the vocabulary')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the model
    model = GPT(
        vocab_size=args.vocab_size,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        heads=args.heads,
        max_length=args.max_length
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    
    # Convert prompt to token IDs (this is a placeholder - in a real application,
    # you would use a proper tokenizer here)
    prompt_ids = [42, 43, 44]  # Dummy values - replace with actual token IDs
    if args.prompt:
        print(f"Starting with prompt: {args.prompt}")
        # In a real application, you would tokenize args.prompt here
    
    # Generate text
    generated_ids = generate(
        model, 
        prompt_ids, 
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print("\nGenerated token IDs:")
    print(generated_ids)
    print("\nIn a real application, these would be converted back to text using a tokenizer.")

if __name__ == "__main__":
    main()
