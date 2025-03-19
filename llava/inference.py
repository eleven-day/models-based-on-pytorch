import argparse
import torch
from PIL import Image
from llava import LLaVA
from utils import LLaVAProcessor, Conversation

def load_model(checkpoint_path, vision_model, language_model, device):
    # Initialize model
    model = LLaVA(
        vision_tower=vision_model,
        llm=language_model,
        freeze_vision=True,
        use_lora=True
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def generate(model, processor, image_path, prompt, device="cuda", max_length=512, temperature=0.7):
    # Create conversation
    conversation = Conversation()
    conversation.add_message("user", f"<image> {prompt}", image=image_path)
    
    # Process inputs
    inputs = processor.process_conversation(conversation, max_length=max_length)
    
    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    images = inputs["images"].to(device) if inputs["images"] is not None else None
    
    # Generate response
    with torch.no_grad():
        gen_tokens = model.llm.model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode response
    response = processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    response = response.split("[ASSISTANT]")[-1].strip()
    
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with LLaVA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--language-model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = LLaVAProcessor(args.vision_model, args.language_model)
    
    # Load model
    model = load_model(args.checkpoint, args.vision_model, args.language_model, args.device)
    
    # Generate response
    response = generate(
        model, 
        processor, 
        args.image, 
        args.prompt, 
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(f"Prompt: {args.prompt}")
    print(f"Response: {response}")
