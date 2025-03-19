import torch
import argparse
from config import WhisperConfig
from model import WhisperModel
from audio import prepare_audio_input
from tokenizer import SimpleTokenizer

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Initialize model configuration
    config = WhisperConfig.get_model_config(args.model_size)
    print(f"Initializing {args.model_size} Whisper model")
    
    # Initialize model
    model = WhisperModel(config)
    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load audio and prepare input
    print(f"Processing audio file: {args.audio_file}")
    mel = prepare_audio_input(args.audio_file, n_mels=config.n_mels)
    mel = mel.to(device)
    
    # Run inference
    print("Generating transcription...")
    with torch.no_grad():
        _, transcription = model.generate(mel, tokenizer, temperature=args.temperature)
    
    print("\nTranscription:")
    print(transcription)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper model inference")
    parser.add_argument("--audio-file", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model-size", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Model size")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    args = parser.parse_args()
    main(args)
