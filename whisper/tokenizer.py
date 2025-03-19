import torch
import re
from typing import List, Dict, Optional, Union

class SimpleTokenizer:
    def __init__(self):
        # Basic vocabulary - in a real implementation, this would be much larger
        # and include special tokens for task-specific prompts
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        
        # Basic English characters and common punctuation
        chars = " abcdefghijklmnopqrstuvwxyz"
        chars += "0123456789"
        chars += ".,?!-:;\"'()[]{}@#$%^&*+=/\\|<>~`"
        
        # Create vocabulary
        self.vocab = {c: i + len(self.special_tokens) for i, c in enumerate(chars)}
        self.vocab.update(self.special_tokens)
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Vocabulary size
        self.vocab_size = len(self.vocab)
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token ids."""
        # Simple preprocessing
        text = text.lower().strip()
        
        # Tokenize by characters (simplistic approach)
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.special_tokens["<UNK>"])
                
        # Add BOS and EOS tokens
        tokens = [self.special_tokens["<BOS>"]] + tokens + [self.special_tokens["<EOS>"]]
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token ids to text."""
        # Remove special tokens
        special_ids = set(self.special_tokens.values())
        filtered_ids = [id for id in token_ids if id not in special_ids]
        
        # Convert to characters
        text = "".join([self.id_to_token.get(id, "") for id in filtered_ids])
        return text
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a batch of texts to a padded tensor."""
        batch_tokens = [self.encode(text) for text in texts]
        
        # Apply max_length if specified
        if max_length is not None:
            batch_tokens = [tokens[:max_length-1] + [self.special_tokens["<EOS>"]] 
                           if len(tokens) > max_length else tokens 
                           for tokens in batch_tokens]
        
        # Find max length in batch
        max_len = max(len(tokens) for tokens in batch_tokens)
        
        # Pad sequences
        padded_batch = [tokens + [self.special_tokens["<PAD>"]] * (max_len - len(tokens)) 
                        for tokens in batch_tokens]
        
        return torch.tensor(padded_batch)
    
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return self.vocab_size
