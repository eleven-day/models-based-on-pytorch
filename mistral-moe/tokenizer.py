from typing import List, Dict, Optional, Union
import os
import json
import torch
import re
from sentencepiece import SentencePieceProcessor

class MistralTokenizer:
    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ):
        self.vocab_file = vocab_file
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Special token ids
        self.bos_token_id = self.sp_model.piece_to_id(bos_token)
        self.eos_token_id = self.sp_model.piece_to_id(eos_token)
        self.pad_token_id = self.sp_model.piece_to_id(pad_token)
        self.unk_token_id = self.sp_model.piece_to_id(unk_token)
        
        self.vocab_size = self.sp_model.GetPieceSize()
        
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        tokens = self.sp_model.encode(text)
        
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids 
                         if token_id not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
            
        return self.sp_model.decode(token_ids)
    
    def batch_encode(
        self, 
        texts: List[str], 
        add_bos: bool = True, 
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        batch_token_ids = []
        
        for text in texts:
            token_ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            
            if truncation and max_length is not None and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                
            batch_token_ids.append(token_ids)
            
        if padding:
            max_len = max(len(ids) for ids in batch_token_ids)
            padded_token_ids = []
            attention_mask = []
            
            for token_ids in batch_token_ids:
                padding_length = max_len - len(token_ids)
                padded_token_ids.append(token_ids + [self.pad_token_id] * padding_length)
                attention_mask.append([1] * len(token_ids) + [0] * padding_length)
        else:
            padded_token_ids = batch_token_ids
            attention_mask = [[1] * len(ids) for ids in batch_token_ids]
            
        result = {
            "input_ids": padded_token_ids,
            "attention_mask": attention_mask
        }
        
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
            
        return result
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        vocab_file = os.path.join(model_path, "tokenizer.model")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Tokenizer model file not found at {vocab_file}")
        
        return cls(vocab_file)
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the SentencePiece model
        output_vocab_file = os.path.join(save_directory, "tokenizer.model")
        with open(self.vocab_file, 'rb') as f:
            with open(output_vocab_file, 'wb') as out_f:
                out_f.write(f.read())
                
        # Save tokenizer config
        config = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
