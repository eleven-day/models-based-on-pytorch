import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
from typing import List, Dict, Optional, Union, Any
import json

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer.batch_encode(
            [text], 
            max_length=self.max_length,
            padding=False,
            truncation=True,
            add_bos=True,
            add_eos=True,
            return_tensors=None
        )
        
        input_ids = torch.tensor(encodings["input_ids"][0], dtype=torch.long)
        attention_mask = torch.tensor(encodings["attention_mask"][0], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For causal language modeling
        }


class CausalLanguageModelingCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad to max length in batch
        max_len = min(self.max_length, max([len(ids) for ids in input_ids]))
        
        # Truncate and pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for ids, mask, lbl in zip(input_ids, attention_mask, labels):
            if len(ids) > max_len:
                padded_input_ids.append(ids[:max_len])
                padded_attention_mask.append(mask[:max_len])
                padded_labels.append(lbl[:max_len])
            else:
                padding_length = max_len - len(ids)
                padded_input_ids.append(ids.tolist() + [self.tokenizer.pad_token_id] * padding_length)
                padded_attention_mask.append(mask.tolist() + [0] * padding_length)
                padded_labels.append(lbl.tolist() + [-100] * padding_length)  # -100 is ignored in loss calculation
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels)
        }


def load_dataset_from_files(file_paths, tokenizer, max_length=1024, split_ratio=0.9):
    """Load text data from files and split into train/val"""
    all_texts = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            all_texts.extend([t.strip() for t in texts if t.strip()])
    
    # Shuffle and split
    random.shuffle(all_texts)
    split_idx = int(len(all_texts) * split_ratio)
    
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    
    return train_dataset, val_dataset


def load_dataset_from_json(file_path, tokenizer, max_length=1024, split_ratio=0.9, text_key="text"):
    """Load text data from JSON file and split into train/val"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_texts = []
    
    # Extract texts based on format
    if isinstance(data, list):
        if isinstance(data[0], str):
            all_texts = data
        elif isinstance(data[0], dict) and text_key in data[0]:
            all_texts = [item[text_key] for item in data if text_key in item]
        else:
            raise ValueError(f"Cannot find text data with key '{text_key}' in JSON")
    elif isinstance(data, dict) and text_key in data:
        all_texts = data[text_key]
    else:
        raise ValueError(f"Unsupported JSON format or cannot find text data with key '{text_key}'")
    
    # Shuffle and split
    random.shuffle(all_texts)
    split_idx = int(len(all_texts) * split_ratio)
    
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    
    return train_dataset, val_dataset
