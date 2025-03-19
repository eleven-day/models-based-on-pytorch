import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class LanguageModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", lora_rank=8, use_lora=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        
        # Apply LoRA if needed
        if use_lora:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.hidden_size = self.config.hidden_size
    
    def get_hidden_size(self):
        return self.hidden_size
