import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union, Dict

class MistralMoEConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        num_experts=8,
        num_experts_per_token=2,
        expert_hidden_size=14336,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_hidden_size = expert_hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x * norm


def precompute_rotary_emb_cache(dim, seq_len, theta=10000.0, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(seq_len, 1, dim)
    sin = emb.sin().view(seq_len, 1, dim)
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(query, key, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key


class MistralAttention(nn.Module):
    def __init__(self, config: MistralMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        # Set up cache for rotary embeddings
        if self.cos_cached is None or self.cos_cached.size(0) < kv_seq_len:
            self.cos_cached, self.sin_cached = precompute_rotary_emb_cache(
                self.head_dim, kv_seq_len, self.config.rope_theta, device=hidden_states.device
            )
            
        cos, sin = self.cos_cached, self.sin_cached
        query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin, position_ids)
        
        if past_key_value is not None:
            # Reuse pre-computed key and value states
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat key/value heads if necessary
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
            
        # Sliding window attention
        if self.sliding_window is not None and key_states.shape[2] > self.sliding_window:
            window_size = self.sliding_window
            attention_scores = torch.matmul(query_states, key_states.transpose(2, 3))
            
            # Create sliding window causal mask
            mask = torch.ones((query_states.shape[2], key_states.shape[2]), device=query_states.device)
            mask = torch.tril(mask, diagonal=0)
            # Apply sliding window
            for i in range(mask.shape[0]):
                window_start = max(0, i - window_size + 1)
                mask[i, :window_start] = 0
            
            attention_scores = attention_scores * mask.view(1, 1, mask.shape[0], mask.shape[1])
            attention_scores = attention_scores.masked_fill(mask.view(1, 1, mask.shape[0], mask.shape[1]) == 0, float('-inf'))
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(2, 3))
            
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Calculate attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(query_states)
        
        # Apply attention to value states
        context_layer = torch.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, -1)
        
        # Output projection
        output = self.o_proj(context_layer)
        
        outputs = (output, None, past_key_value) if use_cache else (output, None)
        
        return outputs


class MistralMLP(nn.Module):
    def __init__(self, config: MistralMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralExpertMLP(nn.Module):
    def __init__(self, config: MistralMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.expert_hidden_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(self.expert_hidden_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralSparseMoE(nn.Module):
    def __init__(self, config: MistralMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Router for selecting experts
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Initialize experts
        self.experts = nn.ModuleList([MistralExpertMLP(config) for _ in range(self.num_experts)])
        
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)  # (batch_size * sequence_length, hidden_size)
        
        # Get router logits and expert weights
        router_logits = self.router(hidden_states)  # (batch_size * sequence_length, num_experts)
        
        # Get routing weights with top-k gating
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_token, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Compute expert outputs
        expert_outputs = torch.zeros(
            (batch_size * sequence_length, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        # Compute the output of each expert for selected tokens
        for i, expert in enumerate(self.experts):
            # Find which tokens have this expert as one of their selected experts
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get the indices where this expert was selected
            expert_indices = torch.nonzero(expert_mask).squeeze(-1)
            
            # Find the position in the top-k list for this expert
            expert_position = (selected_experts == i).int().argmax(dim=-1)[expert_mask]
            
            # Get the corresponding routing weights
            expert_weights = routing_weights[expert_indices, expert_position].unsqueeze(-1)
            
            # Compute the output for the expert
            expert_output = expert(hidden_states[expert_indices])
            
            # Add the weighted expert output to the final output
            expert_outputs[expert_indices] += expert_weights * expert_output
            
        # Reshape output back to (batch_size, sequence_length, hidden_size)
        expert_outputs = expert_outputs.view(batch_size, sequence_length, hidden_size)
        
        return expert_outputs


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config)
        
        # Use MoE FFN or standard MLP based on layer index and configuration
        if layer_idx % 2 == 0:  # Every other layer uses MoE
            self.mlp = MistralSparseMoE(config)
        else:
            self.mlp = MistralMLP(config)
            
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = attn_outputs[0]
        
        # Add residual connection to attention output
        hidden_states = residual + hidden_states
        
        # FFN (either MoE or MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attn_outputs[1:]
        
        return outputs


class MistralMoEModel(nn.Module):
    def __init__(self, config: MistralMoEConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Set defaults
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask if attention_mask.dtype == torch.bool else attention_mask
            # Convert attention mask to float
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_length]
            
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            
        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache += (layer_outputs[2 if output_attentions else 1],)
                
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


class MistralMoEForCausalLM(nn.Module):
    def __init__(self, config: MistralMoEConfig):
        super().__init__()
        self.config = config
        self.model = MistralMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        
        # Optional weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
        }
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # If past_key_values are used, only the last token should be passed
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        # Create position_ids based on attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=None,
        eos_token_id=None,
    ):
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        batch_size = input_ids.shape[0]
        past_key_values = None
        generated = input_ids
        
        # Set up position_ids for the initial input
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            seq_length = input_ids.shape[1]
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            attention_mask = torch.ones_like(input_ids)
            
        # Generation loop
        for _ in range(max_length - input_ids.shape[1]):
            model_inputs = self.prepare_inputs_for_generation(
                generated, past_key_values=past_key_values, attention_mask=attention_mask
            )
            
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            past_key_values = outputs["past_key_values"]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Top-k and top-p filtering
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create index mask
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Extend attention mask for next token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=input_ids.device)], dim=1
            )
            
            # Extend generated token sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
            
            # Check if all sequences have reached the end token
            if (next_token == eos_token_id).all():
                break
                
        return generated
