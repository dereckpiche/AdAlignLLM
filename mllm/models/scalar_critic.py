"""
File: mllm/models/scalar_critic.py
Summary: Defines a scalar critic network and helper utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.models.adapter_training_wrapper import AdapterWrapper


class ScalarCritic(nn.Module):
    """
    A causal-LM critic_adapter + a scalar value head:
        V_φ(s) = wᵀ h_last + b
    Only LoRA adapters (inside critic_adapter) and the value head are trainable.
    """

    def __init__(self, critic_adapter: AdapterWrapper):
        super().__init__()
        self.critic_adapter = critic_adapter
        hidden_size = self.critic_adapter.shared_llm.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(
            dtype=critic_adapter.dtype, device=critic_adapter.device
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # AdapterWrapper activates its own adapter internally
        outputs = self.critic_adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        h_last = outputs.hidden_states[-1]  # (B, S, H)
        values = self.value_head(h_last).squeeze(-1)  # (B, S)
        return values

    def parameters(self, recurse: bool = True):
        """Iterator over *trainable* parameters for this critic."""
        # 1) LoRA params for *this* adapter
        for p in self.critic_adapter.parameters():
            yield p
        # 2) scalar head
        yield from self.value_head.parameters()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.critic_adapter.gradient_checkpointing_enable(*args, **kwargs)

    @property
    def dtype(self):
        return self.critic_adapter.dtype

    @property
    def device(self):
        return self.critic_adapter.device
