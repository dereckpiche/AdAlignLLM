"""
File: mllm/models/adapter_training_wrapper.py
Summary: Wraps a shared LLM with adapter-specific PEFT handling for training.
"""

import logging
from typing import Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class AdapterWrapper(nn.Module):
    """
    A thin façade that
      • keeps a reference to a *shared* PEFT-wrapped model,
      • ensures `set_adapter(adapter)` is called on every forward,
      • exposes only the parameters that should be trained for that adapter
        (plus whatever extra modules you name).
    """

    def __init__(
        self,
        shared_llm: nn.Module,
        adapter_id: str,
        lora_config: dict,
        path: Union[str, None] = None,
    ):
        super().__init__()
        self.shared_llm = shared_llm
        self.adapter_id = adapter_id
        lora_config = LoraConfig(**lora_config)
        # this modifies the shared llm in place, adding a lora adapter inside
        self.shared_llm = get_peft_model(
            model=shared_llm,
            peft_config=lora_config,
            adapter_name=adapter_id,
        )
        self.shared_llm.train()
        # Load external adapter weights if provided
        loaded_from: str | None = None
        if path:
            try:
                # Supports both local filesystem paths and HF Hub repo IDs
                self.shared_llm.load_adapter(
                    is_trainable=True,
                    model_id=path,
                    adapter_name=adapter_id,
                )
                loaded_from = path
            except (
                Exception
            ) as exc:  # noqa: BLE001 - want to log any load failure context
                logger.warning(
                    f"Adapter '{adapter_id}': failed to load from '{path}': {exc}"
                )

        if loaded_from:
            logger.info(
                f"Adapter '{adapter_id}': loaded initial weights from '{loaded_from}'."
            )
        else:
            logger.info(
                f"Adapter '{adapter_id}': initialized with fresh weights (no initial weights found)."
            )

    def parameters(self, recurse: bool = True):
        """
        "recurse" is just for pytorch compatibility
        """
        self.shared_llm.set_adapter(self.adapter_id)
        params = [p for p in self.shared_llm.parameters() if p.requires_grad]

        return params

    def get_base_model_logits(self, contexts):
        """
        Run the base model (without adapter) in inference mode, without tracking gradients.
        This is useful to get reference logits for KL-divergence computation.
        """
        with torch.no_grad():
            with self.shared_llm.disable_adapter():
                return self.shared_llm(input_ids=contexts)[0]

    def forward(self, *args, **kwargs):
        self.shared_llm.set_adapter(self.adapter_id)
        return self.shared_llm(*args, **kwargs)

    def save_pretrained(self, save_path):
        self.shared_llm.save_pretrained(save_path)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.shared_llm.gradient_checkpointing_enable(*args, **kwargs)

    @property
    def dtype(self):
        return self.shared_llm.dtype

    @property
    def device(self):
        return self.shared_llm.device
