"""
File: mllm/models/inference_backend.py
Summary: Declares the inference backend interface and shared dataclasses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMInferenceOutput:
    content: str
    reasoning_content: str | None = None
    log_probs: list[float] | None = None
    out_token_ids: list[int] | None = None


class LLMInferenceBackend(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def prepare_adapter(
        self, adapter_id: str, weights_got_updated: bool = False
    ) -> None:
        """Ensure adapter is ready/loaded for next generation call."""

    @abstractmethod
    async def generate(self, prompt: list[dict], regex: Optional[str] = None) -> str:
        ...

    @abstractmethod
    def toggle_training_mode(self) -> None:
        ...

    @abstractmethod
    def toggle_eval_mode(self) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...
