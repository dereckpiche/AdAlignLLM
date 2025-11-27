"""
File: mllm/chat_utils/chat_turn.py
Summary: Defines the ChatTurn schema plus helpers for serialization and validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import jsonschema
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentId = str


class ChatTurn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # needed for torch tensors

    role: str = Field(pattern="^(user|assistant)$")
    agent_id: AgentId  # ID of the agent with which the chat occured
    content: str
    reasoning_content: str | None = None
    chat_template_token_ids: torch.LongTensor | None = None  # Token ids of chat template format. For example, token ids of "<assistant>{content}</assistant>""
    out_token_ids: torch.LongTensor | None = (
        None  # tokens generated from inference engine
    )
    log_probs: torch.FloatTensor | None = None
    is_state_end: bool = False  # indicates whether this chat turn marks the end of a state in the trajectory
