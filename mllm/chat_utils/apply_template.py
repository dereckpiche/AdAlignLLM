"""
File: mllm/chat_utils/apply_template.py
Summary: Applies tokenizer-specific chat templates and stitches chat token IDs.
"""

import torch

from mllm.chat_utils.chat_turn import ChatTurn
from mllm.chat_utils.template_specific import (
    custom_gemma3_template,
    custom_llama3_template,
    custom_qwen2_template,
    custom_qwen3_template,
    gemma3_assistant_postfix,
    qwen2_assistant_postfix,
    qwen3_assistant_postfix,
)


def get_custom_chat_template(tokenizer) -> str:
    """
    Get the chat template for the tokenizer.
    """
    if "qwen2" in tokenizer.name_or_path.lower():
        return custom_qwen2_template
    elif "llama" in tokenizer.name_or_path.lower():
        return custom_llama3_template
    elif "qwen3" in tokenizer.name_or_path.lower():
        return custom_qwen3_template
    elif "gemma" in tokenizer.name_or_path.lower():
        return custom_gemma3_template
    else:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")


def get_custom_assistant_postfix(tokenizer) -> torch.Tensor:
    """
    Get the custom assistant postfix for the tokenizer.
    """
    if "qwen2" in tokenizer.name_or_path.lower():
        return qwen2_assistant_postfix
    elif "qwen3" in tokenizer.name_or_path.lower():
        return qwen3_assistant_postfix
    elif "gemma" in tokenizer.name_or_path.lower():
        return gemma3_assistant_postfix
    return torch.tensor([], dtype=torch.long)


def tokenize_chats(chats: list[ChatTurn], tokenizer, enable_thinking) -> None:
    """
    Set the chat_template_token_ids for each chat turn.
    We rely on tokenizer-side templates because engine-provided cached tokens are not exposed yet.
    """
    custom_template = get_custom_chat_template(tokenizer)
    custom_assistant_postfix: torch.Tensor = get_custom_assistant_postfix(tokenizer)
    for i, chat in enumerate(chats):
        if chat.chat_template_token_ids is None:
            if chat.role == "user":
                next_chat = chats[i + 1] if i + 1 < len(chats) else None
                add_generation_prompt = True
                if next_chat and next_chat.role == "user":
                    add_generation_prompt = False
                encoded_chat = tokenizer.apply_chat_template(
                    [chat],
                    return_tensors="pt",
                    chat_template=custom_template,
                    add_generation_prompt=add_generation_prompt,
                    add_system_prompt=True if i == 0 else False,
                    enable_thinking=enable_thinking,
                ).flatten()
                previous_chat = chats[i - 1] if i > 0 else None
                if previous_chat and previous_chat.role == "assistant":
                    encoded_chat = torch.cat([custom_assistant_postfix, encoded_chat])
            elif chat.role == "assistant":
                encoded_chat = chat.out_token_ids
            chat.chat_template_token_ids = encoded_chat


def chat_turns_to_token_ids(
    chats: list[ChatTurn], tokenizer, enable_thinking
) -> list[int]:
    """
    Tokenize the chat turns and set the chat_template_token_ids for each chat turn.
    """
    tokenize_chats(chats=chats, tokenizer=tokenizer, enable_thinking=enable_thinking)
    token_ids = []
    for chat in chats:
        token_ids.append(chat.chat_template_token_ids)
    return torch.cat(token_ids)
