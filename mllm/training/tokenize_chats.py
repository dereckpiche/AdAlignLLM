"""
File: mllm/training/tokenize_chats.py
Summary: Tokenizes chat datasets and prepares tensors for training.
"""

import logging
import sys

import regex
import torch
from transformers import AutoTokenizer

from mllm.training.training_data_utils import TrainingChatTurn, TrajectoryBatch

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def process_training_chat(
    tokenizer: AutoTokenizer,
    chat_history: list[TrainingChatTurn],
    entropy_mask_regex: str | None = None,
    exploration_prompts_to_remove: list[str] = [],
    use_engine_out_token_ids: bool = False,
) -> tuple[torch.IntTensor, torch.BoolTensor, torch.IntTensor, torch.BoolTensor]:
    """Tokenize a single training chat and build aligned per-token masks.

    Given an ordered list of `TrainingChatTurn`, this function tokenizes each
    turn independently using the tokenizer's chat template, then concatenates
    all resulting token sequences. It also constructs three parallel 1D masks
    that align with the concatenated tokens:

    - input_ids: token ids for the entire chat, turn by turn
    - action_mask: True for tokens that belong to assistant turns (i.e., model
      actions), False for tokens from other roles
    - timesteps: per-token time step copied from the originating turn's
      `time_step`
    - state_ends_mask: True for the last token of any turn where
      `is_state_end` is True, otherwise False

    Important details:
    - Each turn is passed as a single-message list to
      `tokenizer.apply_chat_template` and flattened; the per-turn outputs are
      then concatenated in the original order.
    - Turn boundaries are not explicitly encoded beyond what the chat template
      inserts; masks provide alignment for learning signals and state endings.
    - No truncation or padding is performed here; downstream code should handle
      batching/padding as needed.
    - Note on dtypes: `input_ids` will be a LongTensor (int64). `action_mask`
      and `state_ends_mask` are BoolTensors. `timesteps` is currently created
      as a float tensor; adjust the implementation if integer dtype is
      required downstream.

    Args:
        tokenizer: A Hugging Face tokenizer supporting `apply_chat_template`.
        chat_history: Ordered list of `TrainingChatTurn` forming one dialogue.

    Returns:
        A tuple of four 1D tensors, all of equal length N (the total number of
        tokens across all turns), in the following order:
        - input_ids (LongTensor)
        - action_mask (BoolTensor)
        - timesteps (FloatTensor as implemented; see note above)
        - state_ends_mask (BoolTensor)
    """
    state_ends_mask = []
    input_ids = []
    action_mask = []
    timesteps = []
    entropy_mask = []
    engine_log_probs = []
    for train_chat_turn in chat_history:
        is_state_end = train_chat_turn.is_state_end
        time_step = train_chat_turn.time_step
        is_action = train_chat_turn.role == "assistant"

        # Remove exploration prompts from training data
        for exploration_prompt in exploration_prompts_to_remove:
            if exploration_prompt in train_chat_turn.content:
                train_chat_turn.content = train_chat_turn.content.replace(
                    exploration_prompt, ""
                )

        chat_turn = {
            "role": train_chat_turn.role,
            "content": train_chat_turn.content,
        }
        if entropy_mask_regex is not None:
            is_entropy_mask_true = (
                regex.search(entropy_mask_regex, train_chat_turn.content) is not None
            )
        else:
            is_entropy_mask_true = True
        if is_action:
            chat_turn_ids = train_chat_turn.out_token_ids
            nb_chat_turns_ids = chat_turn_ids.numel()
            action_mask.append(torch.ones(nb_chat_turns_ids, dtype=torch.bool))
            engine_log_probs.append(train_chat_turn.log_probs)
        else:
            chat_turn_ids = train_chat_turn.chat_template_token_ids
            nb_chat_turns_ids = chat_turn_ids.numel()
            action_mask.append(torch.zeros(nb_chat_turns_ids, dtype=torch.bool))
            engine_log_probs.append(torch.zeros(nb_chat_turns_ids, dtype=torch.float))
        nb_chat_turns_ids = chat_turn_ids.numel()
        state_ends_mask.append(torch.zeros(nb_chat_turns_ids, dtype=torch.bool))
        if is_state_end:
            state_ends_mask[-1][-1] = True  # last token is state end
        input_ids.append(chat_turn_ids)
        entropy_mask.append(torch.ones(nb_chat_turns_ids, dtype=torch.bool))
        if not is_entropy_mask_true:
            entropy_mask[-1] = entropy_mask[-1] * False
        timesteps.append(torch.ones(nb_chat_turns_ids) * time_step)
    input_ids = torch.cat(input_ids)
    action_mask = torch.cat(action_mask)
    entropy_mask = torch.cat(entropy_mask)
    timesteps = torch.cat(timesteps)
    timesteps = timesteps.to(torch.long)
    state_ends_mask = torch.cat(state_ends_mask)
    engine_log_probs = torch.cat(engine_log_probs)

    return (
        input_ids,
        action_mask,
        entropy_mask,
        timesteps,
        state_ends_mask,
        engine_log_probs,
    )
