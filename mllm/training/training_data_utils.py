"""
File: mllm/training/training_data_utils.py
Summary: Utilities for loading, filtering, and batching training data.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from mllm.markov_games.rollout_tree import (
    ChatTurn,
    RolloutTreeBranchNode,
    RolloutTreeNode,
    RolloutTreeRootNode,
)


@dataclass
class AdvantagePacket:
    """Message used by trainers to share per-rollout advantages."""

    agent_id: str
    rollout_ids: torch.IntTensor  # (B,)
    # list-of-tensors
    main_advantages: list[torch.FloatTensor]


class TrainingChatTurn:
    """
    Lightweight ChatTurn variant that records the timestep index alongside role/content.
    """

    def __init__(
        self,
        time_step: int,
        role: str,
        agent_id: str,
        content: str,
        chat_template_token_ids: list[int],
        reasoning_content: str,
        is_state_end: bool,
        out_token_ids: Optional[list[int]] = None,
        log_probs: Optional[list[float]] = None,
    ) -> None:
        self.time_step = time_step
        self.role = role
        self.agent_id = agent_id
        self.content = content
        self.chat_template_token_ids = chat_template_token_ids
        self.reasoning_content = reasoning_content
        self.is_state_end = is_state_end
        self.out_token_ids = out_token_ids
        self.log_probs = log_probs

    def dict(self):
        return {
            "time_step": self.time_step,
            "role": self.role,
            "agent_id": self.agent_id,
            "content": self.content,
            "chat_template_token_ids": self.chat_template_token_ids,
            "reasoning_content": self.reasoning_content,
            "is_state_end": self.is_state_end,
            "out_token_ids": self.out_token_ids,
            "log_probs": self.log_probs,
        }


def get_main_chat_list_and_rewards(
    agent_id: str, root: RolloutTreeRootNode | RolloutTreeNode
) -> Tuple[list[TrainingChatTurn], torch.FloatTensor]:
    """
    This method traverses a rollout tree and returns a the list of ChatTurn
    for an agent. If it encounters a branch node, it follows the main path.
    """
    # Currently follows only the main branch; extend if side branches must be included.
    if isinstance(root, RolloutTreeRootNode):
        current_node = root.child
    else:
        current_node = root

    chat = []
    rewards = []
    while current_node is not None:
        if isinstance(current_node, RolloutTreeBranchNode):
            current_node = current_node.main_child
        reward: float = current_node.step_log.simulation_step_log.rewards[agent_id]
        rewards.append(reward)
        chat_turns: list[TrainingChatTurn] = current_node.step_log.action_logs[
            agent_id
        ].chat_turns
        chat_turns = [
            TrainingChatTurn(time_step=current_node.time_step, **turn.model_dump())
            for turn in chat_turns
        ]
        chat.extend(chat_turns)
        current_node = current_node.child
    return chat, torch.FloatTensor(rewards)


def get_tokenwise_credits(
    # B := batch size, S := number of tokens / seq. length, T := number of states. `j` stands for jagged (see pytorch nested tensors.)
    batch_timesteps: torch.IntTensor | torch.Tensor,  # (B, jS),
    batch_credits: torch.FloatTensor | torch.Tensor,  # (B, jT)
) -> torch.FloatTensor | torch.Tensor:  # (B, jS)
    """
    Expand per-state credits so every token at that timestep receives the same value.
    """
    # The explicit loops keep jagged tensor semantics simple; optimize later if profiling warrants it.
    batch_token_credits = []
    for credits, timesteps in zip(batch_credits, batch_timesteps):
        token_credits = torch.zeros_like(
            timesteps,
            dtype=credits.dtype,
            device=timesteps.device,
        )
        for idx, credit in enumerate(credits):
            token_credits[timesteps == idx] = credit
        batch_token_credits.append(token_credits)
    return batch_token_credits


@dataclass
class TrajectoryBatch:
    """
    Tensorized batch of trajectories using list-of-tensors for jagged dimensions.
    """

    # B := batch size, S := number of tokens / seq. length, T := number of states.
    rollout_ids: torch.IntTensor  # (B,)
    crn_ids: torch.IntTensor  # (B,)
    agent_ids: list[str]  # (B,)
    batch_input_ids: list[torch.LongTensor]  # List[(jS,)]
    batch_action_mask: list[torch.BoolTensor]  # List[(jS,)]
    batch_entropy_mask: list[torch.BoolTensor]  # List[(jS,)]
    batch_timesteps: list[torch.IntTensor]  # List[(jS,)]
    batch_state_ends_mask: list[torch.BoolTensor]  # List[(jS,)]
    batch_engine_log_probs: Optional[list[torch.FloatTensor]]  # List[(jS,)]
    batch_rewards: list[torch.FloatTensor]  # List[(jT,)]
    batch_credits: Optional[list[torch.FloatTensor]] = None  # List[(jS,)]

    def __post_init__(self):
        """
        Validate per-sample consistency.
        """
        B = self.rollout_ids.shape[0]
        assert (
            self.crn_ids.shape[0] == B
        ), "RNG IDs must have length equal to batch size."
        assert (
            len(self.agent_ids) == B
        ), "agent_ids must have length equal to batch size."
        assert (
            len(self.batch_input_ids)
            == len(self.batch_action_mask)
            == len(self.batch_entropy_mask)
            == len(self.batch_timesteps)
            == len(self.batch_state_ends_mask)
            == len(self.batch_engine_log_probs)
            == len(self.batch_rewards)
            == B
        ), "Jagged lists must all have length equal to batch size."

        for b in range(B):
            nb_rewards = int(self.batch_rewards[b].shape[0])
            nb_timesteps = int(torch.max(self.batch_timesteps[b]).item()) + 1
            assert (
                nb_rewards == nb_timesteps
            ), "Number of rewards and timesteps mismatch."
            assert (
                self.batch_input_ids[b].shape[0]
                == self.batch_action_mask[b].shape[0]
                == self.batch_entropy_mask[b].shape[0]
                == self.batch_engine_log_probs[b].shape[0]
                == self.batch_timesteps[b].shape[0]
            ), "Tensors must have the same shape along the jagged dimension."
            assert (
                int(self.batch_state_ends_mask[b].sum())
                == self.batch_rewards[b].shape[0]
            ), "Number of rewards must match number of state ends."

    """
    Entries:
        Here, we ignore the batch dimension.
        input_ids:
            All of the tokens of both the user and the assistant, flattened.
        action_mask:
            Set to true on the tokens of the assistant (tokens generated by the model).
        timesteps:
            Therefore, max(timesteps) = Ns - 1.
        state_ends_idx:
            Indices of the tokens at which state descriptions end.
        rewards:
            rewards[t] := R_t(s_t, a_t)
    Example:
        position:       "0  1  2  3  4  5  6  7  8  9  10 11 12 13 14"
        input_ids:      "U  U  U  a  a  a  U  a  U  a  a  a  U  U  U" (U := User, a := Assistant)
        action_mask:    "x  x  x  ✓  ✓  ✓  x  ✓  x  ✓  ✓  ✓  x  x  x"
        timestep:       "0  0  0  0  0  0  1  1  1  1  1  1  2  2  2"
        state_ends_dx:  [2, 6, 14]
        rewards:        [r0, r1, r2]
    """

    def __getitem__(self, key) -> "TrajectoryBatch":
        if isinstance(key, slice):
            return TrajectoryBatch(
                rollout_ids=self.rollout_ids.__getitem__(key),
                crn_ids=self.crn_ids.__getitem__(key),
                agent_ids=self.agent_ids[key],
                batch_input_ids=self.batch_input_ids[key],
                batch_action_mask=self.batch_action_mask[key],
                batch_entropy_mask=self.batch_entropy_mask[key],
                batch_timesteps=self.batch_timesteps[key],
                batch_state_ends_mask=self.batch_state_ends_mask[key],
                batch_engine_log_probs=self.batch_engine_log_probs[key],
                batch_rewards=self.batch_rewards[key],
                batch_credits=self.batch_credits[key] if self.batch_credits else None,
            )

    def __len__(self):
        return len(self.batch_input_ids)

    def to(self, device):
        self.rollout_ids = self.rollout_ids.to(device)
        self.crn_ids = self.crn_ids.to(device)
        self.batch_input_ids = [t.to(device) for t in self.batch_input_ids]
        self.batch_action_mask = [t.to(device) for t in self.batch_action_mask]
        self.batch_entropy_mask = [t.to(device) for t in self.batch_entropy_mask]
        self.batch_timesteps = [t.to(device) for t in self.batch_timesteps]
        self.batch_state_ends_mask = [t.to(device) for t in self.batch_state_ends_mask]
        self.batch_engine_log_probs = [
            t.to(device) for t in self.batch_engine_log_probs
        ]
        self.batch_rewards = [t.to(device) for t in self.batch_rewards]
        self.batch_credits = (
            [t.to(device) for t in self.batch_credits] if self.batch_credits else None
        )

    def get_padded_tensors_for_critic(self):
        """
        Returns:
            padded_batch_input_ids: (B, P)
            padded_batch_state_ends_mask: (B, P)
            timestep_counts: (B,) tensor of ints indicating number of states per sample
        """
        padded_batch_input_ids = pad_sequence(
            self.batch_input_ids, batch_first=True, padding_value=0
        )
        padded_batch_state_ends_mask = pad_sequence(
            self.batch_state_ends_mask, batch_first=True, padding_value=0
        ).bool()
        # number of states equals number of True in state_ends_mask
        timestep_counts = torch.tensor(
            [int(mask.sum().item()) for mask in self.batch_state_ends_mask],
            device=padded_batch_input_ids.device,
            dtype=torch.long,
        )
        return padded_batch_input_ids, padded_batch_state_ends_mask, timestep_counts


timestep = int


@dataclass
class PaddedTensorTrainingBatch:
    """Helper struct returned by ``TrainingBatch.get_padded_tensors``."""

    batch_input_ids: torch.LongTensor | torch.Tensor
    batch_action_mask: torch.BoolTensor | torch.Tensor
    batch_entropy_mask: Optional[torch.BoolTensor | torch.Tensor]
    batch_credits: torch.FloatTensor | torch.Tensor
    batch_engine_log_probs: torch.FloatTensor | torch.Tensor
    batch_timesteps: torch.IntTensor | torch.Tensor

    def __len__(self):
        return self.batch_input_ids.shape[0]

    def to(self, device):
        self.batch_input_ids = self.batch_input_ids.to(device)
        self.batch_action_mask = self.batch_action_mask.to(device)
        self.batch_entropy_mask = self.batch_entropy_mask.to(device)
        self.batch_credits = self.batch_credits.to(device)
        self.batch_engine_log_probs = self.batch_engine_log_probs.to(device)
        self.batch_timesteps = self.batch_timesteps.to(device)


@dataclass
class TrainingBatch:
    rollout_ids: torch.IntTensor | torch.Tensor  # (B,)
    batch_input_ids: list[torch.LongTensor]  # List[(jS,)]
    batch_action_mask: list[torch.BoolTensor]  # List[(jS,)]
    batch_entropy_mask: Optional[list[torch.BoolTensor]]  # List[(jS,)]
    batch_credits: list[torch.FloatTensor]  # List[(jS,)]
    batch_engine_log_probs: list[torch.FloatTensor]  # List[(jS,)]
    batch_timesteps: list[torch.IntTensor]  # List[(jS,)]

    def __post_init__(self):
        # Ensure batch dimension is present
        assert (
            len(self.batch_input_ids)
            == len(self.batch_action_mask)
            == len(self.batch_entropy_mask)
            == len(self.batch_credits)
            == len(self.batch_engine_log_probs)
            == len(self.batch_timesteps)
            == self.rollout_ids.shape[0]
        ), "Jagged lists must all have length equal to batch size."
        for inp, mask, cred, engine_log_prob, timestep in zip(
            self.batch_input_ids,
            self.batch_action_mask,
            self.batch_credits,
            self.batch_engine_log_probs,
            self.batch_timesteps,
        ):
            assert (
                inp.shape[0]
                == mask.shape[0]
                == cred.shape[0]
                == engine_log_prob.shape[0]
                == timestep.shape[0]
            ), "Tensors must have the same shapes along the jagged dimension."

    def __getitem__(self, key) -> "TrainingBatch":
        if isinstance(key, slice):
            return TrainingBatch(
                rollout_ids=self.rollout_ids.__getitem__(key),
                batch_input_ids=self.batch_input_ids[key],
                batch_action_mask=self.batch_action_mask[key],
                batch_entropy_mask=self.batch_entropy_mask[key],
                batch_credits=self.batch_credits[key],
                batch_engine_log_probs=self.batch_engine_log_probs[key],
                batch_timesteps=self.batch_timesteps[key],
            )

    def __len__(self):
        return len(self.batch_input_ids)

    def to(self, device):
        self.rollout_ids = self.rollout_ids.to(device)
        self.batch_input_ids = [t.to(device) for t in self.batch_input_ids]
        self.batch_action_mask = [t.to(device) for t in self.batch_action_mask]
        self.batch_entropy_mask = [t.to(device) for t in self.batch_entropy_mask]
        self.batch_credits = [t.to(device) for t in self.batch_credits]
        self.batch_engine_log_probs = [
            t.to(device) for t in self.batch_engine_log_probs
        ]
        self.batch_timesteps = [t.to(device) for t in self.batch_timesteps]

    def get_padded_tensors(self, padding: float = 0.0):
        """
        Materialize right-padded tensors so PyTorch ops can run on uniform shapes.
        """
        padded_batch_input_ids = pad_sequence(
            self.batch_input_ids, batch_first=True, padding_value=int(padding)
        )
        padded_batch_action_mask = pad_sequence(
            [m.to(dtype=torch.bool) for m in self.batch_action_mask],
            batch_first=True,
            padding_value=False,
        )
        padded_batch_entropy_mask = pad_sequence(
            self.batch_entropy_mask, batch_first=True, padding_value=False
        )
        padded_batch_credits = pad_sequence(
            self.batch_credits, batch_first=True, padding_value=float(padding)
        )
        padded_batch_engine_log_probs = pad_sequence(
            self.batch_engine_log_probs, batch_first=True, padding_value=float(padding)
        )
        padded_batch_timesteps = pad_sequence(
            self.batch_timesteps, batch_first=True, padding_value=0
        )

        return PaddedTensorTrainingBatch(
            padded_batch_input_ids,
            padded_batch_action_mask,
            padded_batch_entropy_mask,
            padded_batch_credits,
            padded_batch_engine_log_probs,
            padded_batch_timesteps,
        )

    def append(self, other: "TrainingBatch"):
        self.rollout_ids = torch.cat([self.rollout_ids, other.rollout_ids])
        self.batch_input_ids.extend(other.batch_input_ids)
        self.batch_action_mask.extend(other.batch_action_mask)
        self.batch_entropy_mask.extend(other.batch_entropy_mask)
        self.batch_credits.extend(other.batch_credits)
        self.batch_engine_log_probs.extend(other.batch_engine_log_probs)
        self.batch_timesteps.extend(other.batch_timesteps)


timestep = int
