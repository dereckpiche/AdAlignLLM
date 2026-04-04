"""
File: mllm/training/trainer_independent.py
Summary: Trainer for independently optimizing each agent.
"""

import logging
import os
import sys
from typing import Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from pandas._libs.tslibs.offsets import CBMonthBegin
from peft import LoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.markov_games.rollout_tree import *
from mllm.markov_games.rollout_tree import RolloutTreeRootNode
from mllm.training.credit_methods import (
    get_discounted_returns,
    get_discounted_state_visitation_credits,
    get_generalized_advantage_estimates,
    get_rloo_credits,
)
from mllm.training.tally_metrics import Tally
from mllm.training.tally_tokenwise import ContextualizedTokenwiseTally
from mllm.training.tokenize_chats import *
from mllm.training.tokenize_chats import process_training_chat
from mllm.training.trainer_common import BaseTrainer
from mllm.training.training_data_utils import *
from mllm.training.training_data_utils import (
    TrainingBatch,
    TrajectoryBatch,
    get_tokenwise_credits,
)
from mllm.utils.resource_context import resource_logger_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class TrainingData:
    """Caches per-agent trajectory tensors plus their computed advantages."""

    agent_id: str
    main_data: TrajectoryBatch
    # list-of-tensors: per rollout advantages with length jT
    main_advantages: list[torch.FloatTensor] | None = None


class TrainerNaive(BaseTrainer):
    def set_agent_trajectory_data(
        self, agent_id: str, roots: list[RolloutTreeRootNode]
    ) -> None:
        """
        Tokenize rollouts for a given agent and cache the tensors used for training.
        """
        # Reset per-agent buffers; extend this logic if joint training batches are needed.
        self.policy_gradient_data = None

        # Tensorize Chats
        rollout_ids = []
        crn_ids = []  # common random number id
        batch_input_ids = []
        batch_action_mask = []
        batch_entropy_mask = []
        batch_timesteps = []
        batch_state_ends_mask = []
        batch_engine_log_probs = []
        batch_rewards = []
        for root in roots:
            rollout_id = root.id
            self.debug_path_list.append(
                "mgid:" + str(rollout_id) + "_agent_id:" + agent_id
            )
            rollout_ids.append(rollout_id)
            crn_ids.append(root.crn_id)
            chat, rewards = get_main_chat_list_and_rewards(
                agent_id=agent_id,
                root=root,
                reward_agent_id=self.reward_agent_id,
                reward_scale=self.reward_scale,
                reward_peer_agent_id=self.reward_peer_agent_id,
                reward_peer_scale=self.reward_peer_scale,
            )
            (
                input_ids,
                action_mask,
                entropy_mask,
                timesteps,
                state_ends_mask,
                engine_log_probs,
            ) = process_training_chat(
                tokenizer=self.tokenizer,
                chat_history=chat,
                entropy_mask_regex=self.entropy_mask_regex,
                exploration_prompts_to_remove=self.exploration_prompts_to_remove,
            )
            batch_input_ids.append(input_ids)
            batch_action_mask.append(action_mask)
            batch_entropy_mask.append(entropy_mask)
            batch_timesteps.append(timesteps)
            batch_state_ends_mask.append(state_ends_mask)
            batch_engine_log_probs.append(engine_log_probs)
            batch_rewards.append(rewards)

        trajectory_batch = TrajectoryBatch(
            rollout_ids=torch.tensor(rollout_ids, dtype=torch.int32),
            crn_ids=torch.tensor(crn_ids, dtype=torch.int32),
            agent_ids=[agent_id] * len(rollout_ids),
            batch_input_ids=batch_input_ids,
            batch_action_mask=batch_action_mask,
            batch_entropy_mask=batch_entropy_mask,
            batch_timesteps=batch_timesteps,
            batch_state_ends_mask=batch_state_ends_mask,
            batch_rewards=batch_rewards,
            batch_engine_log_probs=batch_engine_log_probs,
        )

        # Get Advantages
        batch_advantages: torch.FloatTensor = (
            self.get_advantages_with_critic_gradient_accumulation(trajectory_batch)
        )

        # Discount state visitation (the mathematically correct way)
        if not self.skip_discounted_state_visitation:
            for i in range(len(batch_advantages)):
                batch_advantages[i] = get_discounted_state_visitation_credits(
                    batch_advantages[i].unsqueeze(0),
                    self.discount_factor,
                ).squeeze(0)

        self.training_data[agent_id] = TrainingData(
            agent_id=agent_id,
            main_data=trajectory_batch,
            main_advantages=batch_advantages,
        )

    def receive_advantage_data(self, advantage_packets: list[AdvantagePacket]):
        """
        This trainer ignores the advantages of the other trainers.
        """
        for agent_id, agent_data in self.training_data.items():
            self.training_data[agent_id] = agent_data.main_data
            self.training_data[agent_id].batch_credits = agent_data.main_advantages

    def share_advantage_data(self) -> list[AdvantagePacket]:
        """
        Share the advantage data with other agents.
        Returns:
            AdvantagePacket: The advantage packet containing the agent's advantages.
        """
        logger.info(f"Sharing advantage data.")
        advantage_packets = []
        for agent_id, agent_data in self.training_data.items():
            advantage_packets.append(
                AdvantagePacket(
                    agent_id=agent_id,
                    rollout_ids=agent_data.main_data.rollout_ids,
                    main_advantages=agent_data.main_advantages,
                )
            )
        return advantage_packets
