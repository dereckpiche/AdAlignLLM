"""
File: mllm/training/trainer_common.py
Summary: Shared trainer utilities, base classes, and gradient helpers.
"""

import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from typing import Callable, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from pandas._libs.tslibs.offsets import CBMonthBegin
from peft import LoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from mllm.markov_games.rollout_tree import *
from mllm.markov_games.rollout_tree import RolloutTreeRootNode
from mllm.training.annealing_methods import sigmoid_annealing
from mllm.training.credit_methods import (
    get_discounted_returns,
    get_generalized_advantage_estimates,
    get_rloo_credits,
    whiten_advantages,
    whiten_advantages_time_step_wise,
)
from mllm.training.tally_metrics import Tally
from mllm.training.tally_rollout import RolloutTally, RolloutTallyItem
from mllm.training.tally_tokenwise import ContextualizedTokenwiseTally
from mllm.training.tokenize_chats import *
from mllm.training.tokenize_chats import process_training_chat
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
class TrainerAnnealingState:
    annealing_step_counter: int = 0


class BaseTrainer(ABC):
    """
    Shared scaffolding for policy-gradient trainers (optimizer wiring, logging, etc.).

    Subclasses implement `set_agent_trajectory_data` / `share_advantage_data`
    to plug in algorithm-specific behavior.
    """

    def __init__(
        self,
        policy: AutoModelForCausalLM,
        policy_optimizer: torch.optim.Optimizer,
        critic: Union[AutoModelForCausalLM, None],
        critic_optimizer: Union[torch.optim.Optimizer, None],
        tokenizer: AutoTokenizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        critic_lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
        ######################################################################
        entropy_coeff: float,
        entropy_topk: int,
        entropy_mask_regex: Union[str, None],
        kl_coeff: float,
        gradient_clipping: Union[float, None],
        restrict_tokens: Union[list[str], None],
        mini_batch_size: int,
        use_gradient_checkpointing: bool,
        temperature: float,
        device: str,
        whiten_advantages: bool,
        whiten_advantages_time_step_wise: bool,
        use_gae: bool,
        use_gae_lambda_annealing: bool,
        gae_lambda_annealing_limit: float,
        gae_lambda_annealing_method: Literal["sigmoid_annealing"],
        gae_lambda_annealing_method_params: dict,
        pg_loss_normalization: Literal["batch", "nb_tokens"],
        use_rloo: bool,
        skip_discounted_state_visitation: bool,
        discount_factor: float,
        enable_tokenwise_logging: bool,
        save_path: str,
        reward_agent_id: str | None = None,
        reward_scale: float = 1.0,
        reward_peer_agent_id: str | None = None,
        reward_peer_scale: float = 0.0,
        reward_normalizing_constant: float = 1.0,
        critic_loss_type: Literal["mse", "huber"] = "huber",
        exploration_prompts_to_remove: list[str] = [],
        filter_higher_refprob_tokens_kl: bool = False,
        truncated_importance_sampling_ratio_cap: float = 0.0,
        importance_sampling_strategy: Literal[
            "per_token", "per_sequence"
        ] = "per_token",
        no_rloo_grouping: bool = False,
    ):
        """
        Initialize the REINFORCE trainer with reward shaping for multi-agent or single-agent training.

        Args:
            model (AutoModelForCausalLM): The main policy model.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            optimizer (torch.optim.Optimizer): Optimizer for the policy model.
            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler for the policy model.
            critic (AutoModelForCausalLM or None): Critic model for value estimation (optional).
            critic_optimizer (torch.optim.Optimizer or None): Optimizer for the critic model (optional).
            critic_lr_scheduler (torch.optim.lr_scheduler.LRScheduler or None): LR scheduler for the critic (optional).
            config (RtConfig): Configuration object for training.
        """
        self.tokenizer = tokenizer
        # self.tokenizer.padding_side = "left"  # needed for flash attention
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.lr_scheduler = lr_scheduler
        self.accelerator = Accelerator()
        (
            self.policy,
            self.policy_optimizer,
            self.critic,
            self.critic_optimizer,
        ) = self.accelerator.prepare(policy, policy_optimizer, critic, critic_optimizer)

        self.critic_lr_scheduler = critic_lr_scheduler
        self.tally = Tally()

        if use_gradient_checkpointing == True:
            self.policy.gradient_checkpointing_enable(dict(use_reentrant=False))
            if critic is not None:
                self.critic.gradient_checkpointing_enable(dict(use_reentrant=False))

        self.save_path = save_path

        # Load trainer state if it exists
        self.trainer_annealing_state_path = os.path.join(
            self.save_path, "trainer_annealing_state.pkl"
        )
        if os.path.exists(self.trainer_annealing_state_path):
            logger.info(
                f"Loading trainer state from {self.trainer_annealing_state_path}"
            )
            self.trainer_annealing_state = pickle.load(
                open(self.trainer_annealing_state_path, "rb")
            )
        else:
            self.trainer_annealing_state = TrainerAnnealingState()

        # Load policy optimizer state if it exists
        self.policy_optimizer_path = os.path.join(
            self.save_path, "policy_optimizer_state.pt"
        )
        if os.path.exists(self.policy_optimizer_path):
            logger.info(
                f"Loading policy optimizer state from {self.policy_optimizer_path}"
            )
            self.policy_optimizer.load_state_dict(
                torch.load(self.policy_optimizer_path)
            )

        # Load critic optimizer state if it exists
        self.critic_optimizer_path = os.path.join(
            self.save_path, "critic_optimizer_state.pt"
        )
        if (
            os.path.exists(self.critic_optimizer_path)
            and self.critic_optimizer is not None
        ):
            logger.info(
                f"Loading critic optimizer state from {self.critic_optimizer_path}"
            )
            self.critic_optimizer.load_state_dict(
                torch.load(self.critic_optimizer_path)
            )
        self.device = self.accelerator.device
        self.entropy_coeff = entropy_coeff
        self.entropy_topk = entropy_topk
        self.entropy_mask_regex = entropy_mask_regex
        self.kl_coeff = kl_coeff
        self.gradient_clipping = gradient_clipping
        self.restrict_tokens = restrict_tokens
        self.mini_batch_size = mini_batch_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.temperature = temperature
        self.use_gae = use_gae
        self.whiten_advantages = whiten_advantages
        self.whiten_advantages_time_step_wise = whiten_advantages_time_step_wise
        self.use_rloo = use_rloo
        self.skip_discounted_state_visitation = skip_discounted_state_visitation
        self.use_gae_lambda_annealing = use_gae_lambda_annealing
        self.gae_lambda_annealing_limit = gae_lambda_annealing_limit
        if use_gae_lambda_annealing:
            self.gae_lambda_annealing_method: Callable[
                [int], float
            ] = lambda step: eval(gae_lambda_annealing_method)(
                step=step, **gae_lambda_annealing_method_params
            )
        self.discount_factor = discount_factor
        self.enable_tokenwise_logging = enable_tokenwise_logging
        self.reward_agent_id = reward_agent_id
        self.reward_scale = reward_scale
        self.reward_peer_agent_id = reward_peer_agent_id
        self.reward_peer_scale = reward_peer_scale
        self.reward_normalizing_constant = reward_normalizing_constant
        self.pg_loss_normalization = pg_loss_normalization
        self.critic_loss_type = critic_loss_type
        self.exploration_prompts_to_remove = exploration_prompts_to_remove
        # Common containers used by all trainers
        self.training_data: dict = {}
        self.debug_path_list: list[str] = []
        self.policy_gradient_data = None
        self.tally = Tally()
        self.rollout_tally = RolloutTally()
        self.tokenwise_tally: Union[ContextualizedTokenwiseTally, None] = None
        self.filter_higher_refprob_tokens_kl = filter_higher_refprob_tokens_kl
        self.truncated_importance_sampling_ratio_cap = (
            truncated_importance_sampling_ratio_cap
        )
        self.importance_sampling_strategy = importance_sampling_strategy
        self.no_rloo_grouping = no_rloo_grouping

    def mask_non_restricted_token_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Masks logits so that only allowed tokens (as specified in config.restrict_tokens)
        and the EOS token are active.
        All other logits are set to -inf, effectively removing them from the softmax.

        Args:
            logits (torch.Tensor): The logits tensor of shape (B, S, V).

        Returns:
            torch.Tensor: The masked logits tensor.
        """
        # Gradients flow only through the kept logits; masking is recomputed per batch for clarity.

        if self.restrict_tokens is not None:
            allowed_token_ids = []
            for token in self.restrict_tokens:
                token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
                allowed_token_ids.append(token_ids[0])
            allowed_token_ids.append(
                self.tokenizer.eos_token_id
            )  # This token should always be active
            allowed_token_ids = torch.tensor(allowed_token_ids, device=logits.device)
            # Mask log_probs and probs to only allowed tokens
            mask = torch.zeros_like(logits).bool()  # (B, S, V)
            mask[..., allowed_token_ids] = True
            logits = torch.where(
                mask,
                logits,
                torch.tensor(-float("inf"), device=logits.device),
            )

        return logits

    def apply_reinforce_step(
        self,
        training_batch: TrainingBatch,
    ) -> None:
        """
        Applies a single REINFORCE policy gradient step using the provided batch of rollouts.
        Handles batching, loss computation (including entropy and KL regularization), gradient accumulation, and optimizer step.
        Optionally logs various metrics and statistics.

        Args:
            paths (list[str]): List of game complete file paths for each rollout.
            contexts (list[torch.Tensor]): List of context tensors for each rollout.
            credits (list[torch.Tensor]): List of credit tensors (rewards/advantages) for each rollout.
            action_masks (list[torch.Tensor]): List of action mask tensors for each rollout.
        """
        with resource_logger_context(logger, "Apply reinforce step"):
            self.policy.train()
            mb_size = self.mini_batch_size
            nb_rollouts = len(training_batch)

            # Initialize running mean logs
            running_mean_logs = {
                "rl_objective": 0.0,
                "policy_gradient_loss": 0.0,
                "policy_gradient_norm": 0.0,
                "log_probs": 0.0,
                "credits": 0.0,
                "entropy": 0.0,
                "engine_log_probs_diff_clampfrac": 0.0,
                "tis_imp_ratio": 0.0,
                "ref_log_probs_diff_clampfrac": 0.0,
                "higher_refprob_frac": 0.0,
                "tis_imp_ratio_clampfrac": 0.0,
            }
            if self.entropy_coeff != 0.0:
                running_mean_logs["entropy"] = 0.0
            if self.kl_coeff != 0.0:
                running_mean_logs["kl_divergence"] = 0.0

            # Get total number of tokens generated
            total_tokens_generated = 0
            for att_mask in training_batch.batch_action_mask:
                total_tokens_generated += att_mask.sum()

            # Obtain loss normalization
            if self.pg_loss_normalization == "nb_tokens":
                normalization_factor = total_tokens_generated
            elif self.pg_loss_normalization == "batch":
                normalization_factor = np.ceil(nb_rollouts / mb_size).astype(int)
            else:
                raise ValueError(
                    f"Invalid pg_loss_normalization: {self.pg_loss_normalization}"
                )

            # Gradient accumulation for each mini-batch
            for mb in range(0, nb_rollouts, mb_size):
                logger.info(f"Processing mini-batch {mb} of {nb_rollouts}")
                loss = 0.0
                training_mb = training_batch[mb : mb + mb_size]
                training_mb = training_mb.get_padded_tensors()
                training_mb.to(self.device)
                (
                    tokens_mb,
                    action_mask_mb,
                    entropy_mask_mb,
                    credits_mb,
                    engine_log_probs_mb,
                    timesteps_mb,
                ) = (
                    training_mb.batch_input_ids,
                    training_mb.batch_action_mask,
                    training_mb.batch_entropy_mask,
                    training_mb.batch_credits,
                    training_mb.batch_engine_log_probs,
                    training_mb.batch_timesteps,
                )

                # Next token prediction
                contexts_mb = tokens_mb[:, :-1]
                shifted_contexts_mb = tokens_mb[:, 1:]
                action_mask_mb = action_mask_mb[:, 1:]
                entropy_mask_mb = entropy_mask_mb[:, 1:]
                credits_mb = credits_mb[:, 1:]
                engine_log_probs_mb = engine_log_probs_mb[:, 1:]
                timesteps_mb = timesteps_mb[:, 1:]

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.set_action_mask(action_mask=action_mask_mb)
                    self.tokenwise_tally.set_range(range=(mb, mb + mb_size))
                    self.tokenwise_tally.add_contexts(contexts=contexts_mb)
                    self.tokenwise_tally.add_data(
                        metric_id="next_token",
                        metrics=shifted_contexts_mb,
                        to_tids=True,
                    )
                    self.tokenwise_tally.add_data(
                        metric_id="entropy_mask",
                        metrics=entropy_mask_mb,
                    )

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_credit", metrics=credits_mb
                    )

                # Forward pass + cast to FP-32 for higher prec. Causal LM attention masks are implicit;
                # wire up a custom mask here only if the policy deviates from standard autoregressive behavior.
                logits = self.policy(input_ids=contexts_mb)[0]  # (B, S, V)

                # Mask non-restricted tokens
                if self.restrict_tokens is not None:
                    logits = self.mask_non_restricted_token_logits(logits)

                logits /= self.temperature  # (B, S, V)

                # Compute new log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # (B, S, V)

                # Get log probabilities of actions taken during rollouts
                action_log_probs = log_probs.gather(
                    dim=-1, index=shifted_contexts_mb.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (B, S)
                if self.pg_loss_normalization == "batch":
                    den_running_mean = action_mask_mb.sum() * normalization_factor
                else:
                    den_running_mean = normalization_factor
                running_mean_logs["log_probs"] += (
                    action_log_probs * action_mask_mb
                ).sum().item() / den_running_mean
                running_mean_logs["credits"] += (
                    credits_mb * action_mask_mb
                ).sum().item() / den_running_mean

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_log_prob",
                        metrics=action_log_probs,
                    )
                    self.tokenwise_tally.add_data(
                        metric_id="engine_next_token_log_prob",
                        metrics=engine_log_probs_mb,
                    )
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_prob",
                        metrics=torch.exp(action_log_probs),
                    )
                    top_k_indices = torch.topk(logits, k=5, dim=-1).indices
                    self.tokenwise_tally.add_data(
                        metric_id=f"top_{5}_tids",
                        metrics=top_k_indices,
                        to_tids=True,
                    )
                    self.tokenwise_tally.add_data(
                        metric_id=f"top_{5}_probs",
                        metrics=torch.exp(log_probs).gather(
                            dim=-1, index=top_k_indices
                        ),
                    )

                rewarded_action_log_probs = (
                    action_mask_mb * credits_mb * action_log_probs
                )
                # (B, S)
                INVALID_LOGPROB = 1.0
                CLAMP_VALUE = 40.0
                masked_action_log_probs = torch.masked_fill(
                    action_log_probs, ~action_mask_mb, INVALID_LOGPROB
                )
                masked_engine_log_probs = torch.masked_fill(
                    engine_log_probs_mb, ~action_mask_mb, INVALID_LOGPROB
                )
                with torch.no_grad():
                    action_engine_log_probs_diff = (
                        masked_action_log_probs - masked_engine_log_probs
                    ).clamp(-CLAMP_VALUE, CLAMP_VALUE)
                running_mean_logs["engine_log_probs_diff_clampfrac"] += (
                    action_engine_log_probs_diff.abs()
                    .eq(CLAMP_VALUE)
                    .float()
                    .sum()
                    .item()
                    / den_running_mean
                )
                if self.importance_sampling_strategy == "per_sequence":
                    tis_imp_ratio = torch.zeros_like(action_engine_log_probs_diff)
                    for mb_idx in range(action_engine_log_probs_diff.shape[0]):
                        valid_token_mask = action_mask_mb[mb_idx]
                        timestep_ids = timesteps_mb[mb_idx][valid_token_mask]
                        timestep_logprob_diffs = action_engine_log_probs_diff[mb_idx][
                            valid_token_mask
                        ]
                        max_timestep = int(timestep_ids.max().item()) + 1
                        timestep_sums = torch.zeros(
                            max_timestep,
                            device=action_engine_log_probs_diff.device,
                            dtype=action_engine_log_probs_diff.dtype,
                        )
                        timestep_sums.scatter_add_(
                            0, timestep_ids, timestep_logprob_diffs
                        )
                        timestep_ratios = torch.exp(timestep_sums)
                        tis_imp_ratio[
                            mb_idx, valid_token_mask
                        ] = timestep_ratios.gather(0, timestep_ids)
                else:
                    tis_imp_ratio = torch.exp(action_engine_log_probs_diff)
                running_mean_logs["tis_imp_ratio"] += (
                    tis_imp_ratio * action_mask_mb
                ).sum().item() / den_running_mean
                if self.truncated_importance_sampling_ratio_cap > 0.0:
                    tis_imp_ratio = torch.clamp(
                        tis_imp_ratio, max=self.truncated_importance_sampling_ratio_cap
                    )
                    running_mean_logs["tis_imp_ratio_clampfrac"] += (
                        tis_imp_ratio.eq(self.truncated_importance_sampling_ratio_cap)
                        .float()
                        .sum()
                        .item()
                    ) / den_running_mean
                    rewarded_action_log_probs = (
                        rewarded_action_log_probs * tis_imp_ratio
                    )

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="next_token_clogπ",
                        metrics=rewarded_action_log_probs,
                    )

                # Add value term to loss
                if self.pg_loss_normalization == "batch":
                    nb_act_tokens = action_mask_mb.sum()
                    mb_value = -rewarded_action_log_probs.sum() / nb_act_tokens
                else:
                    mb_value = -rewarded_action_log_probs.sum()

                loss += mb_value
                running_mean_logs["rl_objective"] += mb_value.item() / den_running_mean

                # -------------------------------------------------
                # Entropy Regularization
                # -------------------------------------------------
                # Only apply entropy on distribution defined over most probable tokens
                if self.entropy_topk is not None:
                    top_k_indices = torch.topk(
                        logits, k=self.entropy_topk, dim=-1
                    ).indices
                    entropy_logits = logits.gather(dim=-1, index=top_k_indices)
                else:
                    entropy_logits = logits

                token_entropy_terms = -F.softmax(
                    entropy_logits, dim=-1
                ) * F.log_softmax(
                    entropy_logits, dim=-1
                )  # (B, S, T)
                token_entropy_terms *= (
                    action_mask_mb[:, :, None] * entropy_mask_mb[:, :, None]
                )  # only get loss on specific action tokens

                mb_entropy = token_entropy_terms.sum(dim=-1)

                if self.enable_tokenwise_logging:
                    self.tokenwise_tally.add_data(
                        metric_id="entropy",
                        metrics=mb_entropy,
                    )
                if self.pg_loss_normalization == "batch":
                    nb_act_tokens = action_mask_mb.sum()
                    mb_entropy = -mb_entropy.sum() / nb_act_tokens
                else:
                    mb_entropy = -mb_entropy.sum()
                running_mean_logs["entropy"] += -mb_entropy.item() / den_running_mean
                if self.entropy_coeff != 0.0:
                    mb_entropy *= self.entropy_coeff
                    loss += mb_entropy

                # -------------------------------------------------
                # KL-DIVERGENCE
                # -------------------------------------------------
                if self.kl_coeff != 0.0:
                    ref_model_logits = self.policy.get_base_model_logits(contexts_mb)
                    ref_model_logits = ref_model_logits / self.temperature
                    # (B, S, V)
                    ref_model_logits = self.mask_non_restricted_token_logits(
                        logits=ref_model_logits
                    )
                    # (B, S, V)
                    ref_model_log_probs = F.log_softmax(ref_model_logits, dim=-1)
                    # (B, S, V)
                    ref_model_action_log_probs = ref_model_log_probs.gather(
                        dim=-1, index=shifted_contexts_mb.unsqueeze(-1)
                    ).squeeze(
                        -1
                    )  # (B,S)
                    # Approximating KL Divergence (see refs in docstring)
                    # Ref 1: http://joschu.net/blog/kl-approx.html
                    # Ref 2: https://github.dev/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1332
                    masked_ref_model_action_log_probs = torch.masked_fill(
                        ref_model_action_log_probs, ~action_mask_mb, INVALID_LOGPROB
                    )
                    action_log_probs_diff = (
                        masked_ref_model_action_log_probs - masked_action_log_probs
                    ).clamp(-CLAMP_VALUE, CLAMP_VALUE)
                    running_mean_logs["ref_log_probs_diff_clampfrac"] += (
                        action_log_probs_diff.abs().eq(CLAMP_VALUE).float().sum().item()
                        / den_running_mean
                    )
                    if self.filter_higher_refprob_tokens_kl:
                        higher_refprob_tokens_mask = action_log_probs_diff > 0.0
                        running_mean_logs["higher_refprob_frac"] += (
                            higher_refprob_tokens_mask.sum().item() / den_running_mean
                        )
                        action_log_probs_diff = action_log_probs_diff * (
                            ~higher_refprob_tokens_mask
                        )
                    kl_div = torch.expm1(action_log_probs_diff) - action_log_probs_diff
                    kl_div *= action_mask_mb  # We only care about KLD of action tokens
                    if self.truncated_importance_sampling_ratio_cap > 0.0:
                        kl_div = kl_div * tis_imp_ratio
                    kl_div *= self.kl_coeff
                    if self.enable_tokenwise_logging:
                        self.tokenwise_tally.add_data(
                            metric_id="ref_model_next_token_log_prob",
                            metrics=ref_model_action_log_probs,
                        )
                        self.tokenwise_tally.add_data(
                            metric_id="kl_divergence",
                            metrics=kl_div,
                        )

                    if self.pg_loss_normalization == "batch":
                        nb_act_tokens = action_mask_mb.sum()
                        mb_kl = kl_div.sum() / nb_act_tokens
                    else:
                        mb_kl = kl_div.sum()
                    running_mean_logs["kl_divergence"] += (
                        mb_kl.item() / den_running_mean
                    )
                    loss += mb_kl

                # Accumulate gradient
                running_mean_logs["policy_gradient_loss"] += (
                    loss.item() / den_running_mean
                )
                loss /= normalization_factor
                self.accelerator.backward(loss)

                # ensure gpu memory is freed
                del training_mb
                del log_probs
                del logits
                del loss
                del action_log_probs
                del rewarded_action_log_probs

            logger.info(
                f"Accumulated the policy gradient loss for {total_tokens_generated} tokens."
            )

            # Clip gradients and take step
            if self.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.policy.parameters(), self.gradient_clipping
                )
                running_mean_logs["policy_gradient_norm"] += grad_norm.item()

            # Take step
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()

            # Store logs
            for key, value in running_mean_logs.items():
                self.tally.add_metric(path=key, metric=value)

            # Clear accelerator state so we do not accumulate references between optimizer steps.
            self.accelerator.clear(self.policy, self.policy_optimizer)
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            return running_mean_logs

    def get_advantages_with_critic_gradient_accumulation(
        self, trajectories: TrajectoryBatch, critic_loss_scaling_factor: float = 2.0
    ) -> torch.FloatTensor:
        """
        Compute (and optionally whiten) advantages while training the critic in mini-batches.
        Uses GAE if enabled, otherwise uses Monte Carlo returns.
        Optionally trains the critic if GAE is used.
        Returns:
            advantages: NestedFloatTensors
        """

        mb_size = self.mini_batch_size
        batch_size = trajectories.rollout_ids.shape[0]
        agent_id = trajectories.agent_ids[0]
        batch_rewards = trajectories.batch_rewards

        ######################################
        # use critic for advantage estimation
        ######################################
        if self.use_gae:
            if "buffer" in agent_id:
                self.critic.eval()
                training = False
            else:
                self.critic.train()
                training = True
            advantages = []
            # critic_loss_scaling_factor comes learning single critic for two agents
            normalization_factor = (
                np.ceil(batch_size / mb_size).astype(int) * critic_loss_scaling_factor
            )
            # For each minibatch
            for mb in range(0, batch_size, mb_size):
                trajectory_mb = trajectories[mb : mb + mb_size]
                trajectory_mb.to(self.device)
                rewards_mb = trajectory_mb.batch_rewards
                (
                    tokens_mb,
                    state_ends_mask_mb,
                    timestep_counts,
                ) = trajectory_mb.get_padded_tensors_for_critic()
                # critic causal attention up to end flags
                if training:
                    vals_estimate_full = self.critic(tokens_mb)
                else:
                    with torch.no_grad():
                        vals_estimate_full = self.critic(tokens_mb)

                # if vals_estimate_full.dim() == 3:
                #     vals_estimate_full = vals_estimate_full.squeeze(-1)

                # Select only positions where states end, per sample → list of (jT,)
                B = tokens_mb.shape[0]
                vals_list = [
                    vals_estimate_full[b][state_ends_mask_mb[b]] for b in range(B)
                ]

                # Pad to (B, max_jT) = (B, S)
                vals_estimate_mb = pad_sequence(
                    vals_list, batch_first=True, padding_value=0.0
                )
                dtype = vals_estimate_mb.dtype
                rewards_mb = pad_sequence(
                    rewards_mb, batch_first=True, padding_value=0.0
                ).to(
                    dtype=dtype
                )  # (B, S)
                self.rollout_tally.add_metric(
                    path=["batch_rewards"],
                    rollout_tally_item=RolloutTallyItem(
                        crn_ids=trajectory_mb.crn_ids,
                        rollout_ids=trajectory_mb.rollout_ids,
                        agent_ids=trajectory_mb.agent_ids,
                        metric_matrix=rewards_mb,
                    ),
                )
                if self.reward_normalizing_constant != 1.0:
                    rewards_mb /= self.reward_normalizing_constant

                det_vals_estimate_mb = vals_estimate_mb.detach()  # (B, max_jT)
                self.rollout_tally.add_metric(
                    path=["mb_value_estimates_critic"],
                    rollout_tally_item=RolloutTallyItem(
                        crn_ids=trajectory_mb.crn_ids,
                        rollout_ids=trajectory_mb.rollout_ids,
                        agent_ids=trajectory_mb.agent_ids,
                        metric_matrix=det_vals_estimate_mb,
                    ),
                )

                # Append a 0 value to the end of the value estimates
                if det_vals_estimate_mb.shape[1] == rewards_mb.shape[1]:
                    Bsize = det_vals_estimate_mb.shape[0]
                    device = det_vals_estimate_mb.device
                    dtype = det_vals_estimate_mb.dtype
                    det_vals_estimate_mb = torch.cat(
                        [
                            det_vals_estimate_mb,
                            torch.zeros((Bsize, 1), device=device, dtype=dtype),
                        ],
                        dim=1,
                    )  # (B, max_jT+1)
                else:
                    raise ValueError(
                        "Incompatible shapes for value estimates and rewards."
                    )

                # Get annealed lambda
                if self.use_gae_lambda_annealing:
                    annealing_constant = self.gae_lambda_annealing_method(
                        step=self.trainer_annealing_state.annealing_step_counter
                    )
                    annealed_lambda = (
                        self.gae_lambda_annealing_limit * annealing_constant
                    )
                    self.tally.add_metric(
                        path="annealed_lambda", metric=annealed_lambda
                    )
                else:
                    annealed_lambda = self.gae_lambda_annealing_limit

                # Get GAE advantages
                gae_advantages = get_generalized_advantage_estimates(
                    rewards=rewards_mb,
                    value_estimates=det_vals_estimate_mb,
                    discount_factor=self.discount_factor,
                    lambda_coef=annealed_lambda,
                )  # (B, max_jT)
                self.rollout_tally.add_metric(
                    path=["mb_gae_advantages"],
                    rollout_tally_item=RolloutTallyItem(
                        crn_ids=trajectory_mb.crn_ids,
                        rollout_ids=trajectory_mb.rollout_ids,
                        agent_ids=trajectory_mb.agent_ids,
                        metric_matrix=gae_advantages,
                    ),
                )
                if training:
                    targets = (
                        gae_advantages.to(dtype=dtype) + det_vals_estimate_mb[:, :-1]
                    )  # (B, max_jT) # A(s, a, b) + V(s) = Q(s, a, b)
                    self.rollout_tally.add_metric(
                        path=["mb_targets_critic"],
                        rollout_tally_item=RolloutTallyItem(
                            crn_ids=trajectory_mb.crn_ids,
                            rollout_ids=trajectory_mb.rollout_ids,
                            agent_ids=trajectory_mb.agent_ids,
                            metric_matrix=targets,
                        ),
                    )
                    if self.critic_loss_type == "mse":
                        loss = F.mse_loss(
                            input=vals_estimate_mb,
                            target=targets,
                        )
                    elif self.critic_loss_type == "huber":
                        loss = F.huber_loss(
                            input=vals_estimate_mb,
                            target=targets,
                        )
                    self.tally.add_metric(path=["mb_critic_loss"], metric=loss.item())
                    # Accumulate gradient
                    loss /= normalization_factor
                    self.accelerator.backward(loss)
                    del loss
                    del targets
                    del vals_estimate_mb
                del trajectory_mb
                del vals_estimate_full

                # Get jagged back using timestep_counts
                advantages.extend(
                    [gae_advantages[i, : timestep_counts[i]] for i in range(B)]
                )

        ######################################
        # use exclusively Monte Carlo returns & rloo for advantage estimation
        ######################################
        else:
            lengths = [len(c) for c in batch_rewards]
            padded_rewards = pad_sequence(
                batch_rewards, batch_first=True, padding_value=0.0
            )
            self.rollout_tally.add_metric(
                path=["mb_rewards"],
                rollout_tally_item=RolloutTallyItem(
                    crn_ids=trajectories.crn_ids,
                    rollout_ids=trajectories.rollout_ids,
                    agent_ids=trajectories.agent_ids,
                    metric_matrix=padded_rewards,
                ),
            )
            if self.reward_normalizing_constant != 1.0:
                padded_rewards /= self.reward_normalizing_constant
            padded_advantages = get_discounted_returns(
                rewards=padded_rewards,
                discount_factor=self.discount_factor,
            )  # no baseline for now
            if self.use_rloo:
                is_grouped_by_rng = (
                    trajectories.crn_ids.unique().shape[0]
                    != trajectories.crn_ids.shape[0]
                )
                if is_grouped_by_rng and not self.no_rloo_grouping:
                    for crn_id in trajectories.crn_ids.unique():
                        rng_mask = trajectories.crn_ids == crn_id
                        rng_advantages = padded_advantages[rng_mask]
                        rng_advantages, _ = get_rloo_credits(credits=rng_advantages)
                        padded_advantages[rng_mask] = rng_advantages
                else:
                    padded_advantages, _ = get_rloo_credits(credits=padded_advantages)
                self.rollout_tally.add_metric(
                    path=["mb_rloo_advantages"],
                    rollout_tally_item=RolloutTallyItem(
                        crn_ids=trajectories.crn_ids,
                        rollout_ids=trajectories.rollout_ids,
                        agent_ids=trajectories.agent_ids,
                        metric_matrix=padded_advantages,
                    ),
                )
            advantages = [
                padded_advantages[i, : lengths[i]]
                for i in range(padded_advantages.shape[0])
            ]

        if self.whiten_advantages_time_step_wise or self.whiten_advantages:
            lengths = [len(c) for c in advantages]
            padded_advantages = pad_sequence(
                advantages, batch_first=True, padding_value=0.0
            )
            if self.whiten_advantages_time_step_wise:
                whitened_padded_advantages = whiten_advantages_time_step_wise(
                    padded_advantages
                )
                path = ["mb_whitened_advantages_time_step_wise"]
            elif self.whiten_advantages:
                whitened_padded_advantages = whiten_advantages(padded_advantages)
                path = ["mb_whitened_advantages"]
            self.rollout_tally.add_metric(
                path=path,
                rollout_tally_item=RolloutTallyItem(
                    crn_ids=trajectories.crn_ids,
                    rollout_ids=trajectories.rollout_ids,
                    agent_ids=trajectories.agent_ids,
                    metric_matrix=whitened_padded_advantages,
                ),
            )
            advantages = [
                whitened_padded_advantages[i, : lengths[i]]
                for i in range(whitened_padded_advantages.shape[0])
            ]

        self.trainer_annealing_state.annealing_step_counter += 1

        return advantages

    @abstractmethod
    def set_agent_trajectory_data(
        self, agent_id: str, roots: list[RolloutTreeRootNode]
    ) -> None:
        """
        Populate self.training_data for a single agent using the provided rollout trees.
        """
        pass

    def set_trajectory_data(
        self, roots: list[RolloutTreeRootNode], agent_ids: list[str]
    ) -> None:
        """
        Convenience wrapper to ingest trajectory data for every training agent.
        """
        for agent_id in agent_ids:
            self.set_agent_trajectory_data(agent_id, roots)

    @abstractmethod
    def share_advantage_data(self) -> list[AdvantagePacket]:
        pass

    @abstractmethod
    def receive_advantage_data(self, advantage_packets: list[AdvantagePacket]) -> None:
        pass

    def set_policy_gradient_data(self, agent_ids: list[str]) -> None:
        """
        Reset and rebuild the policy-gradient minibatches before iterating through agents.
        """
        self.policy_gradient_data = None
        for agent_id in agent_ids:
            assert "buffer" not in agent_id, "Buffer agents do not train policy"
            trajectory_batch = self.training_data[agent_id]
            tokenwise_batch_credits = get_tokenwise_credits(
                batch_timesteps=trajectory_batch.batch_timesteps,
                batch_credits=trajectory_batch.batch_credits,
            )
            policy_gradient_data = TrainingBatch(
                rollout_ids=trajectory_batch.rollout_ids,
                batch_input_ids=trajectory_batch.batch_input_ids,
                batch_action_mask=trajectory_batch.batch_action_mask,
                batch_entropy_mask=trajectory_batch.batch_entropy_mask,
                batch_credits=tokenwise_batch_credits,
                batch_engine_log_probs=trajectory_batch.batch_engine_log_probs,
                batch_timesteps=trajectory_batch.batch_timesteps,
            )
            if self.policy_gradient_data is None:
                self.policy_gradient_data = policy_gradient_data
            else:
                self.policy_gradient_data.append(policy_gradient_data)

        self.training_data = {}
        self.tokenwise_tally = ContextualizedTokenwiseTally(
            tokenizer=self.tokenizer,
            paths=self.debug_path_list,
        )

    def train(self) -> None:
        """
        Entry point for policy updates: prepare batches, compute gradients, and update parameters.
        """
        assert self.policy_gradient_data is not None, "Policy gradient data is not set"
        if self.critic_optimizer is not None:
            if self.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.critic.parameters(), self.gradient_clipping
                )
                self.tally.add_metric(
                    path="gradient_norm_critic", metric=grad_norm.item()
                )
            # Take step
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            self.accelerator.clear(self.critic, self.critic_optimizer)
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        running_mean_logs = self.apply_reinforce_step(
            training_batch=self.policy_gradient_data
        )
        return running_mean_logs

    def export_training_tally(self, identifier: str, folder: str) -> None:
        """
        Saves and resets the collected training metrics using the tally object.
        """
        os.makedirs(folder, exist_ok=True)
        self.tally.save(identifier=identifier, folder=folder)
        self.tokenwise_tally.save(
            path=os.path.join(folder, f"{identifier}_tokenwise.csv")
        )
        self.rollout_tally.save(identifier=identifier, folder=folder)
        self.tally.reset()
        self.tokenwise_tally = None
        self.rollout_tally.reset()
        self.debug_path_list = []

    def export_optimizer_states(self) -> None:
        """
        Saves the optimizer states for both the main model and critic (if it exists).
        """
        try:
            os.makedirs(self.save_path, exist_ok=True)

            torch.save(self.policy_optimizer.state_dict(), self.policy_optimizer_path)
            logger.info(f"Saved main optimizer state to {self.policy_optimizer_path}")

            if self.critic_optimizer is not None:
                torch.save(
                    self.critic_optimizer.state_dict(), self.critic_optimizer_path
                )
                logger.info(
                    f"Saved critic optimizer state to {self.critic_optimizer_path}"
                )
        except Exception as e:
            logger.error(f"Error saving optimizer states: {str(e)}")
            raise

    def export_trainer_annealing_state(self) -> None:
        """
        Saves the trainer state.
        """
        with open(self.trainer_annealing_state_path, "wb") as f:
            pickle.dump(self.trainer_annealing_state, f)
        logger.info(f"Saved trainer state to {self.trainer_annealing_state_path}")

    def export_trainer_states(self) -> None:
        """
        Saves the trainer states.
        """
        self.export_optimizer_states()
        self.export_trainer_annealing_state()
