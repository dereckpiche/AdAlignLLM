"""
File: mllm/markov_games/negotiation/nego_agent.py
Summary: General-purpose negotiation agent coordinating prompts and actions.
"""

import copy
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from mllm.markov_games.agent import Agent
from mllm.markov_games.negotiation.nego_simulation import Message, NegotiationObs, Split
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


@dataclass
class NegotiationAgentState:
    """Lightweight container tracking round progression and message history."""

    round_nb: int
    nb_messages_sent_this_round: int
    chat_counter: int
    chat_history: List[ChatTurn]


class NegotiationAgent(Agent):
    """Base agent that manages prompt scaffolding and regex validation for variants."""

    def __init__(
        self,
        seed: int,
        agent_id: str,
        agent_name: str,
        policy: Callable[[List[Dict]], str],
        goal: str,
        exploration_prompts: List[str] = [],
        exploration_prompt_probs: List[float] = [],
    ):
        self.seed = seed
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.policy = policy
        self.goal = goal
        self.exploration_prompts_toggled = len(exploration_prompts) > 0
        if self.exploration_prompts_toggled:
            exploration_prompts = copy.deepcopy(exploration_prompts)
            exploration_prompts.append(None)
            self.exploration_prompts = exploration_prompts
            self.exploration_prompt_probs = np.array(exploration_prompt_probs)
            assert self.exploration_prompt_probs.sum() <= 1
            assert np.all(self.exploration_prompt_probs >= 0)
            self.exploration_prompt_probs = np.append(
                self.exploration_prompt_probs, 1 - self.exploration_prompt_probs.sum()
            )
        self.state = NegotiationAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )

        # Implemented in variants
        self.intro_prompt = ""
        self.new_round_prompt = ""
        self.last_round_prompt = ""
        self.send_split_prompt = ""
        self.wait_for_message_prompt = ""
        self.last_message_prompt = ""
        self.send_message_prompt = ""

    @abstractmethod
    def get_message_regex(self, observation: NegotiationObs) -> str:
        """Return the regex that outgoing chat messages must satisfy."""
        pass

    @abstractmethod
    def get_split_regex(self, observation: NegotiationObs) -> str:
        """Return the regex that final split proposals must satisfy."""
        pass

    @abstractmethod
    def get_split_action(
        self, policy_output: str, observation: NegotiationObs
    ) -> Split:
        """Convert raw LLM output into the ``Split`` structure required by simulations."""
        pass

    async def act(self, observation: NegotiationObs) -> Tuple[Any, AgentActLog]:
        """
        Assemble the appropriate prompt, query the policy, and return message or split.

        This handles intro text, new-round reminders, quota tracking, and post-processing
        (regex enforcement + ChatTurn logging) so subclasses only customize prompts/regexes.
        """

        def dict_to_str(d: dict) -> str:
            return ", ".join(f"{v} {k}" for k, v in d.items())

        def dict_to_eq_str(d: dict) -> str:
            return ", ".join(f"{k}={v}" for k, v in d.items())

        is_our_turn = observation.current_agent == self.agent_id
        action: Any = None
        round_nb = observation.round_nb

        prompt_parts: List[str] = []
        obs_ctx = vars(observation)
        obs_ctx_formmated = obs_ctx.copy()
        for key in obs_ctx_formmated:
            if isinstance(obs_ctx_formmated[key], dict) and "value" not in key:
                obs_ctx_formmated[key] = dict_to_str(obs_ctx_formmated[key])
            elif isinstance(obs_ctx_formmated[key], dict) and "value" in key:
                obs_ctx_formmated[key] = dict_to_eq_str(obs_ctx_formmated[key])

        #######################################
        # build user prompt
        #######################################

        # First-ever call
        is_intro = round_nb == 0 and self.state.chat_counter == 0
        if is_intro:
            prompt_parts.append(
                self.intro_prompt.format(
                    goal=self.goal, agent=self.agent_name, **obs_ctx_formmated
                )
            )

        # New round
        is_new_round = round_nb > self.state.round_nb
        if is_new_round or is_intro:
            self.state.nb_messages_sent_this_round = 0
            if not is_intro:
                prompt_parts.append(self.last_round_prompt.format(**obs_ctx_formmated))
            prompt_parts.append(self.new_round_prompt.format(**obs_ctx_formmated))
            if self.exploration_prompts_toggled:
                exploration_prompt = self.exploration_prompts[
                    np.random.choice(
                        len(self.exploration_prompts), p=self.exploration_prompt_probs
                    )
                ]
                if exploration_prompt is not None:
                    prompt_parts.append(exploration_prompt)
            self.state.round_nb = round_nb

        # Wait for message
        if not is_our_turn and not observation.split_phase:
            prompt_parts.append(
                self.wait_for_message_prompt.format(**obs_ctx_formmated)
            )

        # Get last message
        if is_our_turn and not is_new_round and not is_intro:
            prompt_parts.append(self.last_message_prompt.format(**obs_ctx_formmated))

        # Prompt to send message
        must_send_message = not observation.split_phase and is_our_turn
        if must_send_message:
            prompt_parts.append(self.send_message_prompt.format(**obs_ctx_formmated))

        # Prompt to give split
        must_send_split = not must_send_message and observation.split_phase
        if must_send_split:
            var_names = ["x", "y", "z", "w"]  # Extend as needed
            items_str = ", ".join(
                [
                    f"{var_names[i]} {item}"
                    for i, item in enumerate(obs_ctx["quantities"].keys())
                ]
            )
            ranges_str = ", ".join(
                [
                    f"{var_names[i]}: 0-{obs_ctx['quantities'][item]} (integer)"
                    for i, item in enumerate(obs_ctx["quantities"].keys())
                ]
            )
            proposal_style = f"Proposal: {items_str} where {ranges_str}."
            proposal_style2 = (
                f"<items_to_self> {items_str} </items_to_self> where {ranges_str}."
            )
            prompt_parts.append(
                self.send_split_prompt.format(
                    proposal_style=proposal_style,
                    proposal_style2=proposal_style2,
                    **obs_ctx_formmated,
                )
            )

        # Append one ChatTurn with is_state_end=True
        user_prompt = "\n".join(prompt_parts)
        self.state.chat_history.append(
            ChatTurn(
                agent_id=self.agent_id,
                role="user",
                content=user_prompt,
                is_state_end=True,
            )
        )

        #######################################
        # Get policy action
        #######################################

        # Query policy for the appropriate format
        if must_send_message:
            return_regex = self.get_message_regex(observation)
            policy_output = await self.policy(
                state=self.state.chat_history,
                agent_id=self.agent_id,
                regex=return_regex,
            )
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output.content,
                    reasoning_content=policy_output.reasoning_content,
                    log_probs=policy_output.log_probs,
                    out_token_ids=policy_output.out_token_ids,
                    is_state_end=False,
                )
            )
            action = Message(message=policy_output.content)
            self.state.nb_messages_sent_this_round += 1

        elif must_send_split:
            return_regex = self.get_split_regex(observation)
            policy_output = await self.policy(
                state=self.state.chat_history,
                agent_id=self.agent_id,
                regex=return_regex,
            )
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content=policy_output.content,
                    reasoning_content=policy_output.reasoning_content,
                    log_probs=policy_output.log_probs,
                    out_token_ids=policy_output.out_token_ids,
                    is_state_end=False,
                )
            )
            action = self.get_split_action(policy_output.content, observation)
        else:
            action = None

        agent_step_log = AgentActLog(
            chat_turns=self.state.chat_history[self.state.chat_counter :], info=None
        )
        self.state.chat_counter = len(self.state.chat_history)
        return action, agent_step_log

    def get_safe_copy(self):
        agent_copy = copy.copy(self)
        agent_copy.state = copy.deepcopy(self.state)
        return agent_copy

    def reset(self):
        self.state = NegotiationAgentState(
            round_nb=0, nb_messages_sent_this_round=0, chat_counter=0, chat_history=[]
        )
