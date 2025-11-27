"""
File: mllm/markov_games/ipd/ipd_agent.py
Summary: Implements the IPD agent abstraction used during simulations.
"""

import copy
import json
import random
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


@dataclass
class IPDAgentState:
    """
    Tracks retry count, round index, and chat history for a single IPD agent.
    """

    nb_retries: int
    round_nb: int
    chat_counter: int
    chat_history: List[ChatTurn]


@dataclass
class IPDAgent(Agent):
    seed: int
    agent_id: str
    agent_name: str
    policy: Callable[[List[Dict]], str]
    intro_prompt: str  # Introduction prompt explaining the game rules
    goal_prompt: str  # Prompt explaining the agent's goal
    strategy_prompt: str  # Prompt suggesting a strategy to the agent
    max_errors: int  # Maximum number of errors allowed before default action
    allow_reasoning: bool  # Whether to allow reasoning in the response
    max_reasoning_chars: int  # Maximum number of characters for reasoning
    cooperate_string: str  # string parsed as playing cooperate by simulation
    defect_string: str  # string parsed as playing defect by simulation

    def __post_init__(self):
        self.state = IPDAgentState(
            nb_retries=0, round_nb=0, chat_counter=0, chat_history=[]
        )

    async def act(self, observation) -> Tuple[Any, AgentActLog]:
        """
        Run the LLM policy conversation until a valid cooperate/defect action is produced.
        """

        action = None
        action_is_ready = False
        round_nb = observation.round_nb

        # If it's the first round, we need to send the intro prompt
        if round_nb == 0 and self.state.chat_counter == 0:
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="user",
                    content=self.intro_prompt,
                    is_state_end=True,
                )
            )

        # If new round
        if round_nb > self.state.round_nb:
            coagent_action = observation.last_coagent_move
            user_message = f"Last round, the other agent played {coagent_action}."
            self.state.chat_history.append(
                ChatTurn(
                    agent_id=self.agent_id,
                    role="user",
                    content=user_message,
                    is_state_end=True,
                )
            )

        # If not new round, try to get valid action from policy
        output_chat_turn: ChatTurn = await self.policy(
            state=self.state.chat_history,
            agent_id=self.agent_id,
            regex=f"({self.cooperate_string}|{self.defect_string})",
        )
        self.state.chat_history.append(output_chat_turn)
        action = output_chat_turn.content

        agent_step_log = AgentActLog(
            chat_turns=self.state.chat_history[self.state.chat_counter :], info=None
        )
        self.state.chat_counter = len(self.state.chat_history)
        self.state.round_nb = round_nb

        return action, agent_step_log

    def get_safe_copy(self):
        """
        Return a safe copy of the agent.
        """
        agent_copy = copy.copy(self)
        agent_copy.state = copy.deepcopy(self.state)
        return agent_copy

    def reset(self):
        self.state = IPDAgentState()
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def get_agent_info(self):
        pass
