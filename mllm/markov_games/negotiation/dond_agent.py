"""
File: mllm/markov_games/negotiation/dond_agent.py
Summary: Agent implementation for Deal-or-No-Deal style negotiations.
"""

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mllm.markov_games.agent import Agent
from mllm.markov_games.negotiation.dond_simulation import DealNoDealObs
from mllm.markov_games.negotiation.nego_agent import (
    NegotiationAgent,
    NegotiationAgentState,
)
from mllm.markov_games.negotiation.nego_simulation import Split
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


class DealNoDealAgent(NegotiationAgent):
    """NegotiationAgent tailored to the Deal-or-No-Deal stock/value revelation rules."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intro_prompt = (
            "You are {agent_id}. You are playing an iterated game. "
            "At each round, you and other agent will try to distribute among yourselves items of types {item_types}. "
            "You only know how much you value each item type, but not the other agent's values. "
            "You can communicate with the other agent by sending up to {quota_messages_per_agent_per_round} short messages per round. "
            "Each round, after exchanging messages, you and the other agent will submit a private proposal. "
            "A deal is accepted only if both proposals match exactly and are within stock; otherwise no deal (0 points for both at that round). "
            "The values of the items of the other agent at the previous round are revealed to you after each round. "
            "Your goal is: {goal}."
        )
        self.new_round_prompt = (
            "New round {round_nb}. Items: {stock}. Your values: {values}. "
        )
        self.last_round_prompt = (
            "Last round, other agent's values: {previous_values_coagent}. "
        )
        self.send_split_prompt = "Respond with <split>...</split> where you propose how many items of each type you want to keep."

    def get_message_regex(self, observation: DealNoDealObs) -> str:
        """Allow short XML messages (<400 chars) between proposal phases."""
        return r"<message>[\s\S]{0,400}</message>"

    def get_split_regex(self, observation: DealNoDealObs) -> str:
        """Constrain split proposals to per-item XML tags bounded by the current stock."""
        parts = []
        for t in observation.item_types:
            s = int(observation.quantities.get(t, 0))
            allowed = "|".join(str(k) for k in range(0, s + 1))
            rng = f"({allowed})"
            parts.append(rf"<{t}>{rng}</{t}>")
        items_block = "".join(parts)
        return rf"(<split>{items_block}</split>)"

    def get_split_action(self, policy_output: str, observation: DealNoDealObs) -> Split:
        """Convert the XML proposal into a Split dataclass understood by the simulator."""
        import re as _re

        allocations: Dict[str, int] = {}
        for t in observation.item_types:
            m = _re.search(rf"<{t}>([0-9]+)</{t}>", policy_output)
            if m:
                allocations[t] = int(m.group(1))
            else:
                allocations[t] = 0
        return Split(items_given_to_self=allocations)
