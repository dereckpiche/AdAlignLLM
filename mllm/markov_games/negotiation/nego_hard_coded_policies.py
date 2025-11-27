"""
File: mllm/markov_games/negotiation/nego_hard_coded_policies.py
Summary: Provides deterministic negotiation policies for testing and baselines.
"""

import asyncio
from typing import Any, Optional, Tuple

from mllm.markov_games.negotiation.nego_agent import NegotiationAgent
from mllm.markov_games.negotiation.nego_simulation import Split
from mllm.markov_games.negotiation.no_press_nego_agent import NoPressAgent
from mllm.markov_games.negotiation.no_press_nego_simulation import NoPressObs
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


class HardCodedNegoWelfareMaximizingPolicy(NoPressAgent):
    async def act(self, observation: NoPressObs) -> Tuple[Any, AgentActLog]:
        """
        Policy that gives all of the items to the agent who values them more.
        If the items are equally valued, give them to the agent who values them more.
        """
        quantities = observation.quantities
        my_values = observation.value
        other_values = observation.other_value

        items_given_to_self = {}
        for item, qty in quantities.items():
            my_v = float(my_values.get(item, 0))
            other_v = float(other_values.get(item, 0))
            if my_v == other_v:
                items_given_to_self[item] = int(qty) / 2
            else:
                items_given_to_self[item] = int(qty if my_v > other_v else 0)

        action = Split(items_given_to_self=items_given_to_self)
        act_log = AgentActLog(
            chat_turns=[
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content="Using welfare-maximizing split (all to higher-value agent).",
                    is_state_end=True,
                )
            ],
            info=None,
        )
        return action, act_log


class HardCodedNegoGreedyPolicy(NoPressAgent):
    async def act(self, observation: NoPressObs) -> Tuple[Any, AgentActLog]:
        """
        Always gives itself all of the items.
        """
        quantities = observation.quantities
        items_given_to_self = {item: int(qty) for item, qty in quantities.items()}

        action = Split(items_given_to_self=items_given_to_self)
        act_log = AgentActLog(
            chat_turns=[
                ChatTurn(
                    agent_id=self.agent_id,
                    role="assistant",
                    content="Using greedy split (keep all items).",
                    is_state_end=True,
                )
            ],
            info=None,
        )
        return action, act_log
