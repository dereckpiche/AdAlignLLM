"""
File: mllm/markov_games/ipd/Ipd_hard_coded_agents.py
Summary: Contains hand-crafted IPD policies used as deterministic baselines.
"""

from dataclasses import dataclass
from typing import Any, Tuple

from mllm.markov_games.ipd.ipd_agent import IPDAgent
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


@dataclass
class AlwaysCooperateIPDAgent(IPDAgent):
    async def act(self, observation) -> Tuple[Any, AgentActLog]:
        """
        Always plays the cooperate action, ignoring observation.
        Returns the configured cooperate_string so the simulation parses it as "C".
        """

        action = self.cooperate_string

        # Log a minimal, structured chat turn for consistency with other agents
        turn_text = f"Playing cooperate: {action}"
        self.state.chat_history.append(
            ChatTurn(
                agent_id=self.agent_id,
                role="assistant",
                content=turn_text,
                is_state_end=True,
            )
        )

        act_log = AgentActLog(
            chat_turns=[self.state.chat_history[-1]],
            info=None,
        )

        # Advance internal counters similar to IPDAgent semantics
        self.state.chat_counter = len(self.state.chat_history)
        self.state.round_nb = observation.round_nb

        return action, act_log


@dataclass
class AlwaysDefectIPDAgent(IPDAgent):
    async def act(self, observation) -> Tuple[Any, AgentActLog]:
        """
        Always plays the defect action, ignoring observation.
        Returns the configured defect_string so the simulation parses it as "D".
        """

        action = self.defect_string

        # Log a minimal, structured chat turn for consistency with other agents
        turn_text = f"Playing defect: {action}"
        self.state.chat_history.append(
            ChatTurn(
                agent_id=self.agent_id,
                role="assistant",
                content=turn_text,
                is_state_end=True,
            )
        )

        act_log = AgentActLog(
            chat_turns=[self.state.chat_history[-1]],
            info=None,
        )

        # Advance internal counters similar to IPDAgent semantics
        self.state.chat_counter = len(self.state.chat_history)
        self.state.round_nb = observation.round_nb

        return action, act_log
