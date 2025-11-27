"""
File: mllm/markov_games/negotiation/no_press_nego_simulation.py
Summary: Simulation driver for no-press negotiation scenarios.
"""

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

from mllm.markov_games.negotiation.nego_simulation import (
    NegotiationObs,
    NegotiationSimulation,
    NegotiationState,
    Split,
    compute_tas_style_rewards,
)

AgentId = str


@dataclass
class NoPressState(NegotiationState):
    """NegotiationState alias used to clarify we run in always-split phase."""

    pass


@dataclass
class NoPressObs(NegotiationObs):
    """Observation that includes both agents' values (since there is no messaging)."""

    other_value: Dict[str, float]


class NoPressSimulation(NegotiationSimulation):
    def __init__(
        self,
        game_type: Literal["10-1-exclusive", "10-1-ties", "1-to-20"] = "1-to-20",
        same_round_value: bool = True,
        atleast_one_conflict: bool = False,
        *args,
        **kwargs,
    ):
        self.game_type = game_type
        self.same_round_value = same_round_value
        self.atleast_one_conflict = atleast_one_conflict
        super().__init__(*args, **kwargs)

    def _sample_values(self) -> Dict[AgentId, dict]:
        """Sample per-item valuations according to the configured template."""
        values = defaultdict(dict)
        if self.state is None:
            item_types = self.item_types
        else:
            item_types = list(self.state.quantities.keys())
        while True:
            for item in item_types:
                if self.game_type == "10-1-exclusive":
                    v = int(self.rng.choice([1, 10]))
                    values[self.agent_ids[0]][item] = v
                    values[self.agent_ids[1]][item] = 10 if v == 1 else 1
                elif self.game_type == "10-1-ties":
                    for aid in self.agent_ids:
                        values[aid][item] = int(self.rng.choice([1, 10]))
                elif self.game_type == "1-to-20":
                    for aid in self.agent_ids:
                        values[aid][item] = int(self.rng.integers(1, 21))
            if self.atleast_one_conflict:
                has_conflict = False
                for item in item_types:
                    agent_values_for_item = [
                        values[aid][item] for aid in self.agent_ids
                    ]
                    if len(set(agent_values_for_item)) > 1:
                        has_conflict = True
                        break
                if not has_conflict:
                    continue
            agent_values = [sum(v.values()) for v in values.values()]
            if len(set(agent_values)) == 1 or not self.same_round_value:
                break
        return values

    def _sample_quantities(self) -> Dict[str, int]:
        """No-press setups use symmetric 10-unit stocks for every item."""
        return {item.lower(): 10 for item in self.item_types}

    def set_new_round_of_variant(self):
        """Refresh quantities/values and jump directly into the simultaneous split."""
        self.state.quantities = self._sample_quantities()
        self.state.values = self._sample_values()
        self.state.split_phase = True

    def get_info_of_variant(
        self, state: NegotiationState, actions: Dict[AgentId, Any]
    ) -> Dict[str, Any]:
        """Surface quantities/values/splits so statistics modules can read them."""
        return {
            "quantities": copy.deepcopy(state.quantities),
            "values": copy.deepcopy(state.values),
            "splits": copy.deepcopy(state.splits),
        }

    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        """Reuse TAS reward logic because the split arbitration is identical."""
        return compute_tas_style_rewards(
            self.agent_ids, self.state.values, splits, self.state.quantities
        )

    def get_obs(self):
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

    def get_obs_agent(self, agent_id):
        other_id = self._other(agent_id)
        last_value_coagent = (
            None
            if self.state.previous_values is None
            else self.state.previous_values.get(other_id)
        )
        last_points_coagent = (
            None
            if self.state.previous_points is None
            else round(self.state.previous_points.get(other_id), 1)
        )
        last_value_agent = (
            None
            if self.state.previous_values is None
            else self.state.previous_values.get(agent_id)
        )
        last_points_agent = (
            None
            if self.state.previous_points is None
            else round(self.state.previous_points.get(agent_id), 1)
        )
        last_split_coagent = None
        last_split_agent = None
        if self.state.previous_splits is not None:
            last_split_coagent = self.state.previous_splits[
                other_id
            ].items_given_to_self
            last_split_agent = self.state.previous_splits[agent_id].items_given_to_self
        obs = NoPressObs(
            round_nb=self.state.round_nb,
            last_message="",
            quota_messages_per_agent_per_round=self.quota_messages_per_agent_per_round,
            current_agent=self.state.current_agent,
            other_agent=self.agent_id_to_name[other_id],
            quantities=self.state.quantities,
            item_types=self.item_types,
            value=self.state.values[agent_id],
            split_phase=self.state.split_phase,
            last_split_agent=last_split_agent,
            last_value_agent=last_value_agent,
            last_points_agent=last_points_agent,
            last_split_coagent=last_split_coagent,
            last_value_coagent=last_value_coagent,
            last_points_coagent=last_points_coagent,
            other_value=self.state.values[other_id],
            last_quantities=self.state.previous_quantities,
        )
        return obs

    def reset(self):
        start_agent = self.agent_ids[self._starting_agent_index]
        quantities = self._sample_quantities()
        values = self._sample_values()
        self.state = NoPressState(
            round_nb=0,
            last_message="",
            current_agent=start_agent,
            quantities=quantities,
            values=values,
            previous_values=None,
            splits={aid: None for aid in self.agent_ids},
            nb_messages_sent={aid: 0 for aid in self.agent_ids},
            split_phase=True,
            previous_splits=None,
            previous_points=None,
            previous_quantities=None,
        )
        return self.get_obs()
