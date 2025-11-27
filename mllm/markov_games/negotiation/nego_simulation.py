"""
File: mllm/markov_games/negotiation/nego_simulation.py
Summary: Simulation harness for general negotiation environments.
"""

import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import SimulationStepLog
from mllm.markov_games.simulation import Simulation
from mllm.utils.get_coagent_id import get_coagent_id

AgentId = str


@dataclass
class Split:
    """Structured proposal describing how many units of each item an agent keeps."""

    items_given_to_self: Dict[str, int]


@dataclass
class Message:
    """Single chat utterance exchanged during the negotiation phase."""

    message: str


@dataclass  # gets extended by variants
class NegotiationState:
    """Full simulator state snapshot shared by all negotiation variants."""

    round_nb: int
    last_message: str
    current_agent: AgentId
    quantities: Dict[str, int]
    values: Dict[AgentId, Dict[str, float]]
    splits: Dict[AgentId, Split | None]
    nb_messages_sent: Dict[AgentId, int]
    previous_values: Dict[AgentId, Dict[str, float]] | None
    previous_splits: Dict[AgentId, Dict[str, int] | None] | None
    previous_points: Dict[AgentId, float] | None
    previous_quantities: Dict[str, int] | None
    split_phase: bool


@dataclass  # gets extended by variants
class NegotiationObs:
    """Observation presented to agents each turn (base fields; variants extend)."""

    round_nb: int
    last_message: str
    quota_messages_per_agent_per_round: int
    current_agent: AgentId
    other_agent: str
    quantities: Dict[str, int]
    item_types: List[str]
    value: Dict[str, int]
    split_phase: bool
    last_split_agent: Dict[str, int] | None
    last_value_agent: Dict[str, int] | None
    last_points_agent: float | None
    last_split_coagent: Dict[str, int] | None
    last_value_coagent: Dict[str, int] | None
    last_points_coagent: float | None
    last_quantities: Dict[str, int] | None


def compute_tas_style_rewards(
    agent_ids: List[AgentId],
    values: Dict[AgentId, float],
    splits: Dict[AgentId, Split],
    quantities: Dict[str, int],
) -> Dict[AgentId, float]:
    """
    TAS-like reward computation: if sum of proposed coins exceeds max_coins,
    allocate proportionally. Otherwise, use proposed amounts directly.
    Rewards are quantity_kept * per-coin value for each agent.
    """
    a0, a1 = agent_ids[0], agent_ids[1]
    r0, r1 = 0.0, 0.0

    for item in quantities:
        max_item = quantities[item]
        item_to_self_0 = int(
            (splits[a0].items_given_to_self.get(item, 0))
            if splits[a0] is not None
            else 0
        )
        item_to_self_1 = int(
            (splits[a1].items_given_to_self.get(item, 0))
            if splits[a1] is not None
            else 0
        )
        denom = max(int(max_item), item_to_self_0 + item_to_self_1)
        q0 = float(max_item) * float(item_to_self_0) / float(denom)
        q1 = float(max_item) * float(item_to_self_1) / float(denom)
        if type(values[a0]) is not dict:
            r0 += q0 * float(values[a0])
            r1 += q1 * float(values[a1])
        else:
            r0 += q0 * float(values[a0][item])
            r1 += q1 * float(values[a1][item])
    return {a0: r0, a1: r1}


class NegotiationSimulation(Simulation):
    def __init__(
        self,
        agent_ids: List[AgentId],
        agent_names: List[str],
        seed: int,
        nb_of_rounds: int,
        quota_messages_per_agent_per_round: int,
        item_types: List[str] | None = None,
    ):
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.agent_ids = list(agent_ids)
        self.agent_names = agent_names
        self.agent_id_to_name = {
            agent_id: agent_name for agent_id, agent_name in zip(agent_ids, agent_names)
        }
        self.nb_of_rounds = int(nb_of_rounds)
        self.quota_messages_per_agent_per_round = int(
            quota_messages_per_agent_per_round
        )
        if item_types is not None:
            self.item_types = [item.lower() for item in item_types]
        else:
            self.item_types = ["coins"]
        self.state: NegotiationState | None = None
        self._starting_agent_index = self.rng.choice([0, 1])
        self.reset()

    def _other(self, agent_id: AgentId) -> AgentId:
        return get_coagent_id(self.agent_ids, agent_id)

    @abstractmethod
    def set_new_round_of_variant(self):
        """Variant hook: sample new private values / stock before each round."""
        pass

    @abstractmethod
    def get_info_of_variant(
        self, state: NegotiationState, actions: Dict[AgentId, Any]
    ) -> Dict[str, Any]:
        """Variant hook: populate SimulationStepLog.info with custom diagnostics."""
        pass

    def step(self, actions: Any) -> Tuple[bool, SimulationStepLog]:
        """
        Returns terminated, step_log
        """
        assert self.state is not None
        current_agent = self.state.current_agent
        a0, a1 = self.agent_ids[0], self.agent_ids[1]
        action = actions.get(current_agent)

        # Split phase: require both splits in the same timestep
        if self.state.split_phase:
            action_a0 = actions.get(a0)
            action_a1 = actions.get(a1)
            have_both_splits = isinstance(action_a0, Split) and isinstance(
                action_a1, Split
            )
            if not have_both_splits:
                rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
                return False, SimulationStepLog(
                    rewards=rewards, info={"type": "waiting_for_splits"}
                )

            # Record splits
            self.state.splits[a0] = action_a0
            self.state.splits[a1] = action_a1

            # Compute rewards and end round
            rewards = self.get_rewards(self.state.splits)

            # Info
            info = self.get_info_of_variant(self.state, actions)

            # Prepare next round
            # Alternate starting agent
            self.state.round_nb += 1
            self._starting_agent_index = 1 - self._starting_agent_index
            self.state.current_agent = self.agent_ids[self._starting_agent_index]
            self.state.previous_values = copy.deepcopy(self.state.values)
            self.state.previous_splits = copy.deepcopy(self.state.splits)
            self.state.previous_quantities = copy.deepcopy(self.state.quantities)
            self.state.previous_points = copy.deepcopy(rewards)
            self.state.last_message = ""
            self.set_new_round_of_variant()  # variant specific
            self.state.splits = {agent_id: None for agent_id in self.agent_ids}
            self.state.nb_messages_sent = {agent_id: 0 for agent_id in self.agent_ids}
            is_last_timestep_in_round = True
            done = self.state.round_nb >= self.nb_of_rounds

        # Message phase: roll the conversation forward a single turn.
        elif isinstance(action, Message):
            self.state.last_message = action.message
            self.state.nb_messages_sent[current_agent] += 1

            # Move turn to other agent
            self.state.current_agent = self._other(current_agent)

            # If both agents have reached their message quota, enter split phase
            if all(
                self.state.nb_messages_sent[agent_id]
                >= self.quota_messages_per_agent_per_round
                for agent_id in self.agent_ids
            ):
                self.state.split_phase = True
            is_last_timestep_in_round = False
            done = False
            rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            info = {"type": "message"}

        info[
            "is_last_timestep_in_round"
        ] = is_last_timestep_in_round  # Used later to group round timesteps if needed
        return done, SimulationStepLog(rewards=rewards, info=info)

    def get_obs(self):
        """Returns all agent observations in dict"""
        return {agent_id: self.get_obs_agent(agent_id) for agent_id in self.agent_ids}

    @abstractmethod
    def get_rewards(self, splits: Dict[AgentId, Split]) -> Dict[AgentId, float]:
        pass

    @abstractmethod
    def get_obs_agent(self, agent_id):
        pass

    def get_state(self):
        return self.state

    def get_safe_copy(self):
        """Return a safe copy of the simulation."""
        simulation_copy = copy.copy(self)
        simulation_copy.state = copy.deepcopy(self.state)
        return simulation_copy

    @abstractmethod
    def reset(self) -> dict[AgentId, NegotiationObs]:
        pass
