"""
File: mllm/markov_games/simulation.py
Summary: Core simulation loop utilities and step logging for Markov games.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import SimulationStepLog


class Simulation(ABC):
    @abstractmethod
    def __init__(self, seed: int, *args, **kwargs):
        self.seed = seed
        self.rng = default_rng(self.seed)

    @abstractmethod
    def step(self, actions: Any) -> Tuple[bool, SimulationStepLog]:
        """
        Advance the environment by one logical tick using ``actions``.

        Returns
        -------
        terminated: bool
            Whether the episode has finished.
        SimulationStepLog
            Reward/info bundle describing this transition.
        """
        raise NotImplementedError

    def get_obs(self):
        """Return a dict mapping agent_id -> observation for *all* agents."""
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """Return the observation for a single agent."""
        raise NotImplementedError

    def get_obs_size(self):
        """Describe the observation tensor shape (useful for critic heads)."""
        raise NotImplementedError

    def get_state(self):
        """Return the privileged simulator state if available."""
        raise NotImplementedError

    def get_state_size(self):
        """Describe the state tensor shape."""
        raise NotImplementedError

    def get_avail_actions(self):
        """Return the global action mask/tensor if the space is discrete."""
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """Return the available action mask for a given agent."""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take.

        Implementations currently assume a discrete, one-dimensional action space per agent.
        """
        raise NotImplementedError

    def get_safe_copy(self):
        """
        Return copy of the simulator that shares no mutable state with the original.
        """
        raise NotImplementedError

    def reset(self):
        """Reset to the initial state and return the starting observations."""
        raise NotImplementedError

    def render(self):
        """Optional human-facing visualization."""
        raise NotImplementedError

    def close(self):
        """Release any owned resources (files, processes, etc.)."""
        raise NotImplementedError

    # def seed(self):
    #     raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_simulation_info(self):
        raise NotImplementedError
