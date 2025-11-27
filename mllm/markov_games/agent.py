"""
File: mllm/markov_games/agent.py
Summary: Declares the base Agent interface connecting simulations to policy calls.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Tuple

from numpy.random import default_rng

from mllm.markov_games.rollout_tree import AgentActLog


class Agent(ABC):
    """Abstract policy wrapper that bridges simulations with arbitrary backends."""

    @abstractmethod
    def __init__(
        self,
        seed: int,
        agent_id: str,
        agent_name: str,
        agent_policy: Callable[[list[dict]], str],
        *args,
        **kwargs,
    ):
        """
        Initialize the agent state and seed its RNG.

        Subclasses typically store extra handles (tokenizers, inference clients, etc.)
        but they should always call ``super().__init__`` so sampling remains reproducible.
        """
        self.seed = seed
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.policy = policy
        self.rng = default_rng(self.seed)
        raise NotImplementedError

    async def act(self, observation) -> Tuple[Any, AgentActLog]:
        """
        Produce the next action (and associated chat log) given an environment observation.

        Implementations can iterate with rejection sampling, multi-call deliberation, etc.
        Returns both the chosen action and an `AgentActLog` describing how it was produced.
        """
        raise NotImplementedError

    def get_safe_copy(self):
        """
        Return a deep copy whose future calls do not mutate the original agent.

        Needed for branch exploration/reruns with alternative actions.
        """
        raise NotImplementedError

    def reset(self):
        """Reset any internal state between rollouts."""
        raise NotImplementedError

    def render(self):
        """Optional human-readable visualization of the agent (CLI/UI)."""
        raise NotImplementedError

    def close(self):
        """Release any external resources (network sockets, subprocesses, etc.)."""
        raise NotImplementedError

    def get_agent_info(self):
        """Return diagnostic metadata to embed inside rollout logs."""
        raise NotImplementedError
