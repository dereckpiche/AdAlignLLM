"""
File: mllm/markov_games/markov_game.py
Summary: Defines the MarkovGame base class plus shared simulation interfaces.
"""

import asyncio
import copy
import json
import os
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

from transformers.models.idefics2 import Idefics2Config

from mllm.markov_games.agent import Agent
from mllm.markov_games.rollout_tree import AgentActLog, StepLog
from mllm.markov_games.simulation import Simulation

AgentId = str


@dataclass
class AgentAndActionSafeCopy:
    """Snapshot of an agent, its action, and metadata used for branch replay."""

    action: Any
    action_info: AgentActLog
    agent_after_action: type[Agent]


class MarkovGame(object):
    def __init__(
        self,
        id: int,
        agents: dict[AgentId, type[Agent]],
        simulation: type[Simulation],
        crn_id: int,
    ):
        """
        Initialize the Markov game wrapper.

        Parameters
        ----------
        id:
            Unique rollout identifier (logged into rollout trees).
        agents:
            Mapping of agent_id -> Agent instance.
        simulation:
            Environment implementing the ``Simulation`` interface (IPD, TAS, etc.).
        crn_id:
            Identifier for the common random number stream used by this rollout.
        """
        self.agents = agents
        self.agent_ids = self.agents.keys()
        self.simulation = simulation
        self.simulation_step_log = None
        self.agent_step_logs = {agent_id: None for agent_id in self.agent_ids}
        self.actions = {}
        self.id = id
        self.crn_id = crn_id

    def get_id(self) -> str:
        return self.id

    def get_crn_id(self) -> int:
        return self.crn_id

    def get_agent_ids(self) -> List[AgentId]:
        return list(self.agent_ids)

    async def get_action_of_agent_without_side_effects(
        self, agent_id: AgentId
    ) -> Tuple[Any, AgentActLog]:
        """
        Safe function to get an action of an agent without modifying the agent or the simulation.
        """
        agent = self.agents[agent_id]
        agent_before_action = agent.get_safe_copy()
        obs = self.simulation.get_obs_agent(agent_id)
        action, action_info = await agent.act(observation=obs)
        self.agents[agent_id] = agent_before_action
        agent_after_action = agent.get_safe_copy()
        return AgentAndActionSafeCopy(action, action_info, agent_after_action)

    async def get_actions_of_agents_without_side_effects(
        self,
    ) -> dict[AgentId, AgentAndActionSafeCopy]:
        """
        Safe function to get an action of an agent without modifying the agent or the simulation.
        """
        tasks = []
        for agent_id in self.agent_ids:
            task = asyncio.create_task(
                self.get_action_of_agent_without_side_effects(agent_id)
            )
            tasks.append(task)
        agent_and_action_safe_copies: list[
            AgentAndActionSafeCopy
        ] = await asyncio.gather(*tasks)
        return {
            agent_id: agent_and_action_safe_copy
            for agent_id, agent_and_action_safe_copy in zip(
                self.agent_ids, agent_and_action_safe_copies
            )
        }

    def set_action_and_agent_after_action_manually(
        self,
        agent_id: AgentId,
        agent_action_safe_copy: AgentAndActionSafeCopy,
    ):
        """
        Set the action and the agent after action manually.
        """
        self.actions[agent_id] = agent_action_safe_copy.action
        self.agent_step_logs[agent_id] = agent_action_safe_copy.action_info
        self.agents[agent_id] = agent_action_safe_copy.agent_after_action

    def set_actions_of_agents_manually(
        self, actions: dict[AgentId, AgentAndActionSafeCopy]
    ):
        """
        Set the actions of agents manually.
        """
        for agent_id, agent_action_safe_copy in actions.items():
            self.set_action_and_agent_after_action_manually(
                agent_id, agent_action_safe_copy
            )

    async def set_action_of_agent(self, agent_id: AgentId):
        """
        Query a single agent for its next action and store the result locally.
        """
        agent = self.agents[agent_id]
        obs = self.simulation.get_obs_agent(agent_id)
        action, action_info = await agent.act(observation=obs)
        self.actions[agent_id] = action
        self.agent_step_logs[agent_id] = action_info

    async def set_actions(self):
        """
        Query every agent concurrently and populate the cached actions/logs.
        """
        # background_tasks = set()
        tasks = []
        for agent_id in self.agent_ids:
            task = asyncio.create_task(self.set_action_of_agent(agent_id))
            tasks.append(task)
        await asyncio.gather(*tasks)

    def take_simulation_step(self):
        """
        Advance the simulation by one step using the cached actions.
        """
        terminated, self.simulation_step_log = self.simulation.step(self.actions)
        return terminated

    def get_step_log(self) -> StepLog:
        """
        Package the most recent simulation step and agent logs into a StepLog.
        """
        if self.simulation_step_log is None:
            raise RuntimeError(
                "Simulation step log is empty; call take_simulation_step() first."
            )
        missing_logs = [
            agent_id for agent_id, log in self.agent_step_logs.items() if log is None
        ]
        if missing_logs:
            raise RuntimeError(
                f"Agent action logs missing for: {', '.join(missing_logs)}. "
                "Ensure set_actions() ran before requesting the step log."
            )
        step_log = StepLog(
            simulation_step_log=self.simulation_step_log,
            action_logs=self.agent_step_logs,
        )
        return step_log

    async def step(self) -> Tuple[bool, StepLog]:
        """
        Convenience step that collects actions, advances the simulation, and returns the log.
        """
        await self.set_actions()
        terminated = self.take_simulation_step()
        step_log = self.get_step_log()
        return terminated, step_log

    def get_safe_copy(self):
        """
        Create a shallow copy of the game with deep-copied agents/simulation for branching.
        """

        new_markov_game = copy.copy(self)
        new_simulation = self.simulation.get_safe_copy()
        new_agents = {
            agent_id: agent.get_safe_copy() for agent_id, agent in self.agents.items()
        }

        # Reassign copied components
        new_markov_game.simulation = new_simulation
        new_markov_game.agents = new_agents

        # IMPORTANT: ensure agent_ids references the new agents dict, not the original
        new_markov_game.agent_ids = new_markov_game.agents.keys()

        # Deep-copy step data to avoid correlation
        new_markov_game.simulation_step_log = copy.deepcopy(self.simulation_step_log)
        new_markov_game.actions = copy.deepcopy(self.actions)
        # Rebuild logs to align exactly with new agent ids
        old_agent_step_logs = copy.deepcopy(self.agent_step_logs)
        new_markov_game.agent_step_logs = {
            agent_id: old_agent_step_logs.get(agent_id)
            for agent_id in new_markov_game.agent_ids
        }

        return new_markov_game
