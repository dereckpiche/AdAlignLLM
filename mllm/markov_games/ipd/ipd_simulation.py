"""
File: mllm/markov_games/ipd/ipd_simulation.py
Summary: Runs Iterated Prisoner's Dilemma simulations under the Markov-game API.
"""

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mllm.markov_games.markov_game import Simulation
from mllm.markov_games.rollout_tree import SimulationStepLog
from mllm.utils.get_coagent_id import get_coagent_id


@dataclass
class IPDState:
    """
    State of the Iterated Prisoner's Dilemma game.
    """

    round_nb: int = 0
    done: bool = False
    last_moves: Dict[str, str] | None = None


@dataclass
class IPDObs:
    """
    Observation in Iterated Prisoner's Dilemma game.
    """

    round_nb: int
    last_coagent_move: str | None


class IPD(Simulation):
    """
    Iterated Prisoner's Dilemma simulation following the standard.

    In each round of the game, two agents simultaneously choose to either cooperate (C) or defect (D).
    The payoffs are as follows:
    - If both cooperate: Both receive the "reward" (usually 3 points)
    - If both defect: Both receive the "punishment" (usually 1 point)
    - If one cooperates and one defects: The defector receives the "temptation" (usually 5 points)
      and the cooperator receives the "sucker" payoff (usually 0 points)

    The game is played for a specified number of rounds.
    """

    def __init__(
        self,
        agent_ids: List[str],
        agent_names: List[str],
        seed: int,
        rounds_per_game: int,
        reward: float,  # Both cooperate
        punishment: float,  # Both defect
        temptation: float,  # Defector's reward when other cooperates
        sucker: float,  # Cooperator's reward when other defects
        cooperate_actions: List[str],
        defect_actions: List[str],
    ):
        self.agent_ids = agent_ids
        self.agent_names = agent_names
        self.seed = seed
        self.rounds_per_game = rounds_per_game
        self.reward = reward
        self.punishment = punishment
        self.temptation = temptation
        self.sucker = sucker
        self.cooperate_actions = cooperate_actions
        self.defect_actions = defect_actions
        self.state = IPDState()

    def step(self, actions: Dict[str, str]) -> Tuple[bool, SimulationStepLog]:
        """
        Take a step in the environment using the provided actions.
        Here, the observations are just the states of the game.

        Args:
            actions (dict): A dictionary where keys are agent identifiers and values are actions ('C' or 'D').

        Returns:
            observations (dict): A dictionary where keys are agent identifiers and values are observations.
            done (bool): Whether the episode has ended.
            info (dict): Additional information about the environment.
        """

        # Calculate rewards using payoff matrix
        agent0_action = actions[self.agent_ids[0]]
        agent1_action = actions[self.agent_ids[1]]

        # Normalize actions to standard cooperate/defect/gibberish format
        def normalize_action(action):
            if action in self.cooperate_actions:
                return "C"
            elif action in self.defect_actions:
                return "D"
            else:
                return "D"

        norm_action0 = normalize_action(agent0_action)
        norm_action1 = normalize_action(agent1_action)

        payoffs = {
            ("C", "C"): [self.reward, self.reward],
            ("C", "D"): [self.sucker, self.temptation],
            ("D", "C"): [self.temptation, self.sucker],
            ("D", "D"): [self.punishment, self.punishment],
        }

        round_rewards = {
            self.agent_ids[0]: payoffs[(norm_action0, norm_action1)][0],
            self.agent_ids[1]: payoffs[(norm_action0, norm_action1)][1],
        }

        # Update game state
        self.state.round_nb += 1
        self.state.last_moves = copy.deepcopy(actions)
        done = self.state.round_nb >= self.rounds_per_game
        step_log = SimulationStepLog(
            rewards=round_rewards,
            info={
                "actions": {
                    self.agent_ids[0]: norm_action0,
                    self.agent_ids[1]: norm_action1,
                }
            },
        )

        return done, step_log

    def get_obs(self):
        """Returns all agent observations in dict
        Returns:
            observations
        """
        observations = {}
        for agent_id in self.agent_ids:
            observations[agent_id] = self.get_obs_agent(agent_id)
        return observations

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        if self.state.last_moves != None:
            other_id = get_coagent_id(self.agent_ids, agent_id)
            last_coagent_move = self.state.last_moves[other_id]
        else:
            last_coagent_move = None
        obs = IPDObs(round_nb=self.state.round_nb, last_coagent_move=last_coagent_move)
        return obs

    def reset(self):
        """Returns initial observations and states"""
        self.state = IPDState()
        return self.get_obs()

    def get_safe_copy(self):
        """
        Return a safe copy of the simulation.
        """
        simulation_copy = copy.copy(self)
        simulation_copy.state = copy.deepcopy(self.state)
        return simulation_copy
