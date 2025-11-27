"""
File: mllm/utils/rollout_tree_stats.py
Summary: Computes descriptive statistics from rollout tree collections.
"""

from typing import Any, Callable, List, Tuple

from mllm.markov_games.rollout_tree import RolloutTreeRootNode
from mllm.markov_games.simulation import SimulationStepLog
from mllm.utils.rollout_tree_gather_utils import (
    gather_simulation_step_logs,
    get_rollout_tree_paths,
)
from mllm.utils.stat_pack import StatPack


def get_rollout_tree_stat_tally(
    rollout_tree: RolloutTreeRootNode,
    metrics: List[Callable[[SimulationStepLog], List[Tuple[str, float]]]],
) -> StatPack:
    stat_tally = StatPack()
    # get simulation step logs
    node_list = get_rollout_tree_paths(rollout_tree)[0]
    simulation_step_logs = gather_simulation_step_logs(node_list)
    for simulation_step_log in simulation_step_logs:
        for metric in metrics:
            metric_result = metric(simulation_step_log)
            if metric_result is not None:
                for key, value in metric_result:
                    stat_tally.add_stat(key, value)
    return stat_tally


def get_rollout_tree_mean_stats(
    rollout_tree: RolloutTreeRootNode, metrics: List[Callable[[SimulationStepLog], Any]]
) -> StatPack:
    """Get the mean stats for a rollout tree."""
    stat_tally = get_rollout_tree_stat_tally(rollout_tree, metrics)
    return stat_tally.mean()


def get_mean_rollout_tree_stats(
    rollout_trees: List[RolloutTreeRootNode],
    metrics: List[Callable[[SimulationStepLog], Any]],
) -> StatPack:
    """Get the mean stats for a list of rollout trees."""
    # Compute per-rollout means first, then aggregate them across the entire batch.
    stat_tallies = [
        get_rollout_tree_mean_stats(rollout_tree, metrics)
        for rollout_tree in rollout_trees
    ]
    mean_stat_tally = StatPack()
    for stat_tally in stat_tallies:
        mean_stat_tally.add_stats(stat_tally)
    return mean_stat_tally.mean()
