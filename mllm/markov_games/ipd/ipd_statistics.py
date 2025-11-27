"""
File: mllm/markov_games/ipd/ipd_statistics.py
Summary: Computes statistics and summaries for IPD experiments.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from mllm.markov_games.rollout_tree import SimulationStepLog


def avg_reward(sl: SimulationStepLog) -> List[Tuple[str, float]]:
    for aid in sl.rewards.keys():
        if "buffer" in str(aid) and "live" not in str(aid):
            return None
    # One value per agent at each step
    rewards_dict = {f"reward-{aid}": float(v) for aid, v in (sl.rewards or {}).items()}
    return [(key, value) for key, value in rewards_dict.items() if value is not None]


stat_functs: list[Callable[[SimulationStepLog], List[Tuple[str, float]]]] = [
    avg_reward,
]
