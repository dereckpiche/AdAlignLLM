"""
File: mllm/markov_games/run_markov_games.py
Summary: CLI entry point for running configured Markov-game experiments.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from torch._C import ClassType

from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import RolloutTreeRootNode


async def run_markov_games(
    runner: Callable[[MarkovGame], RolloutTreeRootNode],
    runner_kwargs: dict,
    output_folder: str,
    markov_games: list[MarkovGame],
) -> list[RolloutTreeRootNode]:
    """
    Kick off multiple Markov game rollouts concurrently and return their trees.

    Parameters mirror the Hydra configs (runner callable + kwargs) so callers can
    choose ``LinearRunner``, ``AlternativeActionsRunner`` or future variants.
    """
    tasks = []
    for mg in markov_games:
        tasks.append(
            asyncio.create_task(
                runner(markov_game=mg, output_folder=output_folder, **runner_kwargs)
            )
        )
    return await asyncio.gather(*tasks)
