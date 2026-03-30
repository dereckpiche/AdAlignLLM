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
    runner_kwargs = dict(runner_kwargs)
    max_parallel_games = runner_kwargs.pop("max_parallel_games", None)

    async def run_game(markov_game: MarkovGame) -> RolloutTreeRootNode:
        return await runner(
            markov_game=markov_game,
            output_folder=output_folder,
            **runner_kwargs,
        )

    if max_parallel_games is not None:
        semaphore = asyncio.Semaphore(max(1, int(max_parallel_games)))

        async def run_game(markov_game: MarkovGame) -> RolloutTreeRootNode:
            async with semaphore:
                return await runner(
                    markov_game=markov_game,
                    output_folder=output_folder,
                    **runner_kwargs,
                )

    tasks = []
    for mg in markov_games:
        tasks.append(asyncio.create_task(run_game(mg)))
    return await asyncio.gather(*tasks)
