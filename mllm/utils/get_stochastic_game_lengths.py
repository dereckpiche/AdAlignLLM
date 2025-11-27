"""
File: mllm/utils/get_stochastic_game_lengths.py
Summary: Computes distributions over stochastic game lengths.
"""

import numpy as np


def get_stochastic_game_lengths(
    max_length, nb_games, continuation_prob, same_length_batch=False
):
    """
    Generates stochastic game lengths based on a geometric distribution.

    Args:
        max_length (int): The maximum length a game can have.
        nb_games (int): The number of games to generate lengths for.
        continuation_prob (float): The probability of the game continuing after each round.
        same_length_batch (bool): If True, all games will have the same length.

    Returns:
        Array: An array of game lengths.
    """
    if continuation_prob == 1:
        return [max_length] * nb_games
    if same_length_batch:
        length = np.random.geometric(1 - continuation_prob, 1)
        game_lengths = np.repeat(length, nb_games)
    else:
        game_lengths = np.random.geometric(1 - continuation_prob, nb_games)

    game_lengths = np.where(game_lengths > max_length, max_length, game_lengths)
    return game_lengths.tolist()
