"""
File: mllm/training/annealing_methods.py
Summary: Implements annealing schedules used across training loops.
"""

import numpy as np


def sigmoid_annealing(step: int, temperature: float) -> float:
    """
    Smoothly ramp a scalar from 0 → 1 using a temperature-controlled sigmoid.

    Args:
        step: Current training step or iteration.
        temperature: Controls how sharp the transition is; larger values flatten the curve.

    Returns:
        Float in [-1, 1] that can be rescaled for annealing schedules.
    """
    return 2 / (1 + np.exp(-step / temperature)) - 1
