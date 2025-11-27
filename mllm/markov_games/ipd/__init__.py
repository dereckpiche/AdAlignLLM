"""
File: mllm/markov_games/ipd/__init__.py
Summary: Marks the Iterated Prisoner's Dilemma subpackage.
"""

from .Ipd_hard_coded_agents import AlwaysCooperateIPDAgent, AlwaysDefectIPDAgent

__all__ = [
    "AlwaysCooperateIPDAgent",
    "AlwaysDefectIPDAgent",
]
