"""
File: mllm/markov_games/rollout_tree.py
Summary: Defines rollout tree data structures and serialization helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import jsonschema
from pydantic import BaseModel, Field, model_validator

from mllm.chat_utils.chat_turn import ChatTurn

AgentId = str


class SimulationStepLog(BaseModel):
    """Minimal snapshot of environment-side rewards and auxiliary info."""

    rewards: dict[AgentId, float]
    info: Any = None


class AgentActLog(BaseModel):
    """LLM-side provenance for an action (chat turns + metadata)."""

    chat_turns: list[ChatTurn] | None
    info: Any = None

    @model_validator(mode="after")
    def _exactly_one_state_end(self):
        """
        This method is used to enforce that for each AgentActLog, there is exactly one ChatTurn which is a state end.
        """
        if self.chat_turns != []:
            n = sum(1 for t in self.chat_turns if t.is_state_end)
            if n != 1:
                raise ValueError(
                    f"AgentActLog must have exactly one ChatTurn with is_state_end=True; got {self.chat_turns}."
                )
            return self
        else:
            return self


class StepLog(BaseModel):
    action_logs: dict[AgentId, AgentActLog]
    simulation_step_log: SimulationStepLog


# BranchType = Literal["unilateral_deviation", "common_deviation"] # might not be necessary
# class BranchNodeInfo(BaseModel):
#     branch_id: str
#     branch_for: AgentId
#     branch_type: BranchType


class RolloutTreeNode(BaseModel):
    """Single timestep of the main trajectory (or a branch) plus linkage."""

    step_log: StepLog
    time_step: int
    child: RolloutTreeNode | RolloutTreeBranchNode | None = None


class RolloutTreeBranchNode(BaseModel):
    """
    First item of the tuple indicates which agent "called" for an alternative branch.
    """

    main_child: RolloutTreeNode
    branches: dict[AgentId, list[RolloutTreeNode]] | None = None


class RolloutTreeRootNode(BaseModel):
    """Entry point for serialized rollouts (main path plus optional branches)."""

    id: int
    crn_id: int  # ID of the rng used to generate this rollout tree
    child: RolloutTreeNode | RolloutTreeBranchNode | None = None
    agent_ids: List[AgentId] = Field(min_length=1)


# class RolloutTreeLeafNode(BaseModel):
#     step_log: StepLog
#     time_step: int


# Necessary for self-referential stuff in pydantic
RolloutTreeBranchNode.model_rebuild()
RolloutTreeNode.model_rebuild()
