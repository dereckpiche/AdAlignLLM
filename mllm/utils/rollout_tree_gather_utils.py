"""
File: mllm/utils/rollout_tree_gather_utils.py
Summary: Utilities for gathering rollout tree files and metadata.
"""

from __future__ import annotations

import csv
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from mllm.markov_games.rollout_tree import *


def load_rollout_tree(path: Path) -> RolloutTreeRootNode:
    """Load a rollout tree from a PKL file containing a dict."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return RolloutTreeRootNode.model_validate(data)


@dataclass
class RolloutNodeList:
    id: str
    nodes: List[RolloutTreeNode]


def get_rollout_tree_paths(
    root: RolloutTreeRootNode, mgid: Optional[str] = None
) -> Tuple[RolloutNodeList, List[RolloutNodeList]]:
    """
    Returns:
        main_path: The main path from the root to the end of the tree.
        branch_paths: A list of all branch paths from the root to the end of the tree.
        Each branch path contains a list of nodes that are part of the branch, including the nodes from the main path before the branch was taken.
    """
    branch_paths = []

    def collect_path_nodes(current) -> List[RolloutTreeNode]:
        """Recursively collect all nodes in a path starting from current node."""
        if current is None:
            return []

        if isinstance(current, RolloutTreeNode):
            return [current] + collect_path_nodes(current.child)

        elif isinstance(current, RolloutTreeBranchNode):
            # For branch nodes, we only follow the main_child for path collection
            if current.main_child:
                return [current.main_child] + collect_path_nodes(
                    current.main_child.child
                )
            else:
                return []

    def traverse_for_branches(
        current,
        main_path_prefix: List[RolloutTreeNode],
        path_id: str,
        current_time_step: Optional[int] = 0,
    ):
        """Traverse tree to collect all branch paths."""
        if current is None:
            return

        if isinstance(current, RolloutTreeNode):
            # Continue traversing with this node added to the main path prefix
            new_prefix = main_path_prefix + [current]
            traverse_for_branches(current.child, new_prefix, path_id, current.time_step)

        elif isinstance(current, RolloutTreeBranchNode):
            # Collect all branch paths
            if current.branches:
                for agent_id, branch_node_list in current.branches.items():
                    if branch_node_list:
                        # Start with the main path prefix, then recursively collect all nodes in this branch
                        branch_path_nodes = main_path_prefix.copy()
                        for branch_node in branch_node_list:
                            branch_path_nodes.extend(collect_path_nodes(branch_node))

                        # Create proper branch path ID with mgid, agent_id, and time_step
                        mgid_str = mgid or str(root.id)
                        branch_path_id = f"mgid:{mgid_str}_type:branch_agent:{agent_id}_time_step:{current_time_step}"
                        branch_paths.append(
                            RolloutNodeList(id=branch_path_id, nodes=branch_path_nodes)
                        )

            # Process the main child and add to prefix
            new_prefix = main_path_prefix
            if current.main_child:
                new_prefix = main_path_prefix + [current.main_child]

            # Continue traversing the main path
            if current.main_child:
                traverse_for_branches(
                    current.main_child.child,
                    new_prefix,
                    path_id,
                    current.main_child.time_step,
                )

    # Collect the main path nodes
    main_path_nodes = collect_path_nodes(root.child)

    # Traverse to collect all branch paths
    traverse_for_branches(root.child, [], "")

    # Create the main path with proper mgid format
    mgid_str = mgid or str(root.id)
    main_path = RolloutNodeList(id=f"mgid:{mgid_str}_type:main", nodes=main_path_nodes)

    return main_path, branch_paths


class ChatTurnLog(BaseModel):
    time_step: int
    agent_id: str
    role: str
    content: str
    reasoning_content: Optional[str] = None
    is_state_end: bool
    reward: float


def gather_agent_chat_turns_for_path(
    agent_id: str, path: RolloutNodeList
) -> List[ChatTurnLog]:
    """Iterate through all chat turns for a specific agent in a path sorted by time step."""
    turns = []
    for node in path.nodes:
        action_log = node.step_log.action_logs.get(agent_id, [])
        if action_log:
            for chat_turn in action_log.chat_turns or []:
                turns.append(
                    ChatTurnLog(
                        time_step=node.time_step,
                        agent_id=agent_id,
                        role=chat_turn.role,
                        content=chat_turn.content,
                        reasoning_content=getattr(chat_turn, "reasoning_content", None),
                        is_state_end=chat_turn.is_state_end,
                        reward=node.step_log.simulation_step_log.rewards.get(
                            agent_id, 0
                        ),
                    )
                )
    return turns


def gather_all_chat_turns_for_path(path: RolloutNodeList) -> List[ChatTurnLog]:
    """Iterate through all chat turns for all agents in a path sorted by time step."""
    turns = []

    # Collect turns from all agents, but interleave them per timestep by (user, assistant) pairs
    for node in path.nodes:
        # Build (user[, assistant]) pairs for each agent at this timestep
        agent_ids = sorted(list(node.step_log.action_logs.keys()))
        per_agent_pairs: Dict[str, List[List[ChatTurnLog]]] = {}

        for agent_id in agent_ids:
            action_log = node.step_log.action_logs.get(agent_id)
            pairs: List[List[ChatTurnLog]] = []
            current_pair: List[ChatTurnLog] = []

            if action_log and action_log.chat_turns:
                for chat_turn in action_log.chat_turns:
                    turn_log = ChatTurnLog(
                        time_step=node.time_step,
                        agent_id=agent_id,
                        role=chat_turn.role,
                        content=chat_turn.content,
                        reasoning_content=getattr(chat_turn, "reasoning_content", None),
                        is_state_end=chat_turn.is_state_end,
                        reward=node.step_log.simulation_step_log.rewards.get(
                            agent_id, 0
                        ),
                    )

                    if chat_turn.role == "user":
                        # If a previous pair is open, close it and start a new one
                        if current_pair:
                            pairs.append(current_pair)
                            current_pair = []
                        current_pair = [turn_log]
                    else:
                        # assistant: attach to an open user message if present; otherwise stand alone
                        if (
                            current_pair
                            and len(current_pair) == 1
                            and current_pair[0].role == "user"
                        ):
                            current_pair.append(turn_log)
                            pairs.append(current_pair)
                            current_pair = []
                        else:
                            # No preceding user or already paired; treat as its own unit
                            pairs.append([turn_log])

                if current_pair:
                    # Unpaired trailing user message
                    pairs.append(current_pair)

            per_agent_pairs[agent_id] = pairs

        # Interleave pairs across agents: A1, B1, A2, B2, ...
        index = 0
        while True:
            added_any = False
            for agent_id in agent_ids:
                agent_pairs = per_agent_pairs.get(agent_id, [])
                if index < len(agent_pairs):
                    for tl in agent_pairs[index]:
                        turns.append(tl)
                    added_any = True
            if not added_any:
                break
            index += 1

    return turns


def chat_turns_to_dict(chat_turns: Iterator[ChatTurnLog]) -> Iterator[Dict[str, Any]]:
    """Render all chat turns for a path as structured data for JSON."""
    for chat_turn in chat_turns:
        yield chat_turn.model_dump()


def get_all_agents(root: RolloutTreeRootNode) -> List[str]:
    """list of all agent IDs that appear in the tree."""
    if root.child is None:
        return []

    # Get the first node to extract all agent IDs
    first_node = root.child
    if isinstance(first_node, RolloutTreeBranchNode):
        first_node = first_node.main_child

    if first_node is None:
        return []

    # All agents should be present in the first node
    agents = set(first_node.step_log.action_logs.keys())
    agents.update(first_node.step_log.simulation_step_log.rewards.keys())

    return sorted(list(agents))


def gather_agent_main_rewards(agent_id: str, path: RolloutNodeList) -> List[float]:
    """Gather main rewards for a specific agent in a path."""
    rewards = []
    for node in path.nodes:
        reward = node.step_log.simulation_step_log.rewards[agent_id]
        rewards.append(reward)
    return rewards


def gather_all_rewards(path: RolloutNodeList) -> List[Dict[AgentId, float]]:
    """Gather main rewards from main trajectory in a path."""
    rewards = []
    for node in path.nodes:
        rewards.append(node.step_log.simulation_step_log.rewards.copy())
    return rewards


def gather_simulation_stats(
    path: RolloutNodeList,
    filter: Callable[[SimulationStepLog], bool],
    stat_func: Callable[[SimulationStepLog], Any],
) -> List[Any]:
    """Gather stats from main trajectory in a path."""
    stats = []
    for node in path.nodes:
        sl = node.step_log.simulation_step_log
        if filter(sl):
            stats.append(stat_func(sl))
    return stats


def gather_simulation_step_logs(path: RolloutNodeList) -> List[SimulationStepLog]:
    """Gather simulation information from main trajectory in a path."""
    infos = []
    for node in path.nodes:
        infos.append(node.step_log.simulation_step_log)
    return infos


def export_chat_logs(path: Path, outdir: Path):
    """Process a rollout tree PKL file and generate a JSONL of chat turns as dicts.
    Each line contains an object with path_id and chat_turns for a single path.
    """
    import json

    root = load_rollout_tree(path)
    mgid = root.id

    main_path, branch_paths = get_rollout_tree_paths(root)
    all_paths = [main_path] + branch_paths

    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / f"mgid:{mgid}_plucked_chats.render.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for path_obj in all_paths:
            chat_turns = gather_all_chat_turns_for_path(path_obj)
            output_obj = {
                "path_id": str(path_obj.id),
                "chat_turns": list(chat_turns_to_dict(iter(chat_turns))),
            }
            f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
