"""
File: mllm/markov_games/group_timesteps.py
Summary: Provides timestep-grouping utilities for rollout trees and training.
"""

import copy
from typing import Callable

from mllm.markov_games.markov_game import MarkovGame
from mllm.markov_games.rollout_tree import (
    AgentActLog,
    RolloutTreeBranchNode,
    RolloutTreeNode,
    RolloutTreeRootNode,
    StepLog,
)
from mllm.markov_games.simulation import SimulationStepLog

AgentId = str


def group_time_steps(
    rollout_tree: RolloutTreeRootNode,
    accumulation_stop_condition: Callable[[StepLog], bool],
) -> RolloutTreeRootNode:
    """
    During generation, we create rollout trees according to the real time steps.
    However, during training, we might want to treat groups of time steps as a single time step.
    As a concrete example, take Trust-and-Split. At each round, say we have X time steps of communication and then one time step for the split.
    Then the communication actions will not get any reward, and the split action will get the reward. During REINFORCE training, with discounting, this
    can cause training instability. We could instead treat every action in the round as being part of a single action, and give it the reward of the split action.
    This method helps to do this sort of grouping.
    It accumulates actions until the accumulation_stop_condition is met, and then creates a new node with the accumulated actions.
    It then recursively calls itself on the child node.
    Details:
    - The reward for the group is the reward of the last time step in the group.
    - The simulation log for the group is the simulation log of the last time step in the group.
    - The state end for the group becomes the first state end in the group.
    - The agent info for the group is the agent info of the last time step in the group.
    """

    def group_step_logs(step_logs: list[StepLog]) -> StepLog:
        """
        Concatenate per-agent chat turns across steps; keep only the first is_state_end.
        """
        last_sim_log = step_logs[-1].simulation_step_log
        agent_ids = {aid for s in step_logs for aid in s.action_logs.keys()}
        grouped_logs: dict[AgentId, AgentActLog] = {}
        for aid in agent_ids:
            turns = []
            for s in step_logs:
                act = s.action_logs.get(aid)
                if act and act.chat_turns:
                    turns.extend(copy.deepcopy(act.chat_turns))
            disable_is_state_end = False
            # Only the first state_end should be True, the rest should be False
            for t in turns:
                if t.is_state_end:
                    if disable_is_state_end:
                        t.is_state_end = False
                    else:
                        disable_is_state_end = True
                    continue
            grouped_logs[aid] = AgentActLog(
                chat_turns=turns, info=step_logs[-1].action_logs[aid].info
            )
        return StepLog(action_logs=grouped_logs, simulation_step_log=last_sim_log)

    def group_time_steps_rec(
        current_node: RolloutTreeNode | RolloutTreeBranchNode,
        group_time_step: int,
        accumulation_step_logs: list[StepLog],
    ) -> RolloutTreeNode | RolloutTreeBranchNode:
        """
        Groups time steps. Recursion is used to handle branches.
        """
        assert isinstance(current_node, RolloutTreeNode) or isinstance(
            current_node, RolloutTreeBranchNode
        ), "Current node must be a tree node or a branch node. Is of type: " + str(
            type(current_node)
        )
        first_group_node = None
        current_group_node = None
        while current_node is not None:
            if isinstance(current_node, RolloutTreeBranchNode):
                raise Exception(
                    "Grouping timesteps by round is not supported for branching trajectories yet."
                )

            # Accumulate
            accumulation_step_logs.append(current_node.step_log)
            if accumulation_stop_condition(current_node.step_log):
                grouped_step_logs = group_step_logs(accumulation_step_logs)
                accumulation_step_logs = []
                new_group_node = RolloutTreeNode(
                    step_log=grouped_step_logs, time_step=group_time_step, child=None
                )
                if first_group_node == None:
                    first_group_node = new_group_node
                group_time_step += 1
                if current_group_node is not None:
                    current_group_node.child = new_group_node
                current_group_node = new_group_node
            current_node = current_node.child
        return first_group_node

    node = group_time_steps_rec(
        current_node=rollout_tree.child, group_time_step=0, accumulation_step_logs=[]
    )
    return RolloutTreeRootNode(
        id=rollout_tree.id,
        crn_id=rollout_tree.crn_id,
        child=node,
        agent_ids=rollout_tree.agent_ids,
    )


def stop_when_round_ends(step_log: StepLog) -> bool:
    """
    Simplest stop condition. Will return True if step log is the last time step of a round.
    This will throw an error if this information is not available in the simulation info.
    """
    assert (
        "is_last_timestep_in_round" in step_log.simulation_step_log.info.keys()
    ), "To group by round, is_last_timestep_in_round must be set in the info of your simulation step log at each time step."
    return step_log.simulation_step_log.info["is_last_timestep_in_round"]


def group_by_round(rollout_tree: RolloutTreeRootNode) -> RolloutTreeRootNode:
    """
    Groups time steps by round.
    """
    return group_time_steps(rollout_tree, stop_when_round_ends)
