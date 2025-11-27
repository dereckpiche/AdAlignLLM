"""
File: mllm/markov_games/negotiation/no_press_nego_agent.py
Summary: Agent variant for no-press negotiations without explicit messaging.
"""

from typing import Any, Dict, List, Tuple

from mllm.markov_games.negotiation.nego_agent import (
    NegotiationAgent,
    NegotiationAgentState,
)
from mllm.markov_games.negotiation.nego_simulation import Split
from mllm.markov_games.negotiation.no_press_nego_simulation import NoPressObs
from mllm.markov_games.rollout_tree import AgentActLog, ChatTurn


class NoPressAgent(NegotiationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No communication in this variant
        self.intro_prompt = (
            "Welcome to an iterated game. You are {agent}. The other agent is {other_agent}.\n"
            "Setup:\n"
            "1. The game consists of multiple independent rounds.\n"
            "2. In each round, there are multiple items to split between the two agents.\n"
            "3. Both agents are assigned a per-item value between 1 and 20 (inclusive) in each round.\n"
            "4. You can observe per-item values of both agents.\n"
            "5. Because assignments are random, both agents are equally likely to have same expected per-item value.\n"
            "\n"
            "Protocol:\n"
            "1. Both agents simultaneously propose the amount of each item they will keep.\n"
            "2. If the total sum of proposals is less than or equal to the item quantity, both agents receive their proposed amounts.\n"
            "3. If the total sum of proposals exceeds the item quantity, they are allocated proportionally.\n"
            "4. Your points for the round = (amount you receive per item) x (your per-item value for that round), added across all items.\n"
            "5. Points are accumulated across rounds.\n"
            "Your goal: {goal}\n"
        )
        self.new_round_prompt = (
            "A New Round Begins\n"
            "The items to split are {quantities}.\n"
            "Your per-item values are {value} and {other_agent}'s per-item values are  {other_value}."
        )
        self.last_round_prompt = (
            "Last Round Summary:\n"
            "   - Items to split: {last_quantities}\n"
            "   - Your per-item values: {last_value_agent}\n"
            "   - {other_agent}'s per-item values: {last_value_coagent}\n"
            "   - You proposed: {last_split_agent}\n"
            "   - You earned: {last_points_agent} points\n"
            "   - {other_agent} proposed: {last_split_coagent}\n"
            "   - {other_agent} earned: {last_points_coagent} points\n"
            "   - Round Complete.\n"
        )
        self.send_split_prompt = "Submit Your Proposal\n" "Respond as {proposal_style}"

    def get_message_regex(self, observation: NoPressObs) -> str:
        """Return an empty pattern because the no-press variant forbids chat."""
        return r"^$"  # No messages allowed

    def get_split_regex(self, observation: NoPressObs) -> str:
        """Match proposals like ``Proposal: 4 coins, 6 apples`` case-insensitively."""
        items = list(observation.quantities.keys())
        # Accept both singular and plural forms
        item_pattern = "|".join(
            [f"{item[:-1]}s?" if item.endswith("s") else f"{item}s?" for item in items]
        )
        regex = rf"(?i)Proposal:\s*((?:\s*(?P<num>(10|[0-9]))\s*(?P<item>{item_pattern})\s*,?)+)"
        return regex

    def get_split_action(self, policy_output: str, observation: NoPressObs) -> Split:
        """
        Parse the LLM proposal into a normalized ``Split`` structure.

        The regex-based parser is lenient (accepts pluralization variants) so that
        prompt tweaks do not require re-training the extraction logic.
        """
        items = list(observation.quantities.keys())
        import re as _re

        split_regex = self.get_split_regex(observation)
        items_given_to_self = {item: 0 for item in items}
        m = _re.match(split_regex, policy_output.strip())
        if m:
            # Find all (number, item) pairs
            item_pattern = "|".join(
                [
                    f"{item[:-1]}s?" if item.endswith("s") else f"{item}s?"
                    for item in items
                ]
            )
            inner_regex = rf"(?i)(10|[0-9])\s*({item_pattern})"

            def normalize_item_name(item_str):
                """Canonicalize plural/singular user text back to the config item id."""
                for orig in items:
                    if item_str.lower() == orig.lower():
                        return orig
                    if orig.endswith("s") and item_str.lower() == orig[:-1].lower():
                        return orig
                    if (
                        not orig.endswith("s")
                        and item_str.lower() == orig.lower() + "s"
                    ):
                        return orig

            for num, item in _re.findall(inner_regex, m.group(1)):
                items_given_to_self[normalize_item_name(item)] = int(num)
        return Split(items_given_to_self=items_given_to_self)
