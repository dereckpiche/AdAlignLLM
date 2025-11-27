"""
File: mllm/markov_games/negotiation/tas_agent.py
Summary: Agent implementation for Take-and-Split negotiations.
"""

from mllm.markov_games.negotiation.nego_agent import NegotiationAgent
from mllm.markov_games.negotiation.nego_simulation import Split
from mllm.markov_games.negotiation.tas_simulation import TrustAndSplitObs


class TrustAndSplitAgent(NegotiationAgent):
    """Prompt/template wrapper for the classic multi-item Take-and-Split benchmark."""

    def __init__(self, num_message_chars, *args, **kwargs):
        self.num_message_chars = num_message_chars
        super().__init__(*args, **kwargs)
        self.intro_prompt = (
            "Welcome to an iterated game. You are {agent}. The other agent is {other_agent}.\n"
            "Setup:\n"
            "1. The game has multiple independent rounds.\n"
            "2. In each round, there are multiple items to split between the two agents.\n"
            "3. Both agents are assigned a per-item value between 1 and 20 (inclusive) in each round.\n"
            "4. You can only observe your own per-item values.\n"
            "5. Because assignments are random, both agents are equally likely to have same expected per-item value.\n"
            "\n"
            "Protocol:\n"
            "1. At the start of the round, one agent begins the conversation. The starting role alternates each round.\n"
            "2. Agents exchange a short chat ({quota_messages_per_agent_per_round} messages per round per agent) to negotiate how to split the item.\n"
            "   - Use this chat to communicate your private per-item value to make informed proposals.\n"
            "3. After the chat, both agents simultaneously propose the amount of each item they will keep.\n"
            "4. If the total sum of proposals is less than or equal to the item quantity, both agents receive their proposed amounts.\n"
            "5. If the total sum of proposals exceeds the item quantity, they are allocated proportionally.\n"
            "6. Your points for the round = (amount you receive per item) x (your per-item value for that round), added across all items.\n"
            "7. Points are accumulated across rounds.\n"
            "Your goal: {goal}\n"
        )
        self.new_round_prompt = (
            "A New Round Begins\n"
            "The items to split are {quantities}.\n"
            "Your per-item values are {value}."
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
        self.send_split_prompt = (
            "Message quota is finished for this round.\n"
            "{other_agent} has finalized their proposal.\n"
            "Submit your finalization now\n"
            "Respond with {proposal_style2}"
        )
        # self.wait_for_message_prompt = "Wait for {other_agent} to send a message..."
        self.wait_for_message_prompt = ""
        self.last_message_prompt = "{other_agent} said: {last_message}"
        # self.send_message_prompt = (
        #     f"Send your message now (max {self.num_message_chars} chars)."
        # )
        self.send_message_prompt = f"Send your message now in <message>...</message> (<={self.num_message_chars} chars)."

    def get_message_regex(self, observation: TrustAndSplitObs) -> str:
        """Constrain chat to bounded XML tags for stable parsing."""
        return rf"<message>[\s\S]{{0,{self.num_message_chars}}}</message>"

    # def get_message_regex(self, observation: TrustAndSplitObs) -> str:
    #     return rf"(?s).{{0,{self.num_message_chars}}}"

    def get_split_regex(self, observation: TrustAndSplitObs) -> str:
        """Allow natural-language item names while still returning machine-parsable XML."""
        items = list(observation.quantities.keys())
        # Accept both singular and plural forms
        item_pattern = "|".join(
            [f"{item[:-1]}s?" if item.endswith("s") else f"{item}s?" for item in items]
        )
        regex = rf"(?i)<items_to_self> ?((?:\s*(?P<num>(10|[0-9]))\s*(?P<item>{item_pattern})\s*,?)+) ?</items_to_self>"
        return regex

    def get_split_action(
        self, policy_output: str, observation: TrustAndSplitObs
    ) -> Split:
        """Convert human-readable allocation text back into canonical item IDs."""
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
