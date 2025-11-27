"""
File: mllm/markov_games/negotiation/negotiation_statistics.py
Summary: Aggregates and reports statistics for negotiation experiments.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from mllm.markov_games.negotiation.nego_simulation import Split
from mllm.markov_games.rollout_tree import SimulationStepLog


def avg_reward(sl: SimulationStepLog) -> List[Tuple[str, float]]:
    """Average (per-step) reward for each agent and overall.

    What it computes:
            - Returns the raw reward for every (non-buffer) agent at the current
                simulation step.
            - Adds an aggregate key ``all_agents`` which is the simple arithmetic
                mean across the agents present in ``sl.rewards``.

    Rationale / motivation:
            Monitoring the reward stream at each step helps:
                * Diagnose reward shaping issues (e.g., unintended negative drift).
                * Provide a fairness snapshot (are rewards systematically skewed?).
                * Supply a ubiquitous baseline metric used by other higher‑level
                    summaries (efficiency, surplus allocation, etc.).

    Return shape:
            { agent_id: float, ..., "all_agents": float }
            If any agent id contains the substring "buffer" we treat this step as
            an implementation artifact (e.g., rollout buffer) and return ``None``
            to avoid polluting aggregates.
    """
    for aid in sl.rewards.keys():
        if "buffer" in str(aid) and "live" not in str(aid):
            return None
    # One value per agent at each step
    rewards_dict = {f"reward-{aid}": float(v) for aid, v in (sl.rewards or {}).items()}
    return [(key, value) for key, value in rewards_dict.items() if value is not None]


def split_efficiency(sl: SimulationStepLog) -> List[Tuple[str, float]] | None:
    """Final‑round allocation efficiency relative to an upper bound.

    What it computes (only on the last timestep of a negotiation round):
            - Uses ``info['values']`` (per‑agent per‑item valuations) and
                ``info['quantities']`` (available item counts) to form a greedy
                *upper bound* on achievable total reward: allocate each unit of an
                item to the single agent who values that item most.
            - Compares the actually realized sum of rewards at that final
                timestep to this constructed maximum.
            - Emits a single scalar under key ``"all_agents"`` equal to
                achieved / theoretical_max.

    Motivation:
            Efficiency (a core welfare notion) distinguishes between coordination
            failures (low efficiency) versus strategic distributional disputes
            (high efficiency but uneven splits). Tracking this per round helps
            evaluate whether models learn to identify and realize joint surplus.

    Notes / caveats:
            - Only defined for 2+ non‑buffer agents; if a buffer agent is present
                returns ``None`` to exclude spurious steps.
            - Requires the environment to have populated ``values`` and
                ``quantities``; otherwise returns ``None``.
            - This is an optimistic bound (not necessarily reachable under
                protocol constraints) but is simple, fast, and comparable across
                runs.
    """
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    values = info.get("values") or {}
    if not values or not quantities:
        return None
    agent_ids = list(sl.rewards.keys())
    if type(values[agent_ids[0]]) is dict:
        item_keys = list(values.values())[0].keys()
        max_vals, max_quantities = [], []
        for item in item_keys:
            max_val = max(float(agent_vals[item]) for agent_vals in values.values())
            max_vals.append(max_val)
            max_quantities.append(quantities[item])
    else:
        max_vals = [max(float(v) for v in values.values())]
        max_quantities = [quantities[item] for item in quantities.keys()]
    for aid in sl.rewards.keys():
        if "buffer" in str(aid) and "live" not in str(aid):
            return None
    achieved = sum(float(v) for v in sl.rewards.values())
    max_reward = sum(d * v for d, v in zip(max_quantities, max_vals))
    # Efficiency is a global metric; emit same value for a special key "all"
    return [("split_efficiency", achieved / max_reward)]


def _extract_items_from_split(raw_split: Dict) -> Dict[str, float] | None:
    """Return a mapping item->proposal amount from a split structure.

    Supports both generic negotiation splits with nested structure
    { 'items_given_to_self': {item: qty, ...}}
    and TAS coin-only variants which may already be a flat mapping {'coins': qty}.
    """

    if raw_split is None:
        return {}
    elif isinstance(raw_split, Split):
        return {k: float(v) for k, v in raw_split.items_given_to_self.items()}
    elif isinstance(raw_split, dict):
        if "items_given_to_self" in raw_split and isinstance(
            raw_split["items_given_to_self"], dict
        ):
            return {k: float(v) for k, v in raw_split["items_given_to_self"].items()}
        # Fallback: assume already flat mapping of items
        elif hasattr(raw_split, "items_given_to_self"):
            return {k: float(v) for k, v in raw_split["items_given_to_self"].items()}
        return {
            k: float(v) for k, v in raw_split.items() if isinstance(v, (int, float))
        }
    return {}


def _average_proposal_relative_value(
    sl: SimulationStepLog,
    metric_name: str,
    comparator: Callable[[float, float], bool],
    opposite_comparator: Callable[[float, float], bool],
) -> Dict[str, float | None] | None:
    """Shared implementation for proposal size conditioned on relative value.

    Parameters:
            comparator: returns True when agent_0's value relation (e.g. < or >)
                                    to agent_1 holds for an item and we should collect agent_0's
                                    proposed quantity for that item.
            opposite_comparator: inverse relation used to collect agent_1's items.

    Behavior:
            - Executes only on final timestep of a round (where the definitive
                proposal / allocation is known via ``info['splits']``).
            - For each item, classifies which agent's value satisfies the chosen
                relation and records that agent's proposed quantity from the split.
            - Averages (mean) across all qualifying items per agent; if no items
                qualify for an agent returns ``None`` for that agent id.
            - Adds ``all_agents`` mean across the numeric (non-None) agent values.

    Why this matters:
            Distinguishing how much an agent *asks for* when it subjectively
            values items more (or less) than its counterpart reveals patterns of
            opportunism vs. concession. This is especially useful when raw reward
            differences are subtle but allocation *intent* differs.
    """
    info = sl.info or {}
    if not info or not info.get("is_last_timestep_in_round"):
        return None
    quantities = info.get("quantities") or {}
    splits = info.get("splits") or {}
    values = info.get("values") or {}
    agent_ids: List[str] = list(sl.rewards.keys())
    if len(agent_ids) != 2:
        return None  # Only defined for 2-agent case.
    for aid in agent_ids:
        if "buffer" in str(aid) and "live" not in str(aid):
            return None
    # Extract per-agent item proposals robustly
    split_items = {aid: _extract_items_from_split(splits.get(aid)) for aid in agent_ids}
    agent_0_vals: List[float] = []
    agent_1_vals: List[float] = []
    for item in quantities.keys():
        # Values may be either a float (same for all items) or dict per item
        v0_raw = values[agent_ids[0]]
        v1_raw = values[agent_ids[1]]
        v0 = float(v0_raw[item]) if isinstance(v0_raw, dict) else float(v0_raw)
        v1 = float(v1_raw[item]) if isinstance(v1_raw, dict) else float(v1_raw)
        if comparator(v0, v1):
            agent_0_vals.append(split_items[agent_ids[0]].get(item, 0.0))
        elif opposite_comparator(v0, v1):
            agent_1_vals.append(split_items[agent_ids[1]].get(item, 0.0))
    out: Dict[str, float | None] = {}
    out[f"{metric_name}-{agent_ids[0]}"] = (
        sum(agent_0_vals) / len(agent_0_vals) if agent_0_vals else None
    )
    out[f"{metric_name}-{agent_ids[1]}"] = (
        sum(agent_1_vals) / len(agent_1_vals) if agent_1_vals else None
    )

    return [(key, value) for key, value in out.items() if value is not None]


def average_proposal_when_agent_values_item_lower(
    sl: SimulationStepLog,
) -> List[Tuple[str, float | None]] | None:
    """Mean quantity an agent proposes for items it values *less* than opponent.

    Interpretation:
        A higher value implies the agent still claims (or is allocated) a
        notable share of items where it has a comparative *disadvantage* in
        valuation, signaling either strategic over-claiming or protocol-driven
        egalitarian splits. Conversely, very low numbers can indicate
        efficient specialization or excessive concession.

    Returns:
        Mapping { agent_id: float | None, "all_agents": float | None } where
        None indicates no qualifying items for that agent in the round.
    """
    return _average_proposal_relative_value(
        sl,
        "average_proposal_when_agent_values_item_lower",
        lambda a, b: a < b,
        lambda a, b: a > b,
    )


def average_proposal_when_agent_values_item_higher(
    sl: SimulationStepLog,
) -> List[Tuple[str, float | None]] | None:
    """Mean quantity an agent proposes for items it values *more* than opponent.

    Interpretation:
        Captures how aggressively an agent claims items where it holds a
        comparative *advantage*. Elevated values can reflect rational
        specialization (efficient exploitation of comparative advantage) or
        potentially unfair grabs if paired with low concession in the lower
        valuation metric. Comparing this with the 'lower' counterpart helps
        profile negotiation style (cooperative vs. exploitative).

    Returns:
        Mapping { agent_id: float | None, "all_agents": float | None } where
        None indicates no qualifying items.
    """
    return _average_proposal_relative_value(
        sl,
        "average_proposal_when_agent_values_item_higher",
        lambda a, b: a > b,
        lambda a, b: a < b,
    )


# Explicit list of metric functions exported for rendering. Helper functions
# starting with '_' are intentionally excluded. Update this list when adding
# new public statistics so render.py can rely on it instead of introspecting
# every callable in the module.
stat_functs: list[Callable[[SimulationStepLog], List[Tuple[str, float]]]] = [
    avg_reward,
    average_proposal_when_agent_values_item_lower,
    average_proposal_when_agent_values_item_higher,
    split_efficiency,
]
