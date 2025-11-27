"""
File: mllm/markov_games/statistics_runner.py
Summary: Executes multiple rollouts to compute experiment statistics.
"""

from __future__ import annotations

import gc
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from basic_render import find_iteration_folders

from mllm.markov_games.rollout_tree import (
    RolloutTreeBranchNode,
    RolloutTreeNode,
    RolloutTreeRootNode,
    SimulationStepLog,
)


def _iterate_main_nodes(root: RolloutTreeRootNode) -> Iterator[RolloutTreeNode]:
    """
    Iterate the main path nodes without materializing full path lists.
    """
    current = root.child
    while current is not None:
        if isinstance(current, RolloutTreeNode):
            yield current
            current = current.child
        elif isinstance(current, RolloutTreeBranchNode):
            # Follow only the main child on the main trajectory
            current = current.main_child
        else:
            break


def iterate_main_simulation_logs(
    root: RolloutTreeRootNode,
) -> Iterator[SimulationStepLog]:
    """Yield ``SimulationStepLog`` objects along the main (non-branch) path."""
    for node in _iterate_main_nodes(root):
        yield node.step_log.simulation_step_log


def stream_rollout_files(iteration_folder: Path) -> Iterator[Path]:
    """Iterate over every ``*.rt.pkl`` file under an iteration directory."""
    for p in iteration_folder.rglob("*.rt.pkl"):
        if p.is_file():
            yield p


def load_root(path: Path) -> RolloutTreeRootNode:
    """Load and validate a rollout tree from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return RolloutTreeRootNode.model_validate(data)


@dataclass
class StatRecord:
    """Convenience container for serialized stat rows."""

    mgid: int
    crn_id: Optional[int]
    iteration: str
    values: Dict[str, Any]


class StatComputer:
    """
    Stateful stat computer that consumes SimulationStepLog instances
    and produces final aggregated values for one rollout (mgid).
    """

    def update(self, sl: SimulationStepLog) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


def run_stats(
    data_root: Path,
    game_name: str,
    make_computers: Callable[[], List[StatComputer]],
    output_filename: Optional[str] = None,
    output_format: str = "json",  # "json" (dict of lists) or "jsonl"
) -> Path:
    """
    Compute stats across all iteration_* folders under data_root.
    Writes JSONL to data_root/statistics/<output_filename or f"{game_name}.stats.jsonl">.
    """
    data_root = Path(data_root)
    outdir = data_root / "statistics"
    outdir.mkdir(parents=True, exist_ok=True)
    # Choose extension by format
    default_name = (
        f"{game_name}.stats.json"
        if output_format == "json"
        else f"{game_name}.stats.jsonl"
    )
    outfile = outdir / (
        output_filename if output_filename is not None else default_name
    )

    # Rewrite file each run to keep it clean and small
    if outfile.exists():
        outfile.unlink()

    iteration_folders = find_iteration_folders(str(data_root))

    # If writing JSONL, stream directly; otherwise accumulate minimal records
    if output_format == "jsonl":
        with open(outfile, "w", encoding="utf-8") as w:
            for iteration_folder in iteration_folders:
                iteration_name = Path(iteration_folder).name
                for pkl_path in stream_rollout_files(Path(iteration_folder)):
                    root = load_root(pkl_path)

                    computers = make_computers()
                    for sl in iterate_main_simulation_logs(root):
                        for comp in computers:
                            try:
                                comp.update(sl)
                            except Exception:
                                continue

                    values: Dict[str, Any] = {}
                    for comp in computers:
                        try:
                            values.update(comp.finalize())
                        except Exception:
                            continue

                    rec = {
                        "mgid": getattr(root, "id", None),
                        "crn_id": getattr(root, "crn_id", None),
                        "iteration": iteration_name,
                        "stats": values,
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    del root
                    del computers
                    gc.collect()
    else:
        # Aggregate to dict-of-lists for easier plotting
        records: List[Dict[str, Any]] = []
        # Process in deterministic order
        for iteration_folder in iteration_folders:
            iteration_name = Path(iteration_folder).name
            for pkl_path in stream_rollout_files(Path(iteration_folder)):
                root = load_root(pkl_path)

                computers = make_computers()
                for sl in iterate_main_simulation_logs(root):
                    for comp in computers:
                        try:
                            comp.update(sl)
                        except Exception:
                            continue

                values: Dict[str, Any] = {}
                for comp in computers:
                    try:
                        values.update(comp.finalize())
                    except Exception:
                        continue

                records.append(
                    {
                        "mgid": getattr(root, "id", None),
                        "crn_id": getattr(root, "crn_id", None),
                        "iteration": iteration_name,
                        "stats": values,
                    }
                )

                del root
                del computers
                gc.collect()

        # Build dict-of-lists with nested stats preserved
        # Collect all stat keys and nested agent keys where needed
        mgids: List[Any] = []
        crn_ids: List[Any] = []
        iterations_out: List[str] = []
        # stats_out is a nested structure mirroring keys but with lists
        stats_out: Dict[str, Any] = {}

        # First pass to collect union of keys
        stat_keys: set[str] = set()
        nested_agent_keys: Dict[str, set[str]] = {}
        for r in records:
            stats = r.get("stats", {}) or {}
            for k, v in stats.items():
                stat_keys.add(k)
                if isinstance(v, dict):
                    nested = nested_agent_keys.setdefault(k, set())
                    for ak in v.keys():
                        nested.add(str(ak))

        # Initialize structure
        for k in stat_keys:
            if k in nested_agent_keys:
                stats_out[k] = {ak: [] for ak in sorted(nested_agent_keys[k])}
            else:
                stats_out[k] = []

        # Fill lists
        for r in records:
            mgids.append(r.get("mgid"))
            crn_ids.append(r.get("crn_id"))
            iterations_out.append(r.get("iteration"))
            stats = r.get("stats", {}) or {}
            for k in stat_keys:
                val = stats.get(k)
                if isinstance(stats_out[k], dict):
                    # per-agent dict
                    agent_dict = val if isinstance(val, dict) else {}
                    for ak in stats_out[k].keys():
                        stats_out[k][ak].append(agent_dict.get(ak))
                else:
                    stats_out[k].append(val)

        with open(outfile, "w", encoding="utf-8") as w:
            json.dump(
                {
                    "mgid": mgids,
                    "crn_id": crn_ids,
                    "iteration": iterations_out,
                    "stats": stats_out,
                },
                w,
                ensure_ascii=False,
            )

    return outfile


def run_stats_functional(
    data_root: Path,
    game_name: str,
    metrics: Dict[str, Callable[[SimulationStepLog], Optional[Dict[str, float]]]],
    output_filename: Optional[str] = None,
    output_format: str = "json",
) -> Path:
    """
    Functional variant where metrics is a dict of name -> f(SimulationStepLog) -> {agent_id: value}.
    Aggregates per rollout by averaging over steps where a metric produced a value.
    Writes a single consolidated file in data_root/statistics/.
    """
    data_root = Path(data_root)
    outdir = data_root / "statistics"
    outdir.mkdir(parents=True, exist_ok=True)
    default_name = (
        f"{game_name}.stats.json"
        if output_format == "json"
        else f"{game_name}.stats.jsonl"
    )
    outfile = outdir / (
        output_filename if output_filename is not None else default_name
    )

    if outfile.exists():
        outfile.unlink()

    iteration_folders = find_iteration_folders(str(data_root))

    def finalize_rollout(
        agg: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, float]]:
        # avg per metric per agent
        result: Dict[str, Dict[str, float]] = {}
        for mname, agent_values in agg.items():
            result[mname] = {}
            for aid, vals in agent_values.items():
                if not vals:
                    result[mname][aid] = None  # keep alignment; could be None
                else:
                    result[mname][aid] = sum(vals) / len(vals)
        return result

    if output_format == "jsonl":
        with open(outfile, "w", encoding="utf-8") as w:
            for iteration_folder in iteration_folders:
                iteration_name = Path(iteration_folder).name
                for pkl_path in stream_rollout_files(Path(iteration_folder)):
                    root = load_root(pkl_path)

                    # aggregator structure: metric -> agent_id -> list of values
                    agg: Dict[str, Dict[str, List[float]]] = {
                        m: {} for m in metrics.keys()
                    }

                    for sl in iterate_main_simulation_logs(root):
                        for mname, fn in metrics.items():
                            try:
                                vals = fn(sl)
                            except Exception:
                                vals = None
                            if not vals:
                                continue
                            for aid, v in vals.items():
                                if v is None:
                                    continue
                                lst = agg[mname].setdefault(str(aid), [])
                                try:
                                    lst.append(float(v))
                                except Exception:
                                    continue

                    values = finalize_rollout(agg)
                    rec = {
                        "mgid": getattr(root, "id", None),
                        "crn_id": getattr(root, "crn_id", None),
                        "iteration": iteration_name,
                        "stats": values,
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    del root
                    gc.collect()
    else:
        records: List[Dict[str, Any]] = []
        for iteration_folder in iteration_folders:
            iteration_name = Path(iteration_folder).name
            for pkl_path in stream_rollout_files(Path(iteration_folder)):
                root = load_root(pkl_path)

                agg: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics.keys()}
                for sl in iterate_main_simulation_logs(root):
                    for mname, fn in metrics.items():
                        try:
                            vals = fn(sl)
                        except Exception:
                            vals = None
                        if not vals:
                            continue
                        for aid, v in vals.items():
                            if v is None:
                                continue
                            lst = agg[mname].setdefault(str(aid), [])
                            try:
                                lst.append(float(v))
                            except Exception:
                                continue

                values = finalize_rollout(agg)
                records.append(
                    {
                        "mgid": getattr(root, "id", None),
                        "crn_id": getattr(root, "crn_id", None),
                        "iteration": iteration_name,
                        "stats": values,
                    }
                )

                del root
                gc.collect()

        # Build dict-of-lists output
        mgids: List[Any] = []
        crn_ids: List[Any] = []
        iterations_out: List[str] = []
        stats_out: Dict[str, Any] = {}

        stat_keys: set[str] = set()
        nested_agent_keys: Dict[str, set[str]] = {}
        for r in records:
            stats = r.get("stats", {}) or {}
            for k, v in stats.items():
                stat_keys.add(k)
                if isinstance(v, dict):
                    nested = nested_agent_keys.setdefault(k, set())
                    for ak in v.keys():
                        nested.add(str(ak))

        for k in stat_keys:
            if k in nested_agent_keys:
                stats_out[k] = {ak: [] for ak in sorted(nested_agent_keys[k])}
            else:
                stats_out[k] = []

        for r in records:
            mgids.append(r.get("mgid"))
            crn_ids.append(r.get("crn_id"))
            iterations_out.append(r.get("iteration"))
            stats = r.get("stats", {}) or {}
            for k in stat_keys:
                val = stats.get(k)
                if isinstance(stats_out[k], dict):
                    agent_dict = val if isinstance(val, dict) else {}
                    for ak in stats_out[k].keys():
                        stats_out[k][ak].append(agent_dict.get(ak))
                else:
                    stats_out[k].append(val)

        with open(outfile, "w", encoding="utf-8") as w:
            json.dump(
                {
                    "mgid": mgids,
                    "crn_id": crn_ids,
                    "iteration": iterations_out,
                    "stats": stats_out,
                },
                w,
                ensure_ascii=False,
            )

    return outfile
