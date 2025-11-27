"""
File: render.py
Summary: CLI for aggregating rollout trees, stats, and charts.
"""

import argparse
import importlib
import json
import math
import os
import pickle
import shutil
import sys
import textwrap
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import wandb
import yaml
from matplotlib.ticker import MaxNLocator

from mllm.markov_games.rollout_tree import RolloutTreeRootNode
from mllm.utils.gather_training_stats import (
    render_rt_tally_pkl_to_csvs,
    tally_to_stat_pack,
)
from mllm.utils.rollout_tree_chat_htmls import export_html_from_rollout_tree
from mllm.utils.rollout_tree_stats import get_mean_rollout_tree_stats
from mllm.utils.stat_pack import StatPack

# Optional progress bar; tqdm is nice-to-have but not required for headless runs.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def gather_iteration_rollout_trees(iteration_folder):
    """
    Load every ``*.rt.pkl`` under an iteration folder and return validated trees.

    Skips corrupt/empty files but continues processing the rest so a few failures
    don’t block the entire render job.
    """
    rollout_trees = []
    iteration_path = Path(iteration_folder)
    for item in iteration_path.glob("**/*.rt.pkl"):
        try:
            # Skip empty or partially written files
            if item.stat().st_size == 0:
                print(f"Skipping empty pickle file: {item}")
                continue
            with open(item, "rb") as f:
                data = pickle.load(f)
            rollout_tree = RolloutTreeRootNode.model_validate(data)
            rollout_trees.append(rollout_tree)
        except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
            print(f"Warning: failed to load {item}: {e}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: unexpected error loading {item}: {e}. Skipping.")
            continue
    return rollout_trees


def find_iteration_folders(
    global_folder: str | Path,
    from_iteration: int = 0,
    last_iteration: Optional[int] = None,
) -> List[Path]:
    """
    Enumerate iteration_* directories (possibly under seed_* subfolders) in order.

    Respects ``from_iteration``/``last_iteration`` bounds to let users render a
    subset of iterations.
    """
    base = Path(global_folder)

    candidates: List[Path] = []

    # Search in the folder itself
    for item in base.glob("iteration_*"):
        if item.is_dir():
            candidates.append(item)

    # Search in seed_* subdirectories
    for seed_dir in base.glob("seed_*/"):
        if seed_dir.is_dir():
            for item in seed_dir.glob("iteration_*"):
                if item.is_dir():
                    candidates.append(item)

    # Parse numeric iteration indices and filter
    def parse_idx(p: Path) -> Optional[int]:
        name = p.name
        try:
            return int(name.split("_")[-1])
        except Exception:
            return None

    filtered: List[tuple[int, Path]] = []
    for p in candidates:
        idx = parse_idx(p)
        if idx is None:
            continue
        if idx < from_iteration:
            continue
        if (last_iteration is not None) and (idx > last_iteration):
            continue
        filtered.append((idx, p))

    # Sort numerically by idx
    filtered.sort(key=lambda t: t[0])
    return [p for (_idx, p) in filtered]


def clean_render_artifacts(root_folder: Path) -> int:
    """
    Delete any ``*.render.*`` files/directories under ``root_folder`` (cleanup mode).

    Returns how many artifacts were removed so callers can report the cleanup size.
    """
    removed_count = 0

    try:
        all_paths = list(root_folder.rglob("*"))
    except Exception:
        all_paths = []

    file_targets = [p for p in all_paths if ".render." in p.name and p.is_file()]
    dir_targets = [p for p in all_paths if ".render." in p.name and p.is_dir()]

    # Delete files first
    for path in file_targets:
        try:
            path.unlink(missing_ok=True)
            removed_count += 1
        except Exception as e:
            print(f"Warning: failed to delete file {path}: {e}")

    # Delete directories deepest-first
    for path in sorted(dir_targets, key=lambda p: len(p.parts), reverse=True):
        try:
            shutil.rmtree(path, ignore_errors=False)
            removed_count += 1
        except Exception as e:
            print(f"Warning: failed to delete directory {path}: {e}")

    return removed_count


def discover_metric_functions(module_name: str) -> list[Callable[[Any], Any]]:
    """
    Import a statistics module and grab the ``stat_functs`` list it exposes.

    Each callable is expected to accept a ``SimulationStepLog`` and return
    key/value metrics (see negotiation/ipd stats modules for examples).
    """
    mod = importlib.import_module(module_name)
    metrics: list[Callable[[Any], Any]] = []
    # Module must have a stat_functs list attribute
    metrics = getattr(mod, "stat_functs", None)
    return metrics


def load_root(path: Path) -> dict:
    """Load rollout tree as raw dict without Pydantic validation (fast)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_iteration_mean_stats(
    iteration_folder: Path, metrics: list[Callable[[Any], Any]]
) -> StatPack:
    rollout_trees = gather_iteration_rollout_trees(iteration_folder)
    return get_mean_rollout_tree_stats(rollout_trees, metrics)


def get_run_mean_stats_evolution(
    run_folder: Path, metrics: list[Callable[[Any], Any]]
) -> StatPack:
    iteration_folders = find_iteration_folders(run_folder)
    stats_evolution = StatPack()
    stat_tallies = [
        get_iteration_mean_stats(iteration_folder, metrics)
        for iteration_folder in iteration_folders
    ]
    for stat_tally in stat_tallies:
        stats_evolution.add_stats(stat_tally)
    return stats_evolution


def get_training_mean_stats(iteration_folder: Path) -> StatPack:
    file_exists = False
    for fname in sorted(os.listdir(iteration_folder)):
        if fname.endswith(".tally.pkl"):
            pkl_path = os.path.join(iteration_folder, fname)
            tally = pickle.load(open(pkl_path, "rb"))
            training_stats: StatPack = tally_to_stat_pack(tally)
            file_exists = True
    if not file_exists:
        training_stats = StatPack()
    return training_stats.mean()


def get_training_mean_stats_evolution(iteration_folders: List[Path]) -> StatPack:
    training_stats_evolution = StatPack()
    for iteration_folder in iteration_folders:
        training_stats = get_training_mean_stats(iteration_folder)
        training_stats_evolution.add_stats(training_stats)
    return training_stats_evolution


def render_iteration_training_csvs(iteration_dir: str):
    """Convert ``*.rt_tally.pkl`` files into CSVs for quick ad-hoc inspection."""
    input_dir = iteration_dir
    output_dir = os.path.join(iteration_dir, "trainer_stats.render.")
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".rt_tally.pkl"):
            pkl_path = os.path.join(input_dir, fname)
            render_rt_tally_pkl_to_csvs(pkl_path=pkl_path, outdir=output_dir)


def render_training_csvs(iteration_folders: List[Path]):
    for iteration_folder in iteration_folders:
        render_iteration_training_csvs(
            iteration_dir=str(iteration_folder),
        )


def render_iteration_chat_htmls(
    input_dir,
):
    """
    Render chat transcripts (HTML) for every rollout tree inside ``input_dir``.

    Useful for qualitative analysis or sharing dialogues with non-technical folks.
    """
    input_path = Path(input_dir)

    # If no output_dir specified, create analysis files in the same input folder
    output_path = input_path

    pattern = "**/*.rt.pkl"
    files = sorted(input_path.glob(pattern))
    if not files:
        print(f"No PKL rollout trees found in {input_path}.")
        return False

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing folder: {input_path}")
    print(f"Output folder: {output_path}")

    for i, f in enumerate(files, 1):
        export_html_from_rollout_tree(
            path=f,
            outdir=output_path,
            main_only=False,
        )


def render_chat_htmls(iteration_folders: List[Path]):
    for iteration_folder in iteration_folders:
        render_iteration_chat_htmls(
            input_dir=str(iteration_folder),
        )


def render(
    paperdata_folder: Path,
    plot_folder: Path,
    metrics: list[Callable[[Any], Any]],
    from_iteration: int,
    last_iteration: int,
    root_folder: List[Path],
    export_to_wandb: bool,
    should_render_rollout_tree_stats: bool,
    should_render_html: bool,
    should_render_training_csvs: bool,
):
    # Step 1: find which iteration folders we're going to process.
    iteration_folders = find_iteration_folders(
        root_folder, from_iteration=from_iteration, last_iteration=last_iteration
    )

    # Optional: export dialogue replay HTMLs.
    if should_render_html:
        try:
            print(f"Rendering chats as HTML files")
            render_chat_htmls(iteration_folders)
            print(f"Chats rendered as HTML files")
        except Exception as e:
            print(f"Error rendering chats as HTML files: {e}")

    # Optional: dump trainer tallies to structured CSVs for spreadsheet review.
    if should_render_training_csvs:
        try:
            print(f"Rendering training stats")
            render_training_csvs(iteration_folders)
            print(f"Training stats rendered")
        except Exception as e:
            print(f"Error rendering training stats: {e}")

    # Optional: compute aggregate rollout stats (plots/JSON) for paper figures.
    if should_render_rollout_tree_stats:
        print(f"Gathering and storing statistics")
        evolution_stats = get_run_mean_stats_evolution(root_folder, metrics)
        stats_folder = paperdata_folder
        stats_folder.mkdir(exist_ok=True)
        evolution_stats.store_json(
            folder=paperdata_folder, filename="rollout_tree_stats.json"
        )
        evolution_stats.store_numpy(folder=stats_folder)
        evolution_stats.store_plots(folder=plot_folder)

        print(f"Rollout tree stats gathered and stored")

    # Always collect averaged training scalars (even if rollout stats were skipped).
    training_stats = get_training_mean_stats_evolution(iteration_folders)
    training_stats.store_json(folder=paperdata_folder, filename="training_stats.json")
    print(f"Training stats gathered and stored")
    if export_to_wandb:
        complete_stats = {}
        complete_stats.update(training_stats.data)
        complete_stats.update(evolution_stats.data)
        key0 = list(complete_stats.keys())[0]
        for step in range(len(complete_stats[key0])):
            log_dict = {}
            for key in complete_stats.keys():
                log_dict[key] = complete_stats[key][step]
            wandb.log(log_dict, step=step)
    # (Stats gathering used to be wrapped in a broad try/except; we now bubble errors.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Post-process an experiment directory: clean artifacts, "
            "render chats, compute stats, or export summaries."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Experiment root or seed folder containing iteration_* (default: .)",
    )

    # Game selection
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--ipd",
        action="store_true",
        help="Use Iterated Prisoner’s Dilemma stats module",
    )
    g.add_argument(
        "--nego",
        action="store_true",
        help="Use negotiation statistics module (e.g., TAS, Deal-or-No-Deal)",
    )

    parser.add_argument(
        "--from-iteration",
        type=int,
        default=0,
        help="Start at iteration index (inclusive)",
    )
    parser.add_argument(
        "--to-iteration",
        type=int,
        default=None,
        help="Stop at iteration index (inclusive)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete residual `.render.` files (skip stats/render steps)",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Export to wandb",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Render HTML files",
    )

    parser.add_argument(
        "--training-csvs",
        action="store_true",
        help="Render training csvs",
    )

    args = parser.parse_args()
    root_folder = Path(args.path or ".").resolve()

    plot_folder = root_folder / "0B_plots"
    # Early clean and exit if requested
    if args.clean:
        removed = clean_render_artifacts(root_folder)
        print(f"Deleted {removed} '.render.' artifacts under {root_folder}")
        sys.exit(0)

    # If not cleaning, require a game selection
    if not (args.ipd or args.nego):
        parser.error("One of --ipd or --nego is required unless --clean is used.")

    # 1) Discover metrics based on game kind
    if args.ipd:
        stats_mod = "mllm.markov_games.ipd.ipd_statistics"
        game_kind = "ipd"
    else:
        stats_mod = "mllm.markov_games.negotiation.negotiation_statistics"
        game_kind = "negotiation"

    metrics = discover_metric_functions(stats_mod)

    with open(os.path.join(root_folder / ".hydra", "config.yaml"), "r") as f:
        hydra_config = yaml.safe_load(f)
        experiment_name = hydra_config["experiment"]["name"]
    # Export config to paperdata folder
    paperdata_folder = root_folder / f"0A_paperdata_for_{experiment_name}"
    paperdata_folder.mkdir(exist_ok=True)
    with open(os.path.join(paperdata_folder, "config.yaml"), "w") as f:
        yaml.dump(hydra_config, f)

    if args.wandb:
        # Load hydra config to get experiment name
        wandb.init(project="llm_negotiation", name=experiment_name, config=hydra_config)

    render(
        paperdata_folder=paperdata_folder,
        plot_folder=plot_folder,
        metrics=metrics,
        from_iteration=args.from_iteration,
        last_iteration=args.to_iteration,
        root_folder=root_folder,
        export_to_wandb=args.wandb,
        should_render_rollout_tree_stats=not args.html,
        should_render_html=args.html,
        should_render_training_csvs=args.training_csvs,
    )
