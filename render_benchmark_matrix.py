"""
File: render_benchmark_matrix.py
Summary: Builds benchmarking matrices from recorded experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def build_faceoff_matrix_csv(suite_dir: str) -> Path:
    """
    Walk a benchmark_matrix suite directory and produce `faceoff_matrix.csv`.

    - Expects matchup subfolders named like `<alice>_alice_vs_<bob>_bob`.
    - Each matchup should contain a `0A_paperdata.../rollout_tree_stats.json`.
    - The resulting matrix is NxN (agents) where entry (i,j) aggregates Alice/Bob
      averages so you can quickly eyeball who beats whom.
    """
    root = Path(suite_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {root}")

    matchup_name_re = re.compile(r"^(.+?)_alice_vs_(.+?)_bob$")
    matchup_dirs: List[Path] = [p for p in root.iterdir() if p.is_dir()]

    # Discover agents in stable first-seen order
    agents: List[str] = []

    def ensure_agent(name: str) -> None:
        if name not in agents:
            agents.append(name)

    # Collect average rewards for Alice per matchup
    # key: (alice_agent, bob_agent) -> avg_reward_alice
    faceoff_values_alice: Dict[Tuple[str, str], float] = {}
    faceoff_values_bob: Dict[Tuple[str, str], float] = {}

    for d in sorted(matchup_dirs, key=lambda x: x.name):
        m = matchup_name_re.match(d.name)
        if not m:
            continue
        alice_agent, bob_agent = m.group(1), m.group(2)
        ensure_agent(alice_agent)
        ensure_agent(bob_agent)

        # Find subfolder that contains the 0A_paperdata outputs
        paperdata_dir = next(
            (p for p in d.iterdir() if p.is_dir() and "0A_paperdata" in p.name),
            None,
        )
        if paperdata_dir is None:
            # Skip if the expected stats folder is missing
            continue
        stats_path = paperdata_dir / "rollout_tree_stats.json"
        if not stats_path.exists():
            # Skip silently if stats are missing (e.g., failed/unfinished run)
            continue
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
        except Exception:
            continue

        # Expect arrays of rewards; compute mean for Alice
        rewards_alice_agent = stats.get("reward-Alice") or stats.get("reward_alice")
        rewards_bob_agent = stats.get("reward-Bob") or stats.get("reward_bob")

        faceoff_values_alice[(alice_agent, bob_agent)] = rewards_alice_agent[0]
        faceoff_values_bob[(bob_agent, alice_agent)] = rewards_bob_agent[0]

    # Build matrix
    n = len(agents)
    index_of: Dict[str, int] = {name: i for i, name in enumerate(agents)}
    matrix: List[List[str]] = [["" for _ in range(n)] for _ in range(n)]

    # pairs of agents:
    pairs_of_agents = list(
        set(faceoff_values_alice.keys()) | set(faceoff_values_bob.keys())
    )

    for agent_1, agent_2 in pairs_of_agents:
        value_of_agent_1_against_agent_2_as_alice = faceoff_values_alice[
            (agent_1, agent_2)
        ]
        value_of_agent_1_against_agent_2_as_bob = faceoff_values_bob[(agent_1, agent_2)]
        i = index_of[agent_1]
        j = index_of[agent_2]
        matrix[i][j] = (
            value_of_agent_1_against_agent_2_as_alice
            + value_of_agent_1_against_agent_2_as_bob
        ) / 2

    # Write CSV with header row/col of agent names
    out_path = root / "faceoff_matrix.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + agents)
        for i, row_name in enumerate(agents):
            writer.writerow([row_name] + matrix[i])

    print(f"[INFO] faceoff_matrix.csv written: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render faceoff matrix CSV from a benchmark_matrix suite directory"
    )
    parser.add_argument(
        "suite_dir",
        type=str,
        help="Path to benchmark_matrix suite directory (folder containing matchup subfolders)",
    )
    args = parser.parse_args()

    try:
        build_faceoff_matrix_csv(args.suite_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
