"""
File: run_benchmarks.py
Summary: Orchestrates benchmark runs defined in YAML configs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class AgentSpec:
    name: str
    agent_class_name: str
    policy_id: str
    agent_class_kwargs: Dict[str, Any]


@dataclass
class MasterConfig:
    benchmark: Dict[str, Any]
    environment: Dict[str, Any]
    model: Dict[str, Any]
    adapters: Dict[str, str]
    agents: List[AgentSpec]
    options: Dict[str, Any]


def load_master_config(path: Path) -> MasterConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def req(section: str):
        if section not in raw:
            raise ValueError(f"Missing top-level '{section}' in master config")
        return raw[section]

    agents_raw = req("agents")
    agents: List[AgentSpec] = []
    for a in agents_raw:
        for field in ["name", "agent_class_name", "policy_id"]:
            if field not in a:
                raise ValueError(f"Agent entry missing required field '{field}' -> {a}")
        agents.append(
            AgentSpec(
                name=a["name"],
                agent_class_name=a["agent_class_name"],
                policy_id=a["policy_id"],
                agent_class_kwargs=a.get("agent_class_kwargs", {}) or {},
            )
        )

    return MasterConfig(
        benchmark=req("benchmark"),
        environment=req("environment"),
        model=req("model"),
        adapters=req("adapters"),
        agents=agents,
        options=req("options"),
    )


def load_template(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur = d
    for p in path:
        if p not in cur:
            raise KeyError(f"Missing key path: {'.'.join(path)} (stopped at {p})")
        cur = cur[p]
    return cur


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_matchups(
    agents: List[AgentSpec], include_self_play: bool
) -> List[Tuple[AgentSpec, AgentSpec]]:
    pairs = []
    for a, b in itertools.permutations(agents, 2):
        pairs.append((a, b))
    if include_self_play:
        for a in agents:
            pairs.append((a, a))
    return pairs


def inject_matchup(
    template_cfg: Dict[str, Any],
    master: MasterConfig,
    agent_alice: AgentSpec,
    agent_bob: AgentSpec,
    matchup_id: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(template_cfg)

    # Experiment fields
    experiment_name = master.benchmark.get("name", "benchmark_nego")
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = experiment_name

    # Environment / simulation
    cfg.setdefault("markov_games", {})
    # mg_utils imports the simulation classes into the module namespace, and then uses eval on the
    # provided string. If the master config supplies a fully-qualified path like
    # mllm.markov_games.negotiation.no_press_nego_simulation.NoPressSimulation, the "mllm" package
    # symbol is not itself imported into that eval scope (only the class symbols are), resulting in
    # NameError: name 'mllm' is not defined. We therefore normalize by taking only the final segment
    # after the last dot so that eval("NoPressSimulation") succeeds with the imported symbol.
    raw_sim_class = master.environment["simulation_class_name"]
    sim_class_name = raw_sim_class.split(".")[-1]
    cfg["markov_games"]["simulation_class_name"] = sim_class_name
    cfg["markov_games"]["simulation_init_args"] = master.environment.get(
        "simulation_init_args", {}
    )

    # Primary agent definitions expected by run.py (markov_games.agents)
    def normalize_class_name(full: str) -> str:
        # Same reasoning as simulation_class_name normalization above.
        return full.split(".")[-1]

    # Use each agent's adapter_name directly as the adapter id under bench_llm instead of
    # forcing alice_adapter/bob_adapter indirection. This keeps policy_id aligned with actual
    # adapter identifiers present in adapter_configs / initial_adapter_paths.
    agents_dict = {
        0: {
            "agent_id": "Alice",
            "agent_name": "Alice",
            "agent_class_name": normalize_class_name(agent_alice.agent_class_name),
            "policy_id": agent_alice.policy_id,
            "init_kwargs": agent_alice.agent_class_kwargs,
        },
        1: {
            "agent_id": "Bob",
            "agent_name": "Bob",
            "agent_class_name": normalize_class_name(agent_bob.agent_class_name),
            "policy_id": agent_bob.policy_id,
            "init_kwargs": agent_bob.agent_class_kwargs,
        },
    }
    cfg["markov_games"]["agents"] = agents_dict

    # Remove any legacy hard_coded_buffer_agents section inherited from template to avoid confusion
    if "hard_coded_buffer_agents" in cfg["markov_games"]:
        del cfg["markov_games"]["hard_coded_buffer_agents"]
    if "hard_coded_buffer_agent_kwargs" in cfg["markov_games"]:
        del cfg["markov_games"]["hard_coded_buffer_agent_kwargs"]

    # Provide agent_ids for interpolation used in base.yaml (train_on_which_data)
    cfg["agent_ids"] = ["Alice", "Bob"]

    # Model injection
    cfg.setdefault("models", {})
    cfg["models"].setdefault("bench_llm", {})
    cfg["models"]["bench_llm"].setdefault("init_args", {})
    cfg["models"]["bench_llm"]["init_args"]["model_name"] = master.model["model_name"]
    # Ensure llm_id matches 'bench_llm' so inference policy ids are bench_llm/<adapter>
    cfg["models"]["bench_llm"]["init_args"]["llm_id"] = "bench_llm"

    # Adapter paths mapping for alice/bob
    init_paths = cfg["models"]["bench_llm"]["init_args"].setdefault(
        "initial_adapter_paths", master.adapters
    )

    # Hydra run dir override so that each matchup is segregated
    hydra_cfg = cfg.setdefault("hydra", {}).setdefault("run", {})
    # If template had placeholder we replace with output_root/experiment_name/matchup_id
    hydra_cfg[
        "dir"
    ] = f"{matchup_id}"  # Will be interpreted relative to generated config path root when run with --config-path

    # Ensure defaults list has _self_ (Hydra 1.1 composition order expectation)
    defaults_list = cfg.get("defaults", [])
    if isinstance(defaults_list, list) and all(
        not isinstance(d, dict) for d in defaults_list
    ):
        if "_self_" not in defaults_list:
            defaults_list.append("_self_")
            cfg["defaults"] = defaults_list
    return cfg


def copy_referenced_default_yamls(
    template_path: Path, template_cfg: Dict[str, Any], dest_dir: Path
) -> None:
    """Copy any yaml files referenced in the template defaults list into dest_dir.

    This is needed because we invoke hydra with --config-path pointing at the matchup folder.
    Hydra then expects the defaults (e.g. base.yaml) to be present in that folder.
    """
    defaults_list = template_cfg.get("defaults", [])
    template_dir = template_path.parent
    for entry in defaults_list:
        # Skip dict style or _self_
        if isinstance(entry, dict) or entry == "_self_":
            continue
        if isinstance(entry, str):
            # Support both 'base.yaml' and short form 'base'
            candidate_names = (
                [entry] if entry.endswith(".yaml") else [f"{entry}.yaml", entry]
            )
            found = False
            for name in candidate_names:
                src = template_dir / name
                if src.exists():
                    dst = dest_dir / name
                    if not dst.exists():
                        shutil.copy(src, dst)
                    found = True
                    break
            if not found:
                print(
                    f"[WARN] Referenced defaults entry '{entry}' not found as any of {candidate_names} in {template_dir}."
                )


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_subprocess(match_config_dir: Path, config_stem: str) -> int:
    # Call run.py present in the same directory tree (assumes this script is in repo root analog)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run.py"),
        "--config-path",
        str(match_config_dir),
        "--config-name",
        config_stem,
    ]
    print(f"[INFO] Executing: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log_file = match_config_dir / f"{config_stem}__execution.log"
    log_file.write_text(proc.stdout)
    if proc.returncode != 0:
        print(f"[ERROR] Matchup failed (rc={proc.returncode}). See {log_file}")
    return proc.returncode


def run_render(match_dir: Path, render_keyword: str) -> int:
    """Run the local render.py with the given keyword flag, targeting match_dir.

    Example: render_keyword='nego' -> python render.py --nego <match_dir>
    """
    flag = f"--{render_keyword}"
    cmd = [sys.executable, str(SCRIPT_DIR / "render.py"), flag, str(match_dir)]
    print(f"[INFO] Rendering: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log_file = match_dir / f"render__execution.log"
    log_file.write_text(proc.stdout)
    if proc.returncode != 0:
        print(f"[ERROR] Render failed (rc={proc.returncode}). See {log_file}")
    return proc.returncode


def summarize(results: List[Dict[str, Any]], csv_path: Path | None):
    # Simple console table
    headers = ["matchup", "alice", "bob", "status", "config_dir"]
    col_widths = {h: max(len(h), *(len(str(r[h])) for r in results)) for h in headers}

    def fmt_row(r):
        return " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers)

    print("\n=== Benchmark Summary ===")
    print(fmt_row({h: h for h in headers}))
    print("-" * (sum(col_widths.values()) + 3 * (len(headers) - 1)))
    for r in results:
        print(fmt_row(r))
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"[INFO] CSV summary written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM negotiation benchmarks from a master YAML config"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Master benchmark YAML config path or basename (will search current dir and ./configs)",
    )
    args = parser.parse_args()

    supplied = args.config
    candidate_paths = []
    p = Path(supplied)
    if p.suffix == "":
        # try with .yaml and .yml
        candidate_paths.append(p.with_suffix(".yaml"))
        candidate_paths.append(p.with_suffix(".yml"))
    candidate_paths.append(p)
    # also search ./configs relative to cwd and script dir
    for base in [Path.cwd() / "configs", SCRIPT_DIR / "configs"]:
        for c in list(candidate_paths):
            candidate_paths.append(base / c.name)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for c in candidate_paths:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    master_path = None
    for c in uniq:
        if c.exists():
            master_path = c.resolve()
            break
    if master_path is None:
        raise FileNotFoundError(
            f"Could not locate master config '{supplied}'. Tried: "
            + ", ".join(str(x) for x in uniq)
        )
    master = load_master_config(master_path)

    template_path = (SCRIPT_DIR / master.benchmark["template_config"]).resolve()
    template_cfg = load_template(template_path)

    # Expand environment and date patterns in output_root (basic Hydra-like handling)
    raw_output_root = master.benchmark["output_root"]

    def replace_env(match):
        var = match.group(1)
        return os.environ.get(var, f"UNDEFINED_{var}")

    def replace_now(match):
        fmt = match.group(1)
        return datetime.now().strftime(fmt)

    # Patterns: ${oc.env:VAR} or ${env:VAR}
    pattern_env = re.compile(r"\$\{(?:oc\.env|env):([^}]+)\}")
    pattern_now = re.compile(r"\$\{now:([^}]+)\}")
    expanded = pattern_env.sub(replace_env, raw_output_root)
    expanded = pattern_now.sub(replace_now, expanded)
    output_root = Path(expanded).resolve()
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Root folder starts with benchmark_matrix as requested
    # suite_dir = output_root / f"benchmark_matrix_{master.benchmark.get('name','benchmark')}_{timestamp}"
    suite_dir = (
        output_root / f"benchmark_matrix_{master.benchmark.get('name','benchmark')}"
    )
    ensure_dirs(suite_dir)

    include_self_play = bool(master.options.get("include_self_play", False))
    if not include_self_play:
        print(
            "[WARN] include_self_play is false; self-play matchups (agent vs itself) will NOT be generated."
        )
        print(
            "[INFO] To include naive vs naive etc., set options.include_self_play: true in the master config."
        )
    dry_run = bool(master.options.get("dry_run", False))
    resume = bool(master.options.get("resume", True))
    want_csv = bool(master.options.get("csv_summary", True))

    matchups = build_matchups(master.agents, include_self_play)
    print(f"[INFO] Generated {len(matchups)} matchup(s).")

    results: List[Dict[str, Any]] = []
    for idx, (alice_agent, bob_agent) in enumerate(matchups, start=1):
        # Folder naming: <alice_name>_alice_vs_<bob_name>_bob (add _self if same)
        matchup_slug = f"{alice_agent.name}_alice_vs_{bob_agent.name}_bob"
        match_dir = suite_dir / matchup_slug
        ensure_dirs(match_dir)
        # Copy referenced defaults (e.g., base.yaml) into this matchup folder
        copy_referenced_default_yamls(template_path, template_cfg, match_dir)

        # Generate hydra config file name
        config_stem = "benchmark_matchup"  # consistent stem
        config_path = match_dir / f"{config_stem}.yaml"
        # Write Hydra run outputs directly into the matchup folder (no hydra_outputs subdir)
        matchup_id = str(match_dir)
        cfg = inject_matchup(template_cfg, master, alice_agent, bob_agent, matchup_id)

        # Resume check: skip only if a previous Hydra run wrote its metadata here (.hydra exists)
        hydra_out_dir = Path(cfg["hydra"]["run"]["dir"]).resolve()
        hydra_meta_dir = hydra_out_dir / ".hydra"
        if resume and hydra_meta_dir.exists():
            print(f"[INFO] Skipping existing matchup {matchup_slug} (resume enabled)")
            status = "skipped"
            results.append(
                {
                    "matchup": matchup_slug,
                    "alice": alice_agent.name,
                    "bob": bob_agent.name,
                    "status": status,
                    "config_dir": str(match_dir),
                }
            )
            continue

        write_yaml(config_path, cfg)
        status = "pending"
        if dry_run:
            status = "dry-run"
        else:
            rc = run_subprocess(match_dir, config_stem)
            status = "ok" if rc == 0 else f"fail({rc})"
            # Optional render step
            render_keyword = master.options.get("render_keyword")
            if rc == 0 and render_keyword:
                run_render(match_dir, str(render_keyword))
        results.append(
            {
                "matchup": matchup_slug,
                "alice": alice_agent.name,
                "bob": bob_agent.name,
                "status": status,
                "config_dir": str(match_dir),
            }
        )

    csv_path = suite_dir / "summary.csv" if want_csv else None
    summarize(results, csv_path)
    print(f"[INFO] All done. Suite directory: {suite_dir}")


if __name__ == "__main__":
    main()
