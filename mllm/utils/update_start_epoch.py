"""
File: mllm/utils/update_start_epoch.py
Summary: Updates persisted start-epoch metadata when resuming runs.
"""

import os


# During run, set hydra.run.dir=./outputs/{folder}
def update_start_epoch(cfg, output_directory):
    if cfg["experiment"]["resume_experiment"]:
        folders = [
            f for f in os.listdir(output_directory) if f.startswith("iteration_")
        ]
        iterations = [int(f.split("_")[1]) for f in folders] if folders else [0]
        cfg["experiment"]["start_epoch"] = max(iterations)
    return None
