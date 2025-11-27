"""
File: mllm/utils/stat_pack.py
Summary: Implements the StatPack container for incremental statistics.
"""

import csv
import json
import os
import pickle
from collections import Counter
from copy import deepcopy
from locale import strcoll
from statistics import mean
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

plt.style.use(
    "https://raw.githubusercontent.com/dereckpiche/DedeStyle/refs/heads/main/dedestyle.mplstyle"
)

import wandb

from . import wandb_utils


class StatPack:
    def __init__(self):
        self.data = {}

    def add_stat(self, key: str, value: float | int | None):
        assert (
            isinstance(value, float) or isinstance(value, int) or value is None
        ), f"Value {value} is not a valid type"
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def add_stats(self, other: "StatPack"):
        for key in other.keys():
            self.add_stat(key, other[key])

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __contains__(self, key: str):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def mean(self):
        mean_st = StatPack()
        for key in self.keys():
            if isinstance(self[key], list):
                # Ignore None entries so missing measurements do not bias the mean.
                non_none_values = [v for v in self[key] if v is not None]
                if non_none_values:
                    mean_st[key] = np.mean(np.array(non_none_values))
                else:
                    mean_st[key] = None
        return mean_st

    def store_plots(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        for key in self.keys():
            plt.figure(figsize=(10, 5))
            plt.plot(self[key])
            plt.title(key)
            plt.savefig(os.path.join(folder, f"{key}.pdf"))
            plt.close()

    def store_numpy(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        for key in self.keys():
            # Sanitize filename components (avoid slashes, spaces, etc.)
            safe_key = str(key).replace(os.sep, "_").replace("/", "_").replace(" ", "_")
            values = self[key]
            # Convert None to NaN for numpy compatibility
            arr = np.array(
                [(np.nan if (v is None) else v) for v in values], dtype=float
            )
            np.save(os.path.join(folder, f"{safe_key}.npy"), arr)

    def store_json(self, folder: str, filename: str = "stats.json"):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, filename), "w") as f:
            json.dump(self.data, f, indent=4)

    def store_csv(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        for key in self.keys():
            with open(os.path.join(folder, f"stats.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow([key] + self[key])

    def store_pickle(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        for key in self.keys():
            with open(os.path.join(folder, f"stats.pkl"), "wb") as f:
                pickle.dump(self[key], f)
