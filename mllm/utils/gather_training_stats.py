"""
File: mllm/utils/gather_training_stats.py
Summary: Aggregates training statistics from rollouts and exports artifacts.
"""

import copy
import csv
import gc
import json
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean
from typing import Any, Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from mllm.training.tally_metrics import Tally
from mllm.utils.stat_pack import StatPack


def get_from_nested_dict(dictio: dict, path: list[str]):
    for sp in path[:-1]:
        dictio = dictio[sp]
    return dictio.get(path[-1])


def set_at_path(dictio: dict, path: list[str], value):
    for sp in path[:-1]:
        if sp not in dictio:
            dictio[sp] = {}
        dictio = dictio[sp]
    dictio[path[-1]] = value


def produce_tabular_render(inpath: str, outpath: str = None):
    """
    Convert a JSON metrics dump into per-rollout CSV tables for easier inspection.
    """
    with open(inpath, "r") as f:
        data = json.load(f)
    rollout_paths = data.keys()
    for rollout_path in rollout_paths:
        if outpath is None:
            m_path = rollout_path.replace("/", "|")
            m_path = m_path.replace(".json", "")
            m_path = (
                os.path.split(inpath)[0]
                + "/contextualized_tabular_renders/"
                + m_path
                + "_tabular_render.render.csv"
            )
        # import pdb; pdb.set_trace()
        os.makedirs(os.path.split(m_path)[0], exist_ok=True)
        metrics = data[rollout_path]
        d = {k: [] for k in metrics[0].keys()}
        for m in metrics:
            for k, v in m.items():
                d[k].append(v)
        d = pd.DataFrame(d)
        d.to_csv(m_path)


def get_metric_paths(data: list[dict]):
    d = data[0]
    paths = []

    def traverse_dict(d, current_path=[]):
        for key, value in d.items():
            new_path = current_path + [key]
            if isinstance(value, dict):
                traverse_dict(value, new_path)
            else:
                paths.append(new_path)

    traverse_dict(d)
    return paths


def print_metric_paths(data: list[dict]):
    paths = get_metric_paths(data)
    for p in paths:
        print(p)


def get_metric_iteration_list(data: list[dict], metric_path: list[str]):
    if isinstance(metric_path, str):
        metric_path = [metric_path]
    sgl = []
    for d in data:
        sgl.append(get_from_nested_dict(d, metric_path))
    return sgl


def to_1d_numeric(x):
    """Return a 1-D float array (or None if not numeric). Accepts scalars, numpy arrays, or nested list/tuple of them."""
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return np.array([float(x)], dtype=float)
    if isinstance(x, np.ndarray):
        try:
            return x.astype(float).ravel()
        except Exception:
            return None
    if isinstance(x, (list, tuple)):
        parts = []
        for e in x:
            arr = to_1d_numeric(e)
            if arr is not None and arr.size > 0:
                parts.append(arr)
        if parts:
            return np.concatenate(parts)
        return None
    return None


def get_single_metric_vector(data, metric_path, iterations=None):
    if isinstance(metric_path, str):
        metric_path = [metric_path]
    if iterations == None:
        iterations = len(data)
    vecs = []
    for d in data:
        ar = get_from_nested_dict(d, metric_path)
        arr = to_1d_numeric(ar)
        if arr is not None:
            vecs.append(arr)

    return np.concatenate(vecs) if vecs else np.empty(0, dtype=float)


def _load_metrics_file(file_path: str):
    if not (file_path.endswith(".tally.pkl") or file_path.endswith(".pkl")):
        raise ValueError("Only *.tally.pkl files are supported.")
    import pickle

    with open(file_path, "rb") as f:
        tree = pickle.load(f)
    return tree


def get_leaf_items(array_tally: dict, prefix: list[str] = None):
    if prefix is None:
        prefix = []
    for key, value in array_tally.items():
        next_prefix = prefix + [str(key)]
        if isinstance(value, dict):
            yield from get_leaf_items(value, next_prefix)
        else:
            yield next_prefix, value


def _sanitize_filename_part(part: str) -> str:
    s = part.replace("/", "|")
    s = s.replace(" ", "_")
    return s


def render_rt_tally_pkl_to_csvs(pkl_path: str, outdir: str):
    """
    This method takes care of tokenwise logging.
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    # Backward compatibility: older tallies stored the dict directly
    if isinstance(payload, dict) and "array_tally" in payload:
        array_tally = payload.get("array_tally", {})
    else:
        array_tally = payload

    os.makedirs(outdir, exist_ok=True)
    trainer_id = os.path.basename(pkl_path).replace(".rt_tally.pkl", "")
    for path_list, rollout_tally_items in get_leaf_items(array_tally):
        # Create file and initiate writer
        path_part = ".".join(_sanitize_filename_part(p) for p in path_list)
        filename = f"{trainer_id}__{path_part}.render.csv"
        out_path = os.path.join(outdir, filename)

        # Write metric rows to CSV
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header row - need to determine metric column count from first rollout_tally_item
            first_item = rollout_tally_items[0]
            metric_cols = (
                first_item.metric_matrix.shape[1]
                if first_item.metric_matrix.ndim > 1
                else 1
            )
            header = ["agent_id", "crn_id", "rollout_id"] + [
                f"t_{i}" for i in range(metric_cols)
            ]
            writer.writerow(header)

            for rollout_tally_item in rollout_tally_items:
                crn_ids = rollout_tally_item.crn_ids
                rollout_ids = rollout_tally_item.rollout_ids
                agent_ids = rollout_tally_item.agent_ids
                metric_matrix = rollout_tally_item.metric_matrix
                for i in range(metric_matrix.shape[0]):
                    row_vals = metric_matrix[i].reshape(-1)
                    # Convert row_vals to a list to avoid numpy concatenation issues
                    row_vals = (
                        row_vals.tolist()
                        if hasattr(row_vals, "tolist")
                        else list(row_vals)
                    )
                    row_prefix = [
                        agent_ids[i],
                        crn_ids[i],
                        rollout_ids[i],
                    ]
                    writer.writerow(row_prefix + row_vals)


def tally_to_stat_pack(tally: Dict[str, Any]):
    stat_pack = StatPack()
    if "array_tally" in tally:
        tally = tally["array_tally"]

        # backward compatibility: will remove later, flatten keys in tally
        def get_from_nested_dict(dictio: dict, path: list[str]):
            for sp in path[:-1]:
                dictio = dictio[sp]
            return dictio.get(path[-1])

        def get_metric_paths(tally: dict):
            paths = []

            def traverse_dict(tally, current_path=[]):
                for key, value in tally.items():
                    new_path = current_path + [key]
                    if isinstance(value, dict):
                        traverse_dict(value, new_path)
                    else:
                        paths.append(new_path)

            traverse_dict(tally)
            return paths

        paths = get_metric_paths(tally)
        modified_tally = {}
        for p in paths:
            val = get_from_nested_dict(tally, p)
            modified_tally["_".join(p)] = np.mean(val)
        del tally
        tally = modified_tally
    for key, value in tally.items():
        stat_pack.add_stat(key, value)
    return stat_pack
