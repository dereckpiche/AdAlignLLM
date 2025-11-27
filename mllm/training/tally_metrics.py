"""
File: mllm/training/tally_metrics.py
Summary: Transforms tally files into aggregated metric summaries.
"""

import os
from numbers import Number
from typing import Union

import wandb


class Tally:
    """
    Minimal scalar-first tally.
    - Keys are strings.
    - First add stores a scalar; subsequent adds upgrade to a list of scalars.
    """

    def __init__(self):
        self.stats = {}

    def reset(self):
        """Reset all recorded metrics back to an empty dictionary."""
        self.stats = {}

    def _coerce_scalar(self, value: Union[int, float]) -> Union[int, float]:
        """Ensure ``value`` is a plain Python scalar (detach tensors, etc.)."""
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, Number):
            return value
        raise AssertionError("Metric must be a scalar number")

    def add_metric(self, path: str, metric: Union[int, float]):
        """Accumulate a metric under ``path`` (scalar on first add, list thereafter)."""
        metric = float(metric)
        assert isinstance(path, str), "Path must be a string."
        assert isinstance(metric, float), "Metric must be a scalar number."

        scalar = self._coerce_scalar(metric)
        existing = self.stats.get(path)
        if existing is None:
            self.stats[path] = scalar
        elif isinstance(existing, list):
            existing.append(scalar)
        else:
            self.stats[path] = [existing, scalar]

    def save(self, identifier: str, folder: str):
        """Persist the tally as a pickle file under ``folder``."""
        os.makedirs(name=folder, exist_ok=True)
        try:
            import pickle

            pkl_path = os.path.join(folder, f"{identifier}.tally.pkl")
            payload = self.stats
            with open(pkl_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
