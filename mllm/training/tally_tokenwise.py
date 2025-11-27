"""
File: mllm/training/tally_tokenwise.py
Summary: Converts token-level tallies into per-token statistics.
"""

import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class ContextualizedTokenwiseTally:
    """
    Collect, store, and save token-level metrics per rollout.

    - One DataFrame per rollout_id in `paths`
    - Index = timestep (int)
    - Columns are added incrementally via `add_contexts()` and `add_data()`
    - Cells may contain scalars, strings, or lists (dtype=object)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        paths: List[str],
        max_context_length: int = 30,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer used to convert tids -> tokens
            paths: rollout identifiers (parallel to batch dimension)
            max_context_length: truncate context token lists to this length
        """
        self.tokenizer = tokenizer
        self.paths = paths
        self.max_context_length = max_context_length
        self.tally: Dict[str, pd.DataFrame] = {path: pd.DataFrame() for path in paths}

        # set later by setters
        self.contexts: torch.Tensor | None = None
        self.action_mask: torch.Tensor | None = None
        self.range: Tuple[int, int] | None = None

    # --------- Utilities ---------

    def tids_to_str(self, tids: List[int]) -> List[str]:
        """Convert a list of token IDs to a list of token strings."""
        return self.tokenizer.convert_ids_to_tokens(tids)

    def _ensure_ready(self):
        """Validate that action mask and range are configured prior to writes."""
        assert self.action_mask is not None, "call set_action_mask(mask) first"
        assert self.range is not None, "call set_range((start, end)) first"

    @staticmethod
    def _sanitize_filename(name: Any) -> str:
        """Make a safe filename from any rollout_id."""
        s = str(name)
        bad = {os.sep, " ", ":", "|", "<", ">", '"', "'"}
        if os.altsep is not None:
            bad.add(os.altsep)
        for ch in bad:
            s = s.replace(ch, "_")
        return s

    @staticmethod
    def _pad_left(seq: List[Any], length: int, pad_val: Any = "") -> List[Any]:
        """Left-pad a sequence to `length` with `pad_val`."""
        if len(seq) >= length:
            return seq[-length:]
        return [pad_val] * (length - len(seq)) + list(seq)

    # --------- Setters ---------

    def set_action_mask(self, action_mask: torch.Tensor):
        """Register the (B, S) mask indicating which tokens correspond to actions."""
        self.action_mask = action_mask

    def set_range(self, range: Tuple[int, int]):
        """Record which subset of ``paths`` the current mini-batch corresponds to."""
        self.range = range

    # --------- Column builders ---------

    def add_contexts(self, contexts: torch.Tensor):
        """
        Add a single 'context' column (list[str]) for valid steps.

        Expects `contexts` with shape (B, S): token id at each timestep.
        For each valid timestep t, we use the last N tokens up to and including t:
            window = contexts[i, max(0, t - N + 1) : t + 1]
        The list is left-padded with "" to always be length N.
        """
        self._ensure_ready()

        current_paths = self.paths[self.range[0] : self.range[1]]
        B, S = contexts.shape
        N = self.max_context_length

        # to CPU ints once
        contexts_cpu = contexts.detach().to("cpu")

        for i in range(B):
            rollout_id = current_paths[i]
            df = self.tally.get(rollout_id, pd.DataFrame())

            valid_idx = torch.nonzero(
                self.action_mask[i].bool(), as_tuple=False
            ).squeeze(-1)
            if valid_idx.numel() == 0:
                self.tally[rollout_id] = df
                continue

            idx_list = valid_idx.tolist()

            # ensure index contains valid steps
            if df.empty:
                df = pd.DataFrame(index=idx_list)
            else:
                new_index = sorted(set(df.index.tolist()) | set(idx_list))
                if list(df.index) != new_index:
                    df = df.reindex(new_index)

            # build context windows
            ctx_token_lists = []
            for t in idx_list:
                start = max(0, t - N + 1)
                window_ids = contexts_cpu[i, start : t + 1].tolist()
                window_toks = self.tids_to_str([int(x) for x in window_ids])
                if len(window_toks) < N:
                    window_toks = [""] * (N - len(window_toks)) + window_toks
                else:
                    window_toks = window_toks[-N:]
                ctx_token_lists.append(window_toks)

            # single 'context' column
            if "context" not in df.columns:
                df["context"] = pd.Series(index=df.index, dtype=object)
            df.loc[idx_list, "context"] = pd.Series(
                ctx_token_lists, index=idx_list, dtype=object
            )

            self.tally[rollout_id] = df

    def add_data(
        self,
        metric_id: str,
        metrics: torch.Tensor,
        to_tids: bool = False,
    ):
        """
        Add a metric column for valid steps.

        Args:
            metric_id: column name
            metrics: shape (B, S) for scalars/ids or (B, S, K) for top-k vectors
            to_tids: if True, treat ints/lists of ints as tids and convert to tokens
        """
        self._ensure_ready()
        current_paths = self.paths[self.range[0] : self.range[1]]

        if metrics.dim() == 2:
            B, S = metrics.shape
        elif metrics.dim() == 3:
            B, S, _ = metrics.shape
        else:
            raise ValueError("metrics must be (B, S) or (B, S, K)")

        for i in range(B):
            rollout_id = current_paths[i]
            df = self.tally.get(rollout_id, pd.DataFrame())

            valid_idx = torch.nonzero(
                self.action_mask[i].bool(), as_tuple=False
            ).squeeze(-1)
            if valid_idx.numel() == 0:
                self.tally[rollout_id] = df
                continue

            idx_list = valid_idx.detach().cpu().tolist()

            # Ensure index contains valid steps
            if df.empty:
                df = pd.DataFrame(index=idx_list)
            else:
                new_index = sorted(set(df.index.tolist()) | set(idx_list))
                if list(df.index) != new_index:
                    df = df.reindex(new_index)

            # Slice metrics at valid steps
            m_valid = metrics[i][valid_idx]

            # -> pure python lists (1D list or list-of-lists)
            values = m_valid.detach().cpu().tolist()

            # optional tids -> tokens
            if to_tids:

                def _to_tokish(x):
                    if isinstance(x, list):
                        return self.tids_to_str([int(v) for v in x])
                    else:
                        return self.tids_to_str([int(x)])[0]

                values = [_to_tokish(v) for v in values]

            # Ensure column exists with object dtype, then assign via aligned Series
            if metric_id not in df.columns:
                df[metric_id] = pd.Series(index=df.index, dtype=object)

            if isinstance(values, np.ndarray):
                values = values.tolist()

            if len(values) != len(idx_list):
                raise ValueError(
                    f"Length mismatch for '{metric_id}': values={len(values)} vs idx_list={len(idx_list)}"
                )

            df.loc[idx_list, metric_id] = pd.Series(
                values, index=idx_list, dtype=object
            )
            self.tally[rollout_id] = df

    # --------- Saving ---------

    def save(self, path: str):
        """
        Write a manifest JSON and one CSV per rollout.

        - Manifest includes metadata only (safe to JSON).
        - Each rollout CSV is written with index label 'timestep'.
        - Only a single 'context' column (list[str]).
        """
        if not self.tally or all(df.empty for df in self.tally.values()):
            return

        os.makedirs(path, exist_ok=True)
        from datetime import datetime

        now = datetime.now()

        manifest = {
            "created_at": f"{now:%Y-%m-%d %H:%M:%S}",
            "max_context_length": self.max_context_length,
            "num_rollouts": len(self.tally),
            "rollouts": [],
        }

        for rid, df in self.tally.items():
            rid_str = str(rid)
            safe_name = self._sanitize_filename(rid_str)
            csv_path = os.path.join(path, f"{safe_name}_tokenwise.csv")

            # Put 'context' first, then the rest
            cols = ["context"] + [c for c in df.columns if c != "context"]
            try:
                df[cols].to_csv(csv_path, index=True, index_label="timestep")
            except Exception as e:
                continue

            manifest["rollouts"].append(
                {
                    "rollout_id": rid_str,
                    "csv": csv_path,
                    "num_rows": int(df.shape[0]),
                    "columns": cols,
                }
            )

        manifest_path = os.path.join(
            path, f"tokenwise_manifest_{now:%Y-%m-%d___%H-%M-%S}.json"
        )
        with open(manifest_path, "w") as fp:
            json.dump(manifest, fp, indent=2)
