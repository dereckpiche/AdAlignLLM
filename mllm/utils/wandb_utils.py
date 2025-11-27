"""
File: mllm/utils/wandb_utils.py
Summary: Shared Weights & Biases helper functions.
"""

import os
from typing import Any, Dict, Optional

_WANDB_AVAILABLE = False
_WANDB_RUN = None


def _try_import_wandb():
    global _WANDB_AVAILABLE
    if _WANDB_AVAILABLE:
        return True
    try:
        import wandb  # type: ignore

        _WANDB_AVAILABLE = True
        return True
    except Exception:
        _WANDB_AVAILABLE = False
        return False


def _safe_get(cfg: Dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def is_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_safe_get(cfg, ["logging", "wandb", "enabled"], False))


def init(cfg: Dict[str, Any], run_dir: str, run_name: Optional[str] = None) -> None:
    """
    Initialize Weights & Biases if enabled in config. No-op if disabled or wandb not installed.
    """
    global _WANDB_RUN
    if not is_enabled(cfg):
        return
    if not _try_import_wandb():
        return

    import wandb  # type: ignore

    project = _safe_get(cfg, ["logging", "wandb", "project"], "llm-negotiation")
    entity = _safe_get(cfg, ["logging", "wandb", "entity"], None)
    mode = _safe_get(cfg, ["logging", "wandb", "mode"], "online")
    tags = _safe_get(cfg, ["logging", "wandb", "tags"], []) or []
    notes = _safe_get(cfg, ["logging", "wandb", "notes"], None)
    group = _safe_get(cfg, ["logging", "wandb", "group"], None)
    name = _safe_get(cfg, ["logging", "wandb", "name"], run_name)

    # Ensure files are written into the hydra run directory
    os.makedirs(run_dir, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", run_dir)

    # Convert cfg to plain types for W&B config; fallback to minimal dictionary
    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg_container = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except Exception:
        cfg_container = cfg

    _WANDB_RUN = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=name,
        group=group,
        tags=tags,
        notes=notes,
        config=cfg_container,
        dir=run_dir,
        reinit=True,
    )


def log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a flat dictionary of metrics to W&B if active."""
    if not _WANDB_AVAILABLE or _WANDB_RUN is None:
        return
    try:
        import wandb  # type: ignore

        wandb.log(metrics if step is None else dict(metrics, step=step))
    except Exception:
        pass


def _flatten(prefix: str, data: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v


def _summarize_value(value: Any) -> Dict[str, Any]:
    import numpy as np  # local import to avoid hard dependency during disabled mode

    if value is None:
        return {"none": 1}
    # Scalars
    if isinstance(value, (int, float)):
        return {"value": float(value)}
    # Lists or arrays
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return {"size": 0}
        return {
            "mean": float(np.nanmean(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "last": float(arr.reshape(-1)[-1]),
            "size": int(arr.size),
        }
    except Exception:
        # Fallback: string repr
        return {"text": str(value)}


def log_tally(
    array_tally: Dict[str, Any], prefix: str = "", step: Optional[int] = None
) -> None:
    """
    Flatten and summarize Tally.array_tally and log to WandB.
    Each leaf list/array is summarized with mean/min/max/last/size.
    """
    if not _WANDB_AVAILABLE or _WANDB_RUN is None:
        return
    summarized: Dict[str, Any] = {}

    def walk(node: Any, path: list[str]):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [k])
            return
        # node is a list of values accumulated over time
        key = ".".join([p for p in ([prefix] if prefix else []) + path])
        try:
            summary = _summarize_value(node)
            for sk, sv in summary.items():
                summarized[f"{key}.{sk}"] = sv
        except Exception:
            summarized[f"{key}.error"] = 1

    walk(array_tally, [])
    if summarized:
        log(summarized, step=step)


def log_flat_stats(
    stats: Dict[str, Any], prefix: str = "", step: Optional[int] = None
) -> None:
    if not _WANDB_AVAILABLE or _WANDB_RUN is None:
        return
    flat: Dict[str, Any] = {}
    _flatten(prefix, stats, flat)
    if flat:
        log(flat, step=step)
