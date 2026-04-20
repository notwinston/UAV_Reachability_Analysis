"""Value function save/load utilities using .npz format."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


@dataclass
class ValueFunctionData:
    """Container for a computed value function with metadata."""
    values: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    grid_shape: tuple[int, ...]
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def stable_config_hash(config: Any) -> str:
    """Return a stable short hash for a config-like object."""
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif isinstance(config, dict):
        payload = config
    else:
        payload = repr(config)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def standard_metadata(
    config: Any,
    *,
    artifact: str,
    grid_preset: str | None = None,
    paper_valid: bool = False,
    subset_valid: bool | None = None,
    source_artifact: str | None = None,
    calibration: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata common to generated value-function artifacts."""
    metadata: dict[str, Any] = {
        "artifact": artifact,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": stable_config_hash(config),
        "grid_preset": grid_preset or getattr(config, "grid_preset", None),
        "paper_valid": bool(paper_valid),
    }
    if subset_valid is not None:
        metadata["subset_valid"] = bool(subset_valid)
    if source_artifact is not None:
        metadata["source_artifact"] = source_artifact
    if calibration:
        metadata["calibration"] = calibration
    if extra:
        metadata.update(extra)
    return metadata


def is_paper_valid(vf_data: ValueFunctionData) -> bool:
    """Return whether a loaded artifact declares itself paper-valid."""
    params = vf_data.params if isinstance(vf_data.params, dict) else {}
    return bool(params.get("paper_valid", False))


def save_value_function(path: str | Path, data: ValueFunctionData) -> None:
    """Save a value function to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        values=data.values,
        grid_min=np.asarray(data.grid_min),
        grid_max=np.asarray(data.grid_max),
        grid_shape=np.asarray(data.grid_shape),
        params=np.array(data.params, dtype=object),
        description=np.array(data.description),
    )


def load_value_function(path: str | Path) -> ValueFunctionData:
    """Load a value function from a .npz file."""
    path = Path(path)
    # Some checked-in value functions were saved with NumPy 2.x, whose object
    # pickle paths use numpy._core. The OptimizedDP env uses NumPy 1.x.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with np.load(path, allow_pickle=True) as npz:
        return ValueFunctionData(
            values=npz["values"],
            grid_min=npz["grid_min"],
            grid_max=npz["grid_max"],
            grid_shape=tuple(npz["grid_shape"]),
            params=npz["params"].item() if npz["params"].ndim == 0 else dict(npz["params"]),
            description=str(npz["description"]),
        )


def save_time_slices(path: str | Path, all_values: np.ndarray, times: np.ndarray) -> None:
    """Save all time slices of a value function solve.

    Args:
        path: Output .npz file path
        all_values: Array of shape (n_times, *grid_shape) with value at each time
        times: Array of shape (n_times,) with corresponding times
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, all_values=all_values, times=times)


def load_time_slices(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time slices of a value function solve.

    Args:
        path: Path to .npz file saved by save_time_slices

    Returns:
        Tuple of (all_values, times) arrays
    """
    path = Path(path)
    with np.load(path) as npz:
        return npz["all_values"], npz["times"]
