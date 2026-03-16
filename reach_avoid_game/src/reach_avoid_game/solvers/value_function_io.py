"""Value function save/load utilities using .npz format."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
