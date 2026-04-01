"""Visualization of value functions as 2D contour plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reach_avoid_game.solvers.value_function_io import ValueFunctionData


def plot_value_function_2d(
    vf_data: ValueFunctionData,
    slice_dims: Sequence[int] | None = None,
    slice_values: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot a 2D contour of a value function.

    For N-D value functions, specify which dimensions to slice and at what values.
    The remaining 2 dimensions become the x and y axes of the contour plot.

    Args:
        vf_data: Value function data
        slice_dims: Indices of dimensions to fix (len = ndim - 2)
        slice_values: Values at which to fix those dimensions
        ax: Optional matplotlib Axes to plot on
        save_path: Optional path to save the figure
        title: Optional title for the plot

    Returns:
        The matplotlib Axes object
    """
    values = vf_data.values
    ndim = values.ndim

    if ndim == 2:
        plot_data = values
        dim_x, dim_y = 0, 1
    elif ndim > 2:
        if slice_dims is None or slice_values is None:
            # Default: fix all dims except first two at their midpoints
            slice_dims = list(range(2, ndim))
            slice_values = []
            for d in slice_dims:
                mid = (float(vf_data.grid_min[d]) + float(vf_data.grid_max[d])) / 2
                slice_values.append(mid)

        # Determine which dims are free (the plot axes)
        all_dims = set(range(ndim))
        sliced = set(slice_dims)
        free_dims = sorted(all_dims - sliced)
        assert len(free_dims) == 2, f"Expected 2 free dimensions, got {len(free_dims)}"
        dim_x, dim_y = free_dims

        # Build slice indices
        idx = [slice(None)] * ndim
        for d, v in zip(slice_dims, slice_values):
            axis_vals = np.linspace(
                float(vf_data.grid_min[d]), float(vf_data.grid_max[d]),
                values.shape[d],
            )
            closest_idx = int(np.argmin(np.abs(axis_vals - v)))
            idx[d] = closest_idx

        plot_data = values[tuple(idx)]
    else:
        raise ValueError(f"Value function must be at least 2D, got {ndim}D")

    # Create axes for the 2 free dimensions
    x_vals = np.linspace(
        float(vf_data.grid_min[dim_x]), float(vf_data.grid_max[dim_x]),
        values.shape[dim_x],
    )
    y_vals = np.linspace(
        float(vf_data.grid_min[dim_y]), float(vf_data.grid_max[dim_y]),
        values.shape[dim_y],
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    contour = ax.contourf(x_vals, y_vals, plot_data.T, levels=20, cmap="RdBu_r")
    ax.contour(x_vals, y_vals, plot_data.T, levels=[0.0], colors="black", linewidths=2)
    plt.colorbar(contour, ax=ax, label="Value")

    if title:
        ax.set_title(title)
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return ax


def plot_winning_regions(
    phi_data: ValueFunctionData,
    slice_dims: Sequence[int] | None = None,
    slice_values: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot winning regions W_D (blue) and W_A (red).

    W_D = {x : phi(x) <= 0} (defender wins)
    W_A = {x : phi(x) > 0}  (attacker wins)

    Args:
        phi_data: Value function data
        slice_dims: Indices of dimensions to fix
        slice_values: Values at which to fix those dimensions
        ax: Optional matplotlib Axes
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        The matplotlib Axes object
    """
    values = phi_data.values
    ndim = values.ndim

    if ndim == 2:
        plot_data = values
        dim_x, dim_y = 0, 1
    elif ndim > 2:
        if slice_dims is None or slice_values is None:
            slice_dims = list(range(2, ndim))
            slice_values = []
            for d in slice_dims:
                mid = (float(phi_data.grid_min[d]) + float(phi_data.grid_max[d])) / 2
                slice_values.append(mid)

        all_dims = set(range(ndim))
        sliced = set(slice_dims)
        free_dims = sorted(all_dims - sliced)
        assert len(free_dims) == 2
        dim_x, dim_y = free_dims

        idx = [slice(None)] * ndim
        for d, v in zip(slice_dims, slice_values):
            axis_vals = np.linspace(
                float(phi_data.grid_min[d]), float(phi_data.grid_max[d]),
                values.shape[d],
            )
            closest_idx = int(np.argmin(np.abs(axis_vals - v)))
            idx[d] = closest_idx

        plot_data = values[tuple(idx)]
    else:
        raise ValueError(f"Value function must be at least 2D, got {ndim}D")

    x_vals = np.linspace(
        float(phi_data.grid_min[dim_x]), float(phi_data.grid_max[dim_x]),
        values.shape[dim_x],
    )
    y_vals = np.linspace(
        float(phi_data.grid_min[dim_y]), float(phi_data.grid_max[dim_y]),
        values.shape[dim_y],
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # W_D in blue, W_A in red
    w_d = (plot_data <= 0).astype(float)
    w_a = (plot_data > 0).astype(float)

    from matplotlib.colors import ListedColormap
    cmap_wd = ListedColormap(["white", "steelblue"])
    cmap_wa = ListedColormap(["white", "salmon"])

    ax.contourf(x_vals, y_vals, w_a.T, levels=[0.5, 1.5], colors=["salmon"], alpha=0.5)
    ax.contourf(x_vals, y_vals, w_d.T, levels=[0.5, 1.5], colors=["steelblue"], alpha=0.5)
    ax.contour(x_vals, y_vals, plot_data.T, levels=[0.0], colors="black", linewidths=2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.5, label="W_D (defender wins)"),
        Patch(facecolor="salmon", alpha=0.5, label="W_A (attacker wins)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if title:
        ax.set_title(title)
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return ax
