"""Visualization of game trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_2d(
    trajectory_data: dict[str, Any],
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot defender and attacker trajectories in 2D.

    Supports both vertical (z vs t) and horizontal (x vs y) trajectories.

    Args:
        trajectory_data: Dictionary with trajectory arrays.
            For vertical: z_d, z_a, dt keys
            For horizontal: x_d, y_d, x_a, y_a keys
        ax: Optional matplotlib Axes
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        The matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if "x_d" in trajectory_data and "y_d" in trajectory_data:
        # Horizontal 2D trajectory (x vs y)
        x_d = trajectory_data["x_d"]
        y_d = trajectory_data["y_d"]
        x_a = trajectory_data["x_a"]
        y_a = trajectory_data["y_a"]

        ax.plot(x_d, y_d, "b-", linewidth=2, label="Defender")
        ax.plot(x_a, y_a, "r-", linewidth=2, label="Attacker")
        ax.plot(x_d[0], y_d[0], "bs", markersize=10, label="Defender start")
        ax.plot(x_a[0], y_a[0], "rs", markersize=10, label="Attacker start")
        ax.plot(x_d[-1], y_d[-1], "bo", markersize=10)
        ax.plot(x_a[-1], y_a[-1], "ro", markersize=10)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

    elif "z_d" in trajectory_data and "z_a" in trajectory_data:
        # Vertical trajectory (z vs time)
        z_d = trajectory_data["z_d"]
        z_a = trajectory_data["z_a"]
        dt = trajectory_data.get("dt", 0.01)
        t = np.arange(len(z_d)) * dt

        ax.plot(t, z_d, "b-", linewidth=2, label="Defender z")
        ax.plot(t, z_a, "r-", linewidth=2, label="Attacker z")

        # Shade capture zone around attacker
        d_z = trajectory_data.get("d_z_threshold", 1.0)
        ax.fill_between(t, z_a - d_z, z_a + d_z, alpha=0.1, color="green",
                        label=f"Capture zone (d_z={d_z})")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Altitude (m)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return ax


def plot_game_state(
    trajectory: dict[str, Any],
    value_functions: dict[str, Any] | None = None,
    timestep: int = 0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot combined game state view at a given timestep.

    Shows trajectory, value function slice, and game info.

    Args:
        trajectory: Dictionary with trajectory data
        value_functions: Optional dict with value function data for overlay
        timestep: Which timestep to show
        save_path: Optional path to save figure

    Returns:
        The matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: trajectory
    plot_trajectory_2d(trajectory, ax=axes[0], title="Trajectory")

    # Right: game info
    ax_info = axes[1]
    ax_info.axis("off")

    info_lines = [f"Timestep: {timestep}"]

    if "z_d" in trajectory:
        dt = trajectory.get("dt", 0.01)
        t = timestep * dt
        info_lines.append(f"Time: {t:.2f}s")
        if timestep < len(trajectory["z_d"]):
            z_d = trajectory["z_d"][timestep]
            z_a = trajectory["z_a"][timestep]
            info_lines.append(f"z_D = {z_d:.2f}m")
            info_lines.append(f"z_A = {z_a:.2f}m")
            info_lines.append(f"|z_D - z_A| = {abs(z_d - z_a):.2f}m")

    if "captured" in trajectory:
        info_lines.append(f"Captured: {trajectory['captured']}")

    if "mode" in trajectory and timestep < len(trajectory["mode"]):
        mode_names = {0: "Reach", 1: "Track (boundary)", 2: "PID"}
        mode = trajectory["mode"][timestep]
        info_lines.append(f"Mode: {mode_names.get(mode, str(mode))}")

    info_text = "\n".join(info_lines)
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                 fontsize=12, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved game state plot to {save_path}")

    return fig
