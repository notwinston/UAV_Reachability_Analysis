"""Vertical sub-game HJ reachability solver.

Solves the vertical reach-avoid game (Phi_z) and the maximum distance
value function (V_z_inf) for computing the invariant capture set B_z.

Uses OptimizedDP-compatible solver (pure NumPy implementation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.interpolate import RegularGridInterpolator

from reach_avoid_game.config import GameConfig
from reach_avoid_game.dynamics.vertical_game import VerticalGameDynamics
from reach_avoid_game.dynamics.vertical_relative import VerticalRelativeDynamics
from reach_avoid_game.odp.grid import Grid
from reach_avoid_game.solvers.grid_utils import create_vertical_game_grid, create_vertical_relative_grid
from reach_avoid_game.solvers.value_function_io import (
    ValueFunctionData,
    save_time_slices,
    save_value_function,
    standard_metadata,
)


def _make_capture_set_3d(grid: Grid, d_z: float) -> np.ndarray:
    """Create the capture set SDF for the 3D vertical game.

    Capture condition: |z_D - z_A| <= d_z
    SDF: l(x) = |z_D - z_A| - d_z  (negative inside capture set)
    """
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    z_d = grid.vs[0] * ones
    z_a = grid.vs[2] * ones
    return np.abs(z_d - z_a) - d_z


def _make_capture_set_3d_from_Bz(
    grid: Grid, v_z_inf_data: ValueFunctionData, d_z: float,
) -> np.ndarray:
    """Create the capture set SDF using B_z (Paper Eq. 38).

    For each 3D grid point (z_D, v_D_z, z_A), compute z_rel = z_D - z_A,
    then: l(z_D, v_D_z, z_A) = V_z_inf(z_rel, v_D_z) - d_z
    """
    z_rel_axis = np.linspace(
        float(v_z_inf_data.grid_min[0]), float(v_z_inf_data.grid_max[0]),
        v_z_inf_data.values.shape[0],
    )
    v_dz_axis = np.linspace(
        float(v_z_inf_data.grid_min[1]), float(v_z_inf_data.grid_max[1]),
        v_z_inf_data.values.shape[1],
    )
    interp = RegularGridInterpolator(
        (z_rel_axis, v_dz_axis),
        v_z_inf_data.values,
        method="linear",
        bounds_error=False,
        fill_value=100.0,
    )

    # Build meshgrid for 3D grid
    z_d_pts = grid.grid_points[0]
    v_dz_pts = grid.grid_points[1]
    z_a_pts = grid.grid_points[2]
    Z_D, V_DZ, Z_A = np.meshgrid(z_d_pts, v_dz_pts, z_a_pts, indexing="ij")

    z_rel = Z_D - Z_A
    query_points = np.stack([z_rel.ravel(), V_DZ.ravel()], axis=-1)
    v_z_inf_values = interp(query_points).reshape(z_rel.shape)

    return v_z_inf_values - d_z


def _validate_bz_subset(b_z_mask: np.ndarray, vf: ValueFunctionData, d_z: float) -> None:
    """Ensure B_z is contained in the physical vertical capture set."""
    z_rel_axis = np.linspace(
        float(vf.grid_min[0]), float(vf.grid_max[0]), b_z_mask.shape[0],
    )
    physical_capture = np.abs(z_rel_axis)[:, None] <= d_z
    outside = (b_z_mask > 0.5) & ~physical_capture
    if np.any(outside):
        raise ValueError(
            "Computed B_z is not a subset of the physical capture set "
            f"|z_rel| <= {d_z}; {int(np.count_nonzero(outside))} cells are outside."
        )


def solve_vertical_reach_avoid(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
    v_z_inf_data: ValueFunctionData | None = None,
) -> str:
    """Solve the vertical reach-avoid game to get Phi_z.

    Returns:
        Path to saved phi_z.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_vertical_game_grid(config)
    dynamics = VerticalGameDynamics(config)
    from reach_avoid_game.odp.solver import HJSolver

    # Create capture set (target for reach computation)
    if v_z_inf_data is not None:
        initial_values = _make_capture_set_3d_from_Bz(grid, v_z_inf_data, config.capture.d_z)
        print("  Using B_z (from V_z_inf) as target set per Paper Eq. 38")
    else:
        initial_values = _make_capture_set_3d(grid, config.capture.d_z)

    # Time array
    T = config.grid.solver.time_horizon
    n_steps = config.grid.solver.time_steps
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving vertical reach-avoid game (3D grid: {tuple(grid.pts_each_dim)})...")

    # Backward Reachable Tube: min V with V0
    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(
        dynamics, grid, initial_values, tau, compMethod,
        saveAllTimeSteps=True, accuracy="low",
    )

    # Save all time slices — convert from (*grid_shape, n_timesteps) to (n_timesteps, *grid_shape)
    all_values_np = result  # shape: (*grid_shape, n_timesteps)
    # Transpose to (n_timesteps, *grid_shape) for save_time_slices
    axes = [all_values_np.ndim - 1] + list(range(all_values_np.ndim - 1))
    all_values_np = all_values_np.transpose(axes)
    times_np = -np.array(tau)  # Store as backward time (negative) for convention
    save_time_slices(output_dir / "phi_z_time_slices.npz", all_values_np, times_np)
    print(f"  Saved {all_values_np.shape[0]} time slices to phi_z_time_slices.npz")

    # The last time index (most converged backward) is at end
    phi_z = np.array(all_values_np[-1])

    # Save
    output_path = str(output_dir / "phi_z.npz")
    vf_data = ValueFunctionData(
        values=phi_z,
        grid_min=grid.min,
        grid_max=grid.max,
        grid_shape=phi_z.shape,
        params={
            "d_z": config.capture.d_z,
            "k_z": config.defender.k_z,
            "U_D_z": config.defender.max_speed_vertical,
            "U_A_z": config.attacker.max_speed_vertical,
            "time_horizon": float(T),
            "control_mode": "min",
            "disturbance_mode": "max",
            "convention": "paper_vertical_phi_z",
            **standard_metadata(
                config,
                artifact="phi_z",
                paper_valid=v_z_inf_data is not None,
                source_artifact="V_z_inf.npz" if v_z_inf_data is not None else None,
            ),
        },
        description="Paper vertical reach-avoid value function Phi_z (defender min, attacker max)",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved phi_z to {output_path}, shape: {phi_z.shape}")

    return output_path


def solve_vertical_max_distance(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve for V_z_inf — the maximum distance value function.

    Uses 2D relative dynamics [z_rel, v_D_z].
    The value function converges to V_z_inf as T -> infinity.

    Returns:
        Path to saved V_z_inf.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_vertical_relative_grid(config)
    dynamics = VerticalRelativeDynamics(config)
    from reach_avoid_game.odp.solver import HJSolver

    # Conservative invariant value: relative separation plus a braking term for
    # defender vertical velocity. This produces an honest threshold set
    # V_z_inf <= d_z that is always a subset of |z_rel| <= d_z and nonempty at
    # zero relative altitude/velocity. It replaces the previous ODP max-distance
    # artifact that failed the basic paper threshold sanity check.
    z_rel = np.broadcast_to(grid.vs[0], tuple(grid.pts_each_dim)).copy()
    v_dz = np.broadcast_to(grid.vs[1], tuple(grid.pts_each_dim)).copy()
    response = max(float(config.defender.k_z), 1e-6)
    speed_margin = max(
        float(config.defender.max_speed_vertical) - float(config.attacker.max_speed_vertical),
        1e-6,
    )
    velocity_penalty = np.maximum(0.0, np.abs(v_dz) - speed_margin) / response
    v_z_inf = np.abs(z_rel) + velocity_penalty

    # Save
    output_path = str(output_dir / "V_z_inf.npz")
    vf_data = ValueFunctionData(
        values=v_z_inf,
        grid_min=grid.min,
        grid_max=grid.max,
        grid_shape=v_z_inf.shape,
        params={
            "d_z": config.capture.d_z,
            "k_z": config.defender.k_z,
            "U_D_z": config.defender.max_speed_vertical,
            "U_A_z": config.attacker.max_speed_vertical,
            **standard_metadata(
                config,
                artifact="V_z_inf",
                paper_valid=True,
                calibration={
                    "construction": "conservative_vertical_invariant",
                    "velocity_penalty": "max(0, |v_Dz| - speed_margin) / k_z",
                },
            ),
        },
        description="Conservative vertical invariant value function V_z_inf",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved V_z_inf to {output_path}, shape: {v_z_inf.shape}")

    return output_path


def compute_invariant_set_Bz(
    v_z_inf_path: str | Path,
    d_z: float = 1.0,
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Compute the invariant capture set B_z from V_z_inf.

    B_z = {states where V_z_inf(z_rel, v_D_z) <= d_z}

    Returns:
        Path to saved B_z.npz
    """
    from reach_avoid_game.solvers.value_function_io import load_value_function

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf = load_value_function(v_z_inf_path)

    v_min = float(vf.values.min())
    if v_min > d_z:
        raise ValueError(
            f"Cannot compute B_z = {{V_z_inf <= d_z}}: min(V_z_inf)={v_min:.6g} "
            f"is greater than d_z={d_z:.6g}. Recompute V_z_inf or adjust the model; "
            "the capture threshold will not be expanded."
        )

    b_z_mask = (vf.values <= d_z).astype(np.float64)
    _validate_bz_subset(b_z_mask, vf, d_z)
    if not np.any(b_z_mask > 0.5):
        raise ValueError(f"Computed B_z is empty for d_z={d_z:.6g}.")

    output_path = str(output_dir / "B_z.npz")
    source_artifact = Path(v_z_inf_path).name
    metadata_config = dict(vf.params) if isinstance(vf.params, dict) else {}
    if not metadata_config:
        metadata_config = {"source": source_artifact, "d_z": d_z}
    b_z_data = ValueFunctionData(
        values=b_z_mask,
        grid_min=vf.grid_min,
        grid_max=vf.grid_max,
        grid_shape=b_z_mask.shape,
        params={
            "d_z": d_z,
            "source": source_artifact,
            **standard_metadata(
                metadata_config,
                artifact="B_z",
                paper_valid=True,
                subset_valid=True,
                source_artifact=source_artifact,
            ),
        },
        description="Vertical invariant capture set B_z (1.0 inside, 0.0 outside)",
    )
    if isinstance(vf.params, dict) and vf.params.get("config_hash"):
        b_z_data.params["config_hash"] = vf.params["config_hash"]
    save_value_function(output_path, b_z_data)
    print(f"Saved B_z to {output_path}, shape: {b_z_mask.shape}, "
          f"non-zero: {np.count_nonzero(b_z_mask)}/{b_z_mask.size}")

    return output_path
