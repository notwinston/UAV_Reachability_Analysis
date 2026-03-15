"""Vertical sub-game HJ reachability solver.

Solves the vertical reach-avoid game (Phi_z) and the maximum distance
value function (V_z_inf) for computing the invariant capture set B_z.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

from reach_avoid_game.config import GameConfig
from reach_avoid_game.dynamics.vertical_game import VerticalGameDynamics
from reach_avoid_game.dynamics.vertical_relative import VerticalRelativeDynamics
from reach_avoid_game.solvers.grid_utils import create_vertical_game_grid, create_vertical_relative_grid
from reach_avoid_game.solvers.value_function_io import ValueFunctionData, save_value_function


def _make_capture_set_3d(grid: hj.Grid, d_z: float) -> jnp.ndarray:
    """Create the capture set SDF for the 3D vertical game.

    Capture condition: |z_D - z_A| <= d_z
    SDF convention: negative outside target, positive inside
    (for reach game: attacker winning region is where Phi <= 0)

    Actually for hj_reachability, the convention is:
    - Target set L = {x : l(x) <= 0} where l is the initial value function
    - The solver computes the backward reachable set

    So we define: l(x) = |z_D - z_A| - d_z
    l <= 0 means inside capture set
    """
    # grid.states has shape (n_z_d, n_v_dz, n_z_a, 3)
    z_d = grid.states[..., 0]
    z_a = grid.states[..., 2]

    return jnp.abs(z_d - z_a) - d_z


def solve_vertical_reach_avoid(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve the vertical reach-avoid game to get Phi_z.

    The defender maximizes (tries to capture) and the attacker minimizes
    (tries to escape). The value function Phi_z <= 0 defines the set of
    states from which the attacker can escape capture.

    Args:
        config: Game configuration
        preset: Grid preset (unused, config already has preset applied)
        output_dir: Directory to save value function

    Returns:
        Path to saved phi_z.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grid and dynamics
    grid = create_vertical_game_grid(config)
    dynamics = VerticalGameDynamics(config)

    # Create capture set (target for reach computation)
    initial_values = _make_capture_set_3d(grid, config.capture.d_z)

    # Solver settings — no value postprocessor needed for pure reach game
    solver_settings = hj.SolverSettings.with_accuracy(
        config.grid.solver.accuracy,
    )

    # Time array: solve backward from T=0 to T=-time_horizon
    T = config.grid.solver.time_horizon
    n_steps = config.grid.solver.time_steps
    times = jnp.linspace(0, -T, n_steps + 1)

    # Solve
    print(f"Solving vertical reach-avoid game (3D grid: {grid.states.shape[:-1]})...")
    all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values, progress_bar=True)

    # The final time slice is the converged value function
    phi_z = np.array(all_values[-1])

    # Save
    output_path = str(output_dir / "phi_z.npz")
    vf_data = ValueFunctionData(
        values=phi_z,
        grid_min=np.array(grid.domain.lo),
        grid_max=np.array(grid.domain.hi),
        grid_shape=phi_z.shape,
        params={
            "d_z": config.capture.d_z,
            "k_z": config.defender.k_z,
            "U_D_z": config.defender.max_speed_vertical,
            "U_A_z": config.attacker.max_speed_vertical,
            "time_horizon": float(T),
        },
        description="Vertical reach-avoid value function Phi_z",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved phi_z to {output_path}, shape: {phi_z.shape}")

    return output_path


def _make_distance_set_2d(grid: hj.Grid, d_z: float) -> jnp.ndarray:
    """Create initial value for maximum distance tracking (2D relative).

    For tracking, we want to compute the worst-case maximum |z_rel| over time.
    Initial value: l(z_rel, v_D_z) = |z_rel| - d_z
    l <= 0 means inside capture region (|z_rel| <= d_z).
    """
    z_rel = grid.states[..., 0]
    return jnp.abs(z_rel) - d_z


def solve_vertical_max_distance(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve for V_z_inf — the maximum distance value function.

    Uses 2D relative dynamics [z_rel, v_D_z].
    The defender minimizes (tracking) and attacker maximizes (escape).
    We solve backward and take the maximum over time (reach-avoid with no avoid set).

    The value function converges to V_z_inf as T -> infinity.

    Args:
        config: Game configuration
        preset: Grid preset
        output_dir: Directory to save value function

    Returns:
        Path to saved V_z_inf.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create 2D relative grid and dynamics
    grid = create_vertical_relative_grid(config)
    dynamics = VerticalRelativeDynamics(config)

    # Initial value: |z_rel| - d_z
    initial_values = _make_distance_set_2d(grid, config.capture.d_z)

    # For max distance tracking, no special postprocessor needed
    solver_settings = hj.SolverSettings.with_accuracy(
        config.grid.solver.accuracy,
    )

    # Solve for a longer time to approach convergence
    T = 20.0
    n_steps = 200
    times = jnp.linspace(0, -T, n_steps + 1)

    print(f"Solving vertical max distance (2D grid: {grid.states.shape[:-1]})...")
    all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values, progress_bar=True)

    # Take the final converged value function
    v_z_inf = np.array(all_values[-1])

    # Save
    output_path = str(output_dir / "V_z_inf.npz")
    vf_data = ValueFunctionData(
        values=v_z_inf,
        grid_min=np.array(grid.domain.lo),
        grid_max=np.array(grid.domain.hi),
        grid_shape=v_z_inf.shape,
        params={
            "d_z": config.capture.d_z,
            "k_z": config.defender.k_z,
            "U_D_z": config.defender.max_speed_vertical,
            "U_A_z": config.attacker.max_speed_vertical,
            "time_horizon": float(T),
        },
        description="Vertical maximum distance value function V_z_inf",
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
    This means: from these states, the defender can keep |z_rel| <= d_z
    forever, regardless of attacker's strategy.

    Since V_z_inf initial values are |z_rel| - d_z (negative inside d_z),
    B_z = {states where V_z_inf <= 0} (the zero sublevel set).

    Args:
        v_z_inf_path: Path to V_z_inf.npz
        d_z: Vertical capture distance
        output_dir: Directory to save B_z

    Returns:
        Path to saved B_z.npz
    """
    from reach_avoid_game.solvers.value_function_io import load_value_function

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf = load_value_function(v_z_inf_path)

    # B_z = {states where V_z_inf <= 0}
    # V_z_inf was initialized as |z_rel| - d_z, so <= 0 means inside capture region
    b_z_mask = (vf.values <= 0).astype(np.float64)

    output_path = str(output_dir / "B_z.npz")
    b_z_data = ValueFunctionData(
        values=b_z_mask,
        grid_min=vf.grid_min,
        grid_max=vf.grid_max,
        grid_shape=b_z_mask.shape,
        params={"d_z": d_z, "source": str(v_z_inf_path)},
        description="Vertical invariant capture set B_z (1.0 inside, 0.0 outside)",
    )
    save_value_function(output_path, b_z_data)
    print(f"Saved B_z to {output_path}, shape: {b_z_mask.shape}, "
          f"non-zero: {np.count_nonzero(b_z_mask)}/{b_z_mask.size}")

    return output_path
