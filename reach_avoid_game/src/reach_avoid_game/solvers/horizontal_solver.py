"""Horizontal sub-game HJ reachability solver.

Solves the horizontal reach-avoid game (Phi_h), the maximum distance
value function (V_h_T) for computing the invariant capture set B_h,
and the attacker reaching value function for T_goal computation.

Uses OptimizedDP-compatible solver (pure NumPy implementation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.interpolate import RegularGridInterpolator

from reach_avoid_game.config import GameConfig
from reach_avoid_game.dynamics.horizontal_game import HorizontalGameDynamics, HorizontalGameTrackingDynamics
from reach_avoid_game.dynamics.horizontal_relative import HorizontalRelativeDynamics
from reach_avoid_game.dynamics.attacker_reaching import AttackerReachingDynamics
from reach_avoid_game.odp.grid import Grid
from reach_avoid_game.odp.solver import HJSolver
from reach_avoid_game.solvers.grid_utils import (
    create_horizontal_game_grid,
    create_horizontal_relative_grid,
    create_attacker_reaching_grid,
)
from reach_avoid_game.solvers.value_function_io import ValueFunctionData, save_value_function


def _make_horizontal_capture_set(grid: Grid, d_h: float) -> np.ndarray:
    """Create capture set SDF for the 6D horizontal game.

    Capture: sqrt((x_A - x_D)^2 + (y_A - y_D)^2) <= d_h
    SDF: l(x) = sqrt((x_A - x_D)^2 + (y_A - y_D)^2) - d_h
    """
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones
    dist = np.sqrt((x_a - x_d)**2 + (y_a - y_d)**2)
    return dist - d_h


def _make_obstacle_avoid_set(grid: Grid, config: GameConfig) -> np.ndarray | None:
    """Create obstacle avoid set for the 6D horizontal game.

    Obstacles constrain both defender (x_D, y_D) and attacker (x_A, y_A).
    SDF: negative inside obstacle, positive outside.
    """
    if not config.obstacles:
        return None

    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones

    combined = None
    for obs in config.obstacles:
        sdf_d = np.maximum(
            np.maximum(obs.x_min - x_d, x_d - obs.x_max),
            np.maximum(obs.y_min - y_d, y_d - obs.y_max),
        )
        sdf_a = np.maximum(
            np.maximum(obs.x_min - x_a, x_a - obs.x_max),
            np.maximum(obs.y_min - y_a, y_a - obs.y_max),
        )
        sdf = np.minimum(sdf_d, sdf_a)
        if combined is None:
            combined = sdf
        else:
            combined = np.minimum(combined, sdf)

    return combined


def _make_avoid_set_with_Bh(
    grid: Grid, config: GameConfig, v_h_t_data: ValueFunctionData, d_h: float,
) -> np.ndarray:
    """Create avoid set combining obstacles and complement(B_h) (Paper Eq. 39)."""
    ndim_vht = v_h_t_data.values.ndim
    axes = []
    for i in range(ndim_vht):
        axes.append(np.linspace(
            float(v_h_t_data.grid_min[i]), float(v_h_t_data.grid_max[i]),
            v_h_t_data.values.shape[i],
        ))
    interp = RegularGridInterpolator(
        tuple(axes),
        v_h_t_data.values,
        method="linear",
        bounds_error=False,
        fill_value=-100.0,
    )

    v_min = float(v_h_t_data.values.min())
    d_h_effective = max(d_h, v_min * 1.05)

    # Compute relative states from 6D grid
    pts = [grid.grid_points[i] for i in range(6)]
    X_D, Y_D, V_DX, V_DY, X_A, Y_A = np.meshgrid(*pts, indexing="ij")

    x_rel = X_D - X_A
    y_rel = Y_D - Y_A

    # Flatten, interpolate in batches, reshape
    shape = x_rel.shape
    n_total = x_rel.size
    batch_size = 10000

    v_h_t_values = np.zeros(n_total)
    x_rel_flat = x_rel.ravel()
    y_rel_flat = y_rel.ravel()
    v_dx_flat = V_DX.ravel()
    v_dy_flat = V_DY.ravel()

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        query = np.stack([
            x_rel_flat[start:end], y_rel_flat[start:end],
            v_dx_flat[start:end], v_dy_flat[start:end],
        ], axis=-1)
        v_h_t_values[start:end] = interp(query)

    v_h_t_values = v_h_t_values.reshape(shape)

    # l_Bh_complement = d_h_effective - V_h_T
    # Negative outside B_h (avoid), positive inside (safe)
    l_bh_complement = d_h_effective - v_h_t_values

    obstacle_values = _make_obstacle_avoid_set(grid, config)

    if obstacle_values is not None:
        combined = np.minimum(np.asarray(obstacle_values), l_bh_complement)
    else:
        combined = l_bh_complement

    return combined


def solve_horizontal_reach_avoid(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
    v_h_t_data: ValueFunctionData | None = None,
) -> str:
    """Solve the horizontal reach-avoid game to get Phi_h.

    6D grid: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]

    Returns:
        Path to saved phi_h.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_horizontal_game_grid(config)
    dynamics = HorizontalGameDynamics(config)

    # Target: capture set
    target_values = _make_horizontal_capture_set(grid, config.capture.d_h)

    # Avoid set
    if v_h_t_data is not None:
        obstacle_values = _make_avoid_set_with_Bh(grid, config, v_h_t_data, config.capture.d_h)
        print("  Using B_h feedback in avoid set per Paper Eq. 39")
    else:
        obstacle_values = _make_obstacle_avoid_set(grid, config)

    # Time horizon
    T = 10.0 if preset == "dev" else 22.0
    n_steps = 50 if preset == "dev" else 100
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving horizontal reach-avoid game (6D grid: {tuple(grid.pts_each_dim)})...")
    print(f"  Time horizon: T={T}s, steps: {n_steps}")

    if obstacle_values is not None:
        # Reach-avoid: use target and obstacle
        compMethod = {
            "TargetSetMode": "minVWithV0",
            "ObstacleSetMode": "maxVWithObstacle",
        }
        result = HJSolver(
            dynamics, grid, [target_values, -obstacle_values],
            tau, compMethod, accuracy="low",
        )
    else:
        compMethod = {"TargetSetMode": "minVWithV0"}
        result = HJSolver(dynamics, grid, target_values, tau, compMethod, accuracy="low")

    phi_h = result

    output_path = str(output_dir / "phi_h.npz")
    vf_data = ValueFunctionData(
        values=phi_h,
        grid_min=grid.min,
        grid_max=grid.max,
        grid_shape=phi_h.shape,
        params={
            "d_h": config.capture.d_h,
            "k_x": config.defender.k_x,
            "k_y": config.defender.k_y,
            "U_D_h": config.defender.max_speed_horizontal,
            "U_A_h": config.attacker.max_speed_horizontal,
            "time_horizon": float(T),
        },
        description="Horizontal reach-avoid value function Phi_h (6D)",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved phi_h to {output_path}, shape: {phi_h.shape}")

    return output_path


def solve_horizontal_max_distance(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve for V_h_T — the maximum horizontal distance value function.

    Uses 4D relative dynamics [x_rel, y_rel, v_D_x, v_D_y].
    Paper: V_h_T may not converge; use T=2.5s per Section V.

    Returns:
        Path to saved V_h_T.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_horizontal_relative_grid(config)
    dynamics = HorizontalRelativeDynamics(config)

    # Initial value: horizontal distance = sqrt(x_rel^2 + y_rel^2)
    # grid.vs are broadcast-shaped; compute over full grid via broadcasting
    shape = tuple(grid.pts_each_dim)
    x_rel_sq = np.broadcast_to(grid.vs[0]**2, shape)
    y_rel_sq = np.broadcast_to(grid.vs[1]**2, shape)
    initial_values = np.sqrt(x_rel_sq + y_rel_sq).copy()

    # Per paper Section V, use T=2.5s
    T = 2.5
    n_steps = 50
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving horizontal max distance (4D grid: {tuple(grid.pts_each_dim)})...")

    # Max-over-time: max V with V0
    compMethod = {"TargetSetMode": "maxVWithV0"}
    result = HJSolver(dynamics, grid, initial_values, tau, compMethod, accuracy="low")

    v_h_t = result

    output_path = str(output_dir / "V_h_T.npz")
    vf_data = ValueFunctionData(
        values=v_h_t,
        grid_min=grid.min,
        grid_max=grid.max,
        grid_shape=v_h_t.shape,
        params={
            "d_h": config.capture.d_h,
            "k_x": config.defender.k_x,
            "k_y": config.defender.k_y,
            "U_D_h": config.defender.max_speed_horizontal,
            "U_A_h": config.attacker.max_speed_horizontal,
            "time_horizon": float(T),
        },
        description="Horizontal maximum distance value function V_h_T (4D)",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved V_h_T to {output_path}, shape: {v_h_t.shape}")

    return output_path


def compute_invariant_set_Bh(
    v_h_t_path: str | Path,
    d_h: float = 3.0,
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Compute the invariant capture set B_h from V_h_T.

    B_h = {states where V_h_T <= d_h}

    Returns:
        Path to saved B_h.npz
    """
    from reach_avoid_game.solvers.value_function_io import load_value_function

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf = load_value_function(v_h_t_path)

    v_min = float(vf.values.min())
    d_h_effective = max(d_h, v_min * 1.05)
    b_h_mask = (vf.values <= d_h_effective).astype(np.float64)
    if d_h_effective > d_h:
        print(f"  Note: V_h_T min ({v_min:.3f}) > d_h ({d_h:.3f}), "
              f"using effective threshold {d_h_effective:.3f} for B_h")

    output_path = str(output_dir / "B_h.npz")
    b_h_data = ValueFunctionData(
        values=b_h_mask,
        grid_min=vf.grid_min,
        grid_max=vf.grid_max,
        grid_shape=b_h_mask.shape,
        params={"d_h": d_h, "d_h_effective": d_h_effective, "source": str(v_h_t_path)},
        description="Horizontal invariant capture set B_h (1.0 inside, 0.0 outside)",
    )
    save_value_function(output_path, b_h_data)
    print(f"Saved B_h to {output_path}, shape: {b_h_mask.shape}, "
          f"non-zero: {np.count_nonzero(b_h_mask)}/{b_h_mask.size}")

    return output_path


def solve_attacker_reaching(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve attacker reaching value function for T_goal computation.

    2D grid: [x_A, y_A]

    Returns:
        Path to saved phi_A_reach.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_attacker_reaching_grid(config)
    dynamics = AttackerReachingDynamics(config)

    # Target: box region from config
    tr = config.target_region
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_a = grid.vs[0] * ones
    y_a = grid.vs[1] * ones

    # SDF for box: negative inside, positive outside
    target_sdf = np.maximum(
        np.maximum(tr.x_min - x_a, x_a - tr.x_max),
        np.maximum(tr.y_min - y_a, y_a - tr.y_max),
    )

    # Obstacle avoid set for attacker
    obstacle_values = None
    if config.obstacles:
        combined = None
        for obs in config.obstacles:
            sdf = np.maximum(
                np.maximum(obs.x_min - x_a, x_a - obs.x_max),
                np.maximum(obs.y_min - y_a, y_a - obs.y_max),
            )
            if combined is None:
                combined = sdf
            else:
                combined = np.minimum(combined, sdf)
        obstacle_values = combined

    T = 10.0 if preset == "dev" else 22.0
    n_steps = 50
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving attacker reaching (2D grid: {tuple(grid.pts_each_dim)})...")

    if obstacle_values is not None:
        compMethod = {
            "TargetSetMode": "minVWithV0",
            "ObstacleSetMode": "maxVWithObstacle",
        }
        result = HJSolver(
            dynamics, grid, [target_sdf, -obstacle_values],
            tau, compMethod, accuracy="low",
        )
    else:
        compMethod = {"TargetSetMode": "minVWithV0"}
        result = HJSolver(dynamics, grid, target_sdf, tau, compMethod, accuracy="low")

    phi_a = result

    output_path = str(output_dir / "phi_A_reach.npz")
    vf_data = ValueFunctionData(
        values=phi_a,
        grid_min=grid.min,
        grid_max=grid.max,
        grid_shape=phi_a.shape,
        params={
            "U_A_h": config.attacker.max_speed_horizontal,
            "time_horizon": float(T),
            "target_x_min": tr.x_min,
            "target_x_max": tr.x_max,
            "target_y_min": tr.y_min,
            "target_y_max": tr.y_max,
        },
        description="Attacker reaching value function phi_A_reach (2D)",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved phi_A_reach to {output_path}, shape: {phi_a.shape}")

    return output_path


def solve_horizontal_max_distance_6d(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve for V_h_T in 6D absolute coordinates with obstacle penalties.

    Returns:
        Path to saved V_h_T_6d.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_horizontal_game_grid(config)
    dynamics = HorizontalGameTrackingDynamics(config)

    # Initial values: horizontal distance (broadcast grid.vs to full shape)
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones
    initial_values = np.sqrt((x_d - x_a)**2 + (y_d - y_a)**2)

    # Obstacle penalty
    if config.obstacles:
        obstacle_sdf = _make_obstacle_avoid_set(grid, config)
        if obstacle_sdf is not None:
            penalty = np.where(obstacle_sdf < 0, 1000.0, 0.0)
            initial_values = initial_values + penalty

    T = 2.5
    n_steps = 50
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving horizontal max distance 6D (grid: {tuple(grid.pts_each_dim)})...")

    compMethod = {"TargetSetMode": "maxVWithV0"}
    result = HJSolver(dynamics, grid, initial_values, tau, compMethod, accuracy="low")

    v_h_t_6d = result

    output_path = str(output_dir / "V_h_T_6d.npz")
    vf_data = ValueFunctionData(
        values=v_h_t_6d,
        grid_min=grid.min,
        grid_max=grid.max,
        grid_shape=v_h_t_6d.shape,
        params={
            "d_h": config.capture.d_h,
            "k_x": config.defender.k_x,
            "k_y": config.defender.k_y,
            "U_D_h": config.defender.max_speed_horizontal,
            "U_A_h": config.attacker.max_speed_horizontal,
            "time_horizon": float(T),
        },
        description="Horizontal maximum distance value function V_h_T (6D with obstacles)",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved V_h_T_6d to {output_path}, shape: {v_h_t_6d.shape}")

    return output_path
