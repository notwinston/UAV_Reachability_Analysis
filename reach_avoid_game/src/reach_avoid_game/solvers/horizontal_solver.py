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
from reach_avoid_game.solvers.grid_utils import (
    create_horizontal_game_grid,
    create_horizontal_relative_grid,
    create_attacker_reaching_grid,
)
from reach_avoid_game.solvers.value_function_io import (
    ValueFunctionData,
    save_value_function,
    standard_metadata,
)


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


def _make_attacker_target_set(grid: Grid, config: GameConfig) -> np.ndarray:
    """Create SDF for attacker being inside the horizontal target region."""
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones
    tr = config.target_region
    return np.maximum(
        np.maximum(tr.x_min - x_a, x_a - tr.x_max),
        np.maximum(tr.y_min - y_a, y_a - tr.y_max),
    )


def _make_wall_avoid_set(grid: Grid, config: GameConfig, margin: float = 0.5) -> np.ndarray:
    """Create arena wall avoid set SDF for defender position in 6D grid.

    SDF convention: negative inside avoid zone (wall + margin), positive outside (safe).
    A margin of 1.5m means the defender should stay at least 1.5m from walls.
    This is less than d_h (3.0m) so capture near walls is still possible.
    """
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones

    # SDF for each wall with margin (negative within margin distance of wall)
    wall_x_min = x_d - config.room.x_min - margin
    wall_x_max = config.room.x_max - x_d - margin
    wall_y_min = y_d - config.room.y_min - margin
    wall_y_max = config.room.y_max - y_d - margin

    # Combined: min of all distances (closest wall)
    wall_sdf = np.minimum(np.minimum(wall_x_min, wall_x_max),
                          np.minimum(wall_y_min, wall_y_max))
    return wall_sdf


def _make_attacker_wall_avoid_set(grid: Grid, config: GameConfig, margin: float = 0.5) -> np.ndarray:
    """Create arena wall avoid set SDF for attacker position in 6D grid.

    Per Paper Eq. 20b: A_h includes {x_A^h in Omega_obs}.
    Uses grid indices 4 (x_A) and 5 (y_A).
    """
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones

    wall_x_min = x_a - config.room.x_min - margin
    wall_x_max = config.room.x_max - x_a - margin
    wall_y_min = y_a - config.room.y_min - margin
    wall_y_max = config.room.y_max - y_a - margin

    wall_sdf = np.minimum(np.minimum(wall_x_min, wall_x_max),
                          np.minimum(wall_y_min, wall_y_max))
    return wall_sdf


def _make_attacker_obstacle_avoid_set(grid: Grid, config: GameConfig) -> np.ndarray:
    """Create combined obstacle + wall avoid set for attacker position in 6D grid.

    Per Paper Eq. 20b: A_h includes {x_A^h in Omega_obs}.
    """
    combined = _make_attacker_wall_avoid_set(grid, config)

    if config.obstacles:
        shape = tuple(grid.pts_each_dim)
        ones = np.ones(shape)
        x_a = grid.vs[4] * ones
        y_a = grid.vs[5] * ones

        for obs in config.obstacles:
            sdf = np.maximum(
                np.maximum(obs.x_min - x_a, x_a - obs.x_max),
                np.maximum(obs.y_min - y_a, y_a - obs.y_max),
            )
            combined = np.minimum(combined, sdf)

    return combined


def _make_obstacle_avoid_set(grid: Grid, config: GameConfig) -> np.ndarray:
    """Create combined obstacle + wall avoid set for the 6D horizontal game.

    Always returns a value (walls always exist). SDF convention:
    negative inside obstacle/wall (avoid), positive outside (safe).
    """
    # Start with arena wall SDF
    combined = _make_wall_avoid_set(grid, config)

    # Add obstacle SDFs
    if config.obstacles:
        shape = tuple(grid.pts_each_dim)
        ones = np.ones(shape)
        x_d = grid.vs[0] * ones
        y_d = grid.vs[1] * ones

        for obs in config.obstacles:
            sdf = np.maximum(
                np.maximum(obs.x_min - x_d, x_d - obs.x_max),
                np.maximum(obs.y_min - y_d, y_d - obs.y_max),
            )
            combined = np.minimum(combined, sdf)

    return combined


def _interpolate_v_h_t_on_game_grid(
    grid: Grid, config: GameConfig, v_h_t_data: ValueFunctionData, d_h: float,
) -> np.ndarray:
    """Interpolate V_h_T from relative coordinates onto the 6D game grid."""
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
        fill_value=float(d_h) + 1.0,
    )

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
    return v_h_t_values


def _get_obstacle_aware_v_h_on_game_grid(grid: Grid, v_h_t_data: ValueFunctionData) -> np.ndarray:
    """Return an obstacle-aware 6D horizontal tracking value on the game grid."""
    expected_shape = tuple(grid.pts_each_dim)
    if v_h_t_data.values.ndim != 6:
        raise ValueError(
            "Paper horizontal reach-track-avoid requires an obstacle-aware 6D "
            "V_h_T value function over [x_D, y_D, v_Dx, v_Dy, x_A, y_A]. "
            f"Got {v_h_t_data.values.ndim}D data instead."
        )
    if tuple(v_h_t_data.values.shape) != expected_shape:
        raise ValueError(
            "V_h_T_6d grid shape does not match the horizontal game grid: "
            f"{v_h_t_data.values.shape} != {expected_shape}."
        )
    return v_h_t_data.values


def _make_paper_horizontal_reach_set(grid: Grid, config: GameConfig) -> np.ndarray:
    """Create paper R_h: attacker reaches target OR defender hits obstacle/wall."""
    attacker_target = _make_attacker_target_set(grid, config)
    defender_obstacle = _make_obstacle_avoid_set(grid, config)
    return np.minimum(attacker_target, defender_obstacle)


def _make_paper_horizontal_avoid_set(
    grid: Grid, config: GameConfig, v_h_t_data: ValueFunctionData, d_h: float,
) -> np.ndarray:
    """Create paper A_h: (B_h and attacker not in target) OR attacker obstacle."""
    v_h_t_values = _get_obstacle_aware_v_h_on_game_grid(grid, v_h_t_data)
    b_h_sdf = v_h_t_values - d_h

    target_sdf = _make_attacker_target_set(grid, config)
    attacker_not_target_sdf = -target_sdf
    capture_before_target = np.maximum(b_h_sdf, attacker_not_target_sdf)

    attacker_obstacles = _make_attacker_obstacle_avoid_set(grid, config)
    return np.minimum(capture_before_target, attacker_obstacles)


def _validate_bh_subset(b_h_mask: np.ndarray, vf: ValueFunctionData, d_h: float) -> None:
    """Ensure B_h is contained in the physical horizontal capture set."""
    if b_h_mask.ndim == 4:
        x_axis = np.linspace(float(vf.grid_min[0]), float(vf.grid_max[0]), b_h_mask.shape[0])
        y_axis = np.linspace(float(vf.grid_min[1]), float(vf.grid_max[1]), b_h_mask.shape[1])
        x_rel, y_rel = np.meshgrid(x_axis, y_axis, indexing="ij")
        physical_capture = np.sqrt(x_rel**2 + y_rel**2) <= d_h
        physical_capture = physical_capture[:, :, None, None]
    elif b_h_mask.ndim == 6:
        x_d_axis = np.linspace(float(vf.grid_min[0]), float(vf.grid_max[0]), b_h_mask.shape[0])
        y_d_axis = np.linspace(float(vf.grid_min[1]), float(vf.grid_max[1]), b_h_mask.shape[1])
        x_a_axis = np.linspace(float(vf.grid_min[4]), float(vf.grid_max[4]), b_h_mask.shape[4])
        y_a_axis = np.linspace(float(vf.grid_min[5]), float(vf.grid_max[5]), b_h_mask.shape[5])
        x_d, y_d, x_a, y_a = np.meshgrid(x_d_axis, y_d_axis, x_a_axis, y_a_axis, indexing="ij")
        physical_capture = np.sqrt((x_d - x_a)**2 + (y_d - y_a)**2) <= d_h
        physical_capture = physical_capture[:, :, None, None, :, :]
    else:
        raise ValueError(f"B_h validation expects 4D or 6D data, got {b_h_mask.ndim}D.")

    outside = (b_h_mask > 0.5) & ~physical_capture
    if np.any(outside):
        raise ValueError(
            "Computed B_h is not a subset of the physical capture set "
            f"horizontal distance <= {d_h}; {int(np.count_nonzero(outside))} cells are outside."
        )


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
    from reach_avoid_game.odp.solver import HJSolver

    if v_h_t_data is None:
        raise ValueError("Paper Phi_h requires obstacle-aware V_h_T_6d data to construct B_h in A_h")

    target_values = _make_paper_horizontal_reach_set(grid, config)
    obstacle_values = _make_paper_horizontal_avoid_set(
        grid, config, v_h_t_data, config.capture.d_h,
    )
    print("  Using paper R_h: attacker target OR defender obstacle")
    print("  Using paper A_h: (6D obstacle-aware B_h AND attacker not in target) OR attacker obstacle")

    # Time horizon
    T = 10.0 if preset == "dev" else 22.0
    n_steps = 50 if preset == "dev" else 100
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving horizontal reach-avoid game (6D grid: {tuple(grid.pts_each_dim)})...")
    print(f"  Time horizon: T={T}s, steps: {n_steps}")

    # Paper Phi_h: sub-zero means attacker-winning horizontal reach-avoid state.
    compMethod = {
        "TargetSetMode": "minVWithV0",
        "ObstacleSetMode": "maxVWithObstacle",
    }
    result = HJSolver(
        dynamics, grid, [target_values, obstacle_values],
        tau, compMethod, accuracy="low",
    )

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
            "convention": "paper_horizontal_phi_h",
            "attacker_wins": "phi_h <= 0",
            "defender_wins": "phi_h > 0",
            **standard_metadata(
                config,
                artifact="phi_h",
                paper_valid=True,
                source_artifact="V_h_T_6d.npz",
            ),
        },
        description="Paper horizontal reach-avoid value function Phi_h (6D): attacker wins <= 0",
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
    # Conservative diagnostic relative value. The paper-valid horizontal
    # invariant set uses the 6D obstacle-aware value below; this 4D artifact is
    # kept for plots and diagnostics only.
    shape = tuple(grid.pts_each_dim)
    x_rel_sq = np.broadcast_to(grid.vs[0]**2, shape)
    y_rel_sq = np.broadcast_to(grid.vs[1]**2, shape)
    v_dx = np.broadcast_to(grid.vs[2], shape)
    v_dy = np.broadcast_to(grid.vs[3], shape)
    speed_margin = max(
        float(config.defender.max_speed_horizontal) - float(config.attacker.max_speed_horizontal),
        1e-6,
    )
    k_min = max(min(float(config.defender.k_x), float(config.defender.k_y)), 1e-6)
    speed_excess = np.maximum(0.0, np.sqrt(v_dx**2 + v_dy**2) - speed_margin)
    v_h_t = np.sqrt(x_rel_sq + y_rel_sq).copy() + speed_excess / k_min

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
            **standard_metadata(
                config,
                artifact="V_h_T",
                paper_valid=False,
                calibration={"construction": "conservative_4d_diagnostic"},
            ),
        },
        description="Horizontal relative diagnostic value function V_h_T (4D)",
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
    if v_min > d_h:
        raise ValueError(
            f"Cannot compute B_h = {{V_h <= d_h}}: min(V_h)={v_min:.6g} "
            f"is greater than d_h={d_h:.6g}. Recompute V_h or adjust the model; "
            "the capture threshold will not be expanded."
        )

    b_h_mask = (vf.values <= d_h).astype(np.float64)
    _validate_bh_subset(b_h_mask, vf, d_h)
    if not np.any(b_h_mask > 0.5):
        raise ValueError(f"Computed B_h is empty for d_h={d_h:.6g}.")

    output_path = str(output_dir / "B_h.npz")
    source_artifact = Path(v_h_t_path).name
    metadata_config = dict(vf.params) if isinstance(vf.params, dict) else {}
    if not metadata_config:
        metadata_config = {"source": source_artifact, "d_h": d_h}
    b_h_data = ValueFunctionData(
        values=b_h_mask,
        grid_min=vf.grid_min,
        grid_max=vf.grid_max,
        grid_shape=b_h_mask.shape,
        params={
            "d_h": d_h,
            "source": source_artifact,
            **standard_metadata(
                metadata_config,
                artifact="B_h",
                paper_valid=True,
                subset_valid=True,
                source_artifact=source_artifact,
            ),
        },
        description="Horizontal invariant capture set B_h (1.0 inside, 0.0 outside)",
    )
    if isinstance(vf.params, dict) and vf.params.get("config_hash"):
        b_h_data.params["config_hash"] = vf.params["config_hash"]
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
    from reach_avoid_game.odp.solver import HJSolver

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
            dynamics, grid, [target_sdf, obstacle_values],
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
    # Conservative obstacle-aware invariant value. Inside the paper threshold
    # implies physical horizontal capture and defender/attacker obstacle safety.
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones
    v_dx = grid.vs[2] * ones
    v_dy = grid.vs[3] * ones
    distance = np.sqrt((x_d - x_a)**2 + (y_d - y_a)**2)
    speed_margin = max(
        float(config.defender.max_speed_horizontal) - float(config.attacker.max_speed_horizontal),
        1e-6,
    )
    k_min = max(min(float(config.defender.k_x), float(config.defender.k_y)), 1e-6)
    speed_excess = np.maximum(0.0, np.sqrt(v_dx**2 + v_dy**2) - speed_margin)
    v_h_t_6d = distance + speed_excess / k_min

    defender_obstacles = _make_obstacle_avoid_set(grid, config)
    attacker_obstacles = _make_attacker_obstacle_avoid_set(grid, config)
    # Add a finite geometric penalty only where a vehicle is already in an
    # obstacle/wall. This keeps valid same-position safe states small while
    # preventing obstacle cells from entering B_h.
    obstacle_penalty = float(config.capture.d_h) + 1.0
    v_h_t_6d = (
        v_h_t_6d
        + np.where(defender_obstacles < 0, obstacle_penalty, 0.0)
        + np.where(attacker_obstacles < 0, obstacle_penalty, 0.0)
    )

    print(f"Constructing horizontal max distance 6D (grid: {tuple(grid.pts_each_dim)})...")

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
            **standard_metadata(
                config,
                artifact="V_h_T_6d",
                paper_valid=True,
                calibration={
                    "construction": "conservative_obstacle_aware_horizontal_invariant",
                    "velocity_penalty": "max(0, speed - speed_margin) / min(k_x, k_y)",
                },
            ),
        },
        description="Conservative obstacle-aware horizontal invariant value function V_h_T_6d",
    )
    save_value_function(output_path, vf_data)
    print(f"Saved V_h_T_6d to {output_path}, shape: {v_h_t_6d.shape}")

    return output_path
