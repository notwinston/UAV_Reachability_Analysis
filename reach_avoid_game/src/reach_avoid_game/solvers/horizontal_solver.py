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


def _make_horizontal_capture_sdf(grid: Grid, d_h: float) -> np.ndarray:
    """Create capture set SDF for the 6D horizontal game.

    Capture: sqrt((x_A - x_D)^2 + (y_A - y_D)^2) <= d_h
    SDF: negative inside capture, positive outside.
    """
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones
    dist = np.sqrt((x_a - x_d)**2 + (y_a - y_d)**2)
    return dist - d_h


def _make_attacker_target_sdf(grid: Grid, config: GameConfig) -> np.ndarray:
    """Create target region SDF for the attacker position in 6D grid.

    Paper Eq. 20a: R_h includes {x_A^h in T}.
    Uses grid indices 4 (x_A) and 5 (y_A).
    SDF: negative inside target T, positive outside.
    """
    tr = config.target_region
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones

    # Box SDF: negative inside, positive outside
    target_sdf = np.maximum(
        np.maximum(tr.x_min - x_a, x_a - tr.x_max),
        np.maximum(tr.y_min - y_a, y_a - tr.y_max),
    )
    return target_sdf


def _make_reach_set(grid: Grid, config: GameConfig) -> np.ndarray:
    """Create the reach set l(x) for the horizontal game (Paper Eq. 20a).

    R_h = {x^h | x_A^h in T} union {x^h | p_D^h in Omega_obs}

    l(x) = min(l_T(x_A), l_D_obs(p_D))
    Negative when attacker is in target OR defender is in obstacles.
    """
    # l_T: attacker in target region (negative inside T)
    l_T = _make_attacker_target_sdf(grid, config)

    # l_D_obs: defender in obstacles (negative inside obstacles/walls)
    l_D_obs = _make_obstacle_avoid_set(grid, config)

    # Union = min of signed distances
    return np.minimum(l_T, l_D_obs)


def _make_avoid_set(grid: Grid, config: GameConfig, d_h: float) -> np.ndarray:
    """Create the avoid set constraint for the horizontal game (Paper Eq. 20b, Eq. 7).

    A_h = {x^h | ||x_A^h - x_D^h|| <= d_h AND x_A^h not in T}
          union {x^h | x_A^h in Omega_obs}

    Returns constraint function that is POSITIVE in the avoid region.
    This is passed directly as the 'obs' argument to the reach-avoid solver.

    Construction (Paper Eq. 7 adapted to horizontal):
      g_capture = ||x_A - x_D|| - d_h   (negative when captured)
      g_inT = -S(x_A, T)                (positive when IN target T)
      g_A_obs = S(x_A, Omega_obs)        (negative when in obstacle)

      obs = -min(max(g_capture, g_inT), g_A_obs)
      Positive when in avoid set A_h.
    """
    shape = tuple(grid.pts_each_dim)
    ones = np.ones(shape)

    # g_capture: negative when horizontally captured
    x_d = grid.vs[0] * ones
    y_d = grid.vs[1] * ones
    x_a = grid.vs[4] * ones
    y_a = grid.vs[5] * ones
    dist = np.sqrt((x_a - x_d)**2 + (y_a - y_d)**2)
    g_capture = dist - d_h

    # g_inT: positive when attacker IS in target T
    # S(x_A, T) is negative inside T, so -S is positive inside T
    attacker_target_sdf = _make_attacker_target_sdf(grid, config)
    g_inT = -attacker_target_sdf

    # g_A_obs: attacker obstacle SDF (negative inside obstacle)
    g_A_obs = _make_attacker_obstacle_avoid_set(grid, config)

    # Paper Eq. 7: obs = -min(max(g_capture, g_inT), g_A_obs)
    inner_max = np.maximum(g_capture, g_inT)
    combined = np.minimum(inner_max, g_A_obs)
    return -combined


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


def _make_avoid_set_with_Bh(
    grid: Grid, config: GameConfig, v_h_t_data: ValueFunctionData, d_h: float,
) -> np.ndarray:
    """Create avoid set using B_h instead of simple capture (Paper Eq. 39).

    A_h = {x^h | x_h^rel in B_h AND x_A^h not in T} union {x_A in Omega_obs}

    Returns constraint positive in avoid region, same convention as _make_avoid_set.
    Uses V_h_T <= d_h as the B_h membership criterion instead of ||x_A-x_D|| <= d_h.
    """
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
        fill_value=100.0,
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

    # g_Bh_capture: negative when inside B_h (V_h_T <= d_h_effective)
    # This replaces g_capture = ||x_A-x_D|| - d_h with the invariant set condition
    g_bh_capture = v_h_t_values - d_h_effective

    # g_inT: positive when attacker IS in target T
    attacker_target_sdf = _make_attacker_target_sdf(grid, config)
    g_inT = -attacker_target_sdf

    # g_A_obs: attacker obstacle SDF (negative inside obstacle)
    g_A_obs = _make_attacker_obstacle_avoid_set(grid, config)

    # Paper Eq. 7 adapted with B_h: obs = -min(max(g_bh_capture, g_inT), g_A_obs)
    inner_max = np.maximum(g_bh_capture, g_inT)
    combined = np.minimum(inner_max, g_A_obs)
    return -combined


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

    # Paper Eq. 20a: Reach set (target for BRT)
    # R_h = {x_A in T} union {p_D in Omega_obs}
    # l(x) = min(S(x_A, T), S(p_D, Omega_obs))
    reach_set_values = _make_reach_set(grid, config)

    # Paper Eq. 20b/Eq. 7: Avoid set (constraint)
    # A_h = {capture outside T} union {x_A in Omega_obs}
    # Positive in avoid region
    if v_h_t_data is not None:
        avoid_set_values = _make_avoid_set_with_Bh(grid, config, v_h_t_data, config.capture.d_h)
        print("  Using B_h feedback in avoid set per Paper Eq. 39")
    else:
        avoid_set_values = _make_avoid_set(grid, config, config.capture.d_h)
        print("  Using standard avoid set per Paper Eq. 20b")

    # Time horizon
    T = 10.0 if preset == "dev" else 22.0
    n_steps = 50 if preset == "dev" else 100
    small_number = 1e-5
    tau = np.arange(start=0, stop=T + small_number, step=T / n_steps)

    print(f"Solving horizontal reach-avoid game (6D grid: {tuple(grid.pts_each_dim)})...")
    print(f"  Time horizon: T={T}s, steps: {n_steps}")

    # Reach-avoid BRT: attacker minimizes (reach target), defender maximizes (prevent)
    # Paper Eq. 10: max(min{dPhi/dt + H, l - Phi}, g - Phi) = 0
    # Postprocessor: max(min(v, l), obs) where obs = avoid_set_values (positive in avoid)
    compMethod = {
        "TargetSetMode": "minVWithV0",
        "ObstacleSetMode": "maxVWithObstacle",
    }
    result = HJSolver(
        dynamics, grid, [reach_set_values, avoid_set_values],
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
            # Paper Eq. 22: W_A,h = {Phi_h <= 0}, W_D,h = {Phi_h > 0}
        },
        description="Horizontal reach-avoid value function Phi_h (6D). Defender wins when Phi_h > 0.",
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

    # Wall + obstacle penalty: large cost when defender is inside walls or obstacles
    wall_obstacle_sdf = _make_obstacle_avoid_set(grid, config)
    wall_penalty = np.where(wall_obstacle_sdf < 0, 1000.0, 0.0)
    initial_values = initial_values + wall_penalty

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
