"""Horizontal sub-game HJ reachability solver.

Solves the horizontal reach-avoid game (Phi_h), the maximum distance
value function (V_h_T) for computing the invariant capture set B_h,
and the attacker reaching value function for T_goal computation.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

from reach_avoid_game.config import GameConfig
from reach_avoid_game.dynamics.horizontal_game import HorizontalGameDynamics
from reach_avoid_game.dynamics.horizontal_relative import HorizontalRelativeDynamics
from reach_avoid_game.dynamics.attacker_reaching import AttackerReachingDynamics
from reach_avoid_game.solvers.grid_utils import (
    create_horizontal_game_grid,
    create_horizontal_relative_grid,
    create_attacker_reaching_grid,
)
from reach_avoid_game.solvers.value_function_io import ValueFunctionData, save_value_function


def _make_horizontal_capture_set(grid: hj.Grid, d_h: float) -> jnp.ndarray:
    """Create capture set SDF for the 6D horizontal game.

    Capture condition: sqrt((x_A - x_D)^2 + (y_A - y_D)^2) <= d_h
    SDF: l(x) = sqrt((x_A - x_D)^2 + (y_A - y_D)^2) - d_h
    l <= 0 means inside capture set.
    """
    x_d = grid.states[..., 0]
    y_d = grid.states[..., 1]
    x_a = grid.states[..., 4]
    y_a = grid.states[..., 5]

    dist = jnp.sqrt((x_a - x_d)**2 + (y_a - y_d)**2)
    return dist - d_h


def _make_obstacle_avoid_set(grid: hj.Grid, config: GameConfig) -> jnp.ndarray:
    """Create obstacle avoid set for the 6D horizontal game.

    The defender must avoid obstacles. We use the defender's position (x_D, y_D).
    For a box obstacle: SDF is negative inside the obstacle.
    For avoid set: we want l_obs(x) <= 0 to represent states TO AVOID.

    For box obstacle [x_min, x_max] x [y_min, y_max]:
    SDF_box = max(x_min - x_D, x_D - x_max, y_min - y_D, y_D - y_max)
    Negative inside obstacle, positive outside.
    """
    if not config.obstacles:
        return None

    x_d = grid.states[..., 0]
    y_d = grid.states[..., 1]

    # Combine multiple obstacles: take minimum (union of obstacle regions)
    combined = None
    for obs in config.obstacles:
        sdf = jnp.maximum(
            jnp.maximum(obs.x_min - x_d, x_d - obs.x_max),
            jnp.maximum(obs.y_min - y_d, y_d - obs.y_max),
        )
        if combined is None:
            combined = sdf
        else:
            combined = jnp.minimum(combined, sdf)

    return combined


def solve_horizontal_reach_avoid(
    config: GameConfig,
    preset: str = "dev",
    output_dir: str | Path = "/workspace/data/value_functions",
) -> str:
    """Solve the horizontal reach-avoid game to get Phi_h.

    6D grid: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
    Target: horizontal capture set
    Avoid: obstacles (box SDFs in horizontal plane)

    Args:
        config: Game configuration
        preset: Grid preset
        output_dir: Directory to save value function

    Returns:
        Path to saved phi_h.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_horizontal_game_grid(config)
    dynamics = HorizontalGameDynamics(config)

    # Target: capture set
    target_values = _make_horizontal_capture_set(grid, config.capture.d_h)

    # Avoid: obstacles
    obstacle_values = _make_obstacle_avoid_set(grid, config)

    # Time horizon: dev=10s, paper=22s
    T = 10.0 if preset == "dev" else 22.0
    n_steps = 50 if preset == "dev" else 100

    if obstacle_values is not None:
        # Reach-avoid: reach target while avoiding obstacles
        # Use value postprocessor for reach-avoid computation
        # V(t) = max(l_target, min(V_prev, -l_obstacle))
        # where l_obstacle <= 0 means in obstacle (to avoid)
        def reach_avoid_postprocessor(t, v):
            return jnp.maximum(v, jnp.minimum(target_values, -obstacle_values))

        solver_settings = hj.SolverSettings.with_accuracy(
            config.grid.solver.accuracy,
            value_postprocessor=reach_avoid_postprocessor,
        )
    else:
        solver_settings = hj.SolverSettings.with_accuracy(
            config.grid.solver.accuracy,
        )

    times = jnp.linspace(0, -T, n_steps + 1)

    print(f"Solving horizontal reach-avoid game (6D grid: {grid.states.shape[:-1]})...")
    print(f"  Time horizon: T={T}s, steps: {n_steps}")
    all_values = hj.solve(solver_settings, dynamics, grid, times, target_values, progress_bar=True)

    phi_h = np.array(all_values[-1])

    output_path = str(output_dir / "phi_h.npz")
    vf_data = ValueFunctionData(
        values=phi_h,
        grid_min=np.array(grid.domain.lo),
        grid_max=np.array(grid.domain.hi),
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
    Paper notes V_h_T may not converge; use T=2.5s per Section V.

    Args:
        config: Game configuration
        preset: Grid preset
        output_dir: Directory to save value function

    Returns:
        Path to saved V_h_T.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_horizontal_relative_grid(config)
    dynamics = HorizontalRelativeDynamics(config)

    # Initial value: horizontal distance = sqrt(x_rel^2 + y_rel^2)
    x_rel = grid.states[..., 0]
    y_rel = grid.states[..., 1]
    initial_values = jnp.sqrt(x_rel**2 + y_rel**2)

    # Per paper Section V, use T=2.5s (V_h_T may not converge)
    T = 2.5
    n_steps = 50

    # Max-over-time postprocessor for worst-case tracking
    solver_settings = hj.SolverSettings.with_accuracy(
        config.grid.solver.accuracy,
        value_postprocessor=lambda t, v: jnp.maximum(v, initial_values),
    )

    times = jnp.linspace(0, -T, n_steps + 1)

    print(f"Solving horizontal max distance (4D grid: {grid.states.shape[:-1]})...")
    all_values = hj.solve(solver_settings, dynamics, grid, times, initial_values, progress_bar=True)

    v_h_t = np.array(all_values[-1])

    output_path = str(output_dir / "V_h_T.npz")
    vf_data = ValueFunctionData(
        values=v_h_t,
        grid_min=np.array(grid.domain.lo),
        grid_max=np.array(grid.domain.hi),
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

    Args:
        v_h_t_path: Path to V_h_T.npz
        d_h: Horizontal capture distance
        output_dir: Directory to save B_h

    Returns:
        Path to saved B_h.npz
    """
    from reach_avoid_game.solvers.value_function_io import load_value_function

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf = load_value_function(v_h_t_path)

    # V_h_T was initialized as sqrt(x_rel^2 + y_rel^2), so it represents
    # worst-case max horizontal distance. B_h = {x : V_h_T(x) <= d_h}.
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
    Target: the target region from config
    The attacker minimizes time to reach target.

    Args:
        config: Game configuration
        preset: Grid preset
        output_dir: Directory to save value function

    Returns:
        Path to saved phi_A_reach.npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_attacker_reaching_grid(config)
    dynamics = AttackerReachingDynamics(config)

    # Target: box region from config
    tr = config.target_region
    x_a = grid.states[..., 0]
    y_a = grid.states[..., 1]

    # SDF for box: negative inside, positive outside
    target_sdf = jnp.maximum(
        jnp.maximum(tr.x_min - x_a, x_a - tr.x_max),
        jnp.maximum(tr.y_min - y_a, y_a - tr.y_max),
    )

    # Obstacle avoid set for attacker
    obstacle_values = None
    if config.obstacles:
        combined = None
        for obs in config.obstacles:
            sdf = jnp.maximum(
                jnp.maximum(obs.x_min - x_a, x_a - obs.x_max),
                jnp.maximum(obs.y_min - y_a, y_a - obs.y_max),
            )
            if combined is None:
                combined = sdf
            else:
                combined = jnp.minimum(combined, sdf)
        obstacle_values = combined

    T = 10.0 if preset == "dev" else 22.0
    n_steps = 50

    if obstacle_values is not None:
        def reach_avoid_postprocessor(t, v):
            return jnp.maximum(v, jnp.minimum(target_sdf, -obstacle_values))

        solver_settings = hj.SolverSettings.with_accuracy(
            config.grid.solver.accuracy,
            value_postprocessor=reach_avoid_postprocessor,
        )
    else:
        solver_settings = hj.SolverSettings.with_accuracy(
            config.grid.solver.accuracy,
        )

    times = jnp.linspace(0, -T, n_steps + 1)

    print(f"Solving attacker reaching (2D grid: {grid.states.shape[:-1]})...")
    all_values = hj.solve(solver_settings, dynamics, grid, times, target_sdf, progress_bar=True)

    phi_a = np.array(all_values[-1])

    output_path = str(output_dir / "phi_A_reach.npz")
    vf_data = ValueFunctionData(
        values=phi_a,
        grid_min=np.array(grid.domain.lo),
        grid_max=np.array(grid.domain.hi),
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
