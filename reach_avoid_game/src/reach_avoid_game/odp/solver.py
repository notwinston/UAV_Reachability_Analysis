"""HJ PDE solver — OptimizedDP-compatible API using hj_reachability backend.

Provides:
- HJSolver: OptimizedDP-compatible entry point that delegates to hj.solve
- computeSpatDerivArray: spatial derivative computation (pure NumPy)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.odp.grid import Grid


def _odp_grid_to_hj_grid(grid: Grid) -> hj.Grid:
    """Convert OptimizedDP Grid to hj_reachability Grid."""
    domain = hj.sets.Box(
        lo=jnp.array(grid.min),
        hi=jnp.array(grid.max),
    )
    bcs = tuple(
        hj.boundary_conditions.periodic if d in grid.pDim
        else hj.boundary_conditions.extrapolate
        for d in range(grid.dims)
    )
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=domain,
        shape=tuple(int(x) for x in grid.pts_each_dim),
        boundary_conditions=bcs,
    )


def _spatial_derivs(V: np.ndarray, grid: Grid, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute left and right spatial derivatives along one dimension."""
    n = V.shape[dim]
    dx = grid.dx[dim]
    is_periodic = dim in grid.pDim

    left_deriv = np.zeros_like(V)
    right_deriv = np.zeros_like(V)

    def _slc(d, s):
        sl = [slice(None)] * V.ndim
        sl[d] = s
        return tuple(sl)

    # Interior
    left_deriv[_slc(dim, slice(1, n - 1))] = (
        V[_slc(dim, slice(1, n - 1))] - V[_slc(dim, slice(0, n - 2))]
    ) / dx
    right_deriv[_slc(dim, slice(1, n - 1))] = (
        V[_slc(dim, slice(2, n))] - V[_slc(dim, slice(1, n - 1))]
    ) / dx

    V_0, V_1 = V[_slc(dim, 0)], V[_slc(dim, 1)]
    V_nm1, V_nm2 = V[_slc(dim, n - 1)], V[_slc(dim, n - 2)]

    if is_periodic:
        left_deriv[_slc(dim, 0)] = (V_0 - V_nm1) / dx
        right_deriv[_slc(dim, 0)] = (V_1 - V_0) / dx
        left_deriv[_slc(dim, n - 1)] = (V_nm1 - V_nm2) / dx
        right_deriv[_slc(dim, n - 1)] = (V_0 - V_nm1) / dx
    else:
        left_ghost = V_0 + np.abs(V_1 - V_0) * np.sign(V_0)
        left_deriv[_slc(dim, 0)] = (V_0 - left_ghost) / dx
        right_deriv[_slc(dim, 0)] = (V_1 - V_0) / dx
        right_ghost = V_nm1 + np.abs(V_nm1 - V_nm2) * np.sign(V_nm1)
        left_deriv[_slc(dim, n - 1)] = (V_nm1 - V_nm2) / dx
        right_deriv[_slc(dim, n - 1)] = (right_ghost - V_nm1) / dx

    return left_deriv, right_deriv


def computeSpatDerivArray(
    grid: Grid,
    V: np.ndarray,
    deriv_dim: int,
    accuracy: str = "low",
) -> np.ndarray:
    """Compute spatial derivative array (1-indexed dim, matching OptimizedDP)."""
    dim = deriv_dim - 1
    left_deriv, right_deriv = _spatial_derivs(V, grid, dim)
    return (left_deriv + right_deriv) / 2.0


def HJSolver(
    dynamics_obj: Any,
    grid: Grid,
    multiple_value: np.ndarray | list,
    tau: np.ndarray,
    compMethod: dict,
    saveAllTimeSteps: bool = False,
    accuracy: str = "low",
    untilConvergent: bool = False,
    epsilon: float = 2e-3,
) -> np.ndarray:
    """Solve HJ PDE — OptimizedDP-compatible API with hj_reachability backend.

    The dynamics_obj must be an hj_reachability ControlAndDisturbanceAffineDynamics
    subclass (which our dynamics classes are).

    Args:
        dynamics_obj: hj_reachability dynamics object.
        grid: OptimizedDP-compatible Grid.
        multiple_value: Initial value function, or [target, constraint] list.
        tau: Time array (0 to T).
        compMethod: Dict with "TargetSetMode" and optionally "ObstacleSetMode".
        saveAllTimeSteps: Return all time steps.
        accuracy: "low" or "medium".

    Returns:
        Final value function (or all time steps if saveAllTimeSteps).
    """
    print("Welcome to optimized_dp (hj_reachability backend)\n")

    # Parse target and constraint
    if isinstance(multiple_value, list):
        target = multiple_value[0]
        constraint = multiple_value[1]
    else:
        target = multiple_value
        constraint = None

    # Convert Grid
    hj_grid = _odp_grid_to_hj_grid(grid)

    initial_values = jnp.array(target.astype(np.float64))

    # Map compMethod to solver settings
    mode = compMethod["TargetSetMode"]

    if mode == "maxVWithV0":
        solver_settings = hj.SolverSettings.with_accuracy(
            accuracy,
            value_postprocessor=lambda t, v: jnp.maximum(v, initial_values),
        )
    elif mode == "minVWithV0":
        if constraint is not None and "ObstacleSetMode" in compMethod:
            # Reach-avoid BRT (Fisac 2015):
            #   obs = constraint = -obstacle_values > 0 inside obstacles
            #   min(v, l): BRT clamp — target states stay captured (V <= l)
            #   max(..., obs): obstacle states forced positive (V >= obs > 0 = defender loses)
            obs = jnp.array(constraint.astype(np.float64))
            def reach_avoid_pp(t, v):
                return jnp.maximum(jnp.minimum(v, initial_values), obs)
            solver_settings = hj.SolverSettings.with_accuracy(
                accuracy,
                value_postprocessor=reach_avoid_pp,
            )
        else:
            # BRT (backward reachable tube): clamp with min so states that enter
            # the target set stay captured as time propagates backward.
            solver_settings = hj.SolverSettings.with_accuracy(
                accuracy,
                value_postprocessor=lambda t, v: jnp.minimum(v, initial_values),
            )
    elif mode == "none":
        solver_settings = hj.SolverSettings.with_accuracy(accuracy)
    else:
        solver_settings = hj.SolverSettings.with_accuracy(accuracy)

    # Time: hj_reachability expects backward time (0 to -T)
    T = float(tau[-1])
    n_steps = len(tau) - 1

    # Cap steps for memory
    grid_size = int(np.prod(grid.pts_each_dim))
    max_mem_bytes = 1_500_000_000
    max_steps = max(8, int(max_mem_bytes / (grid_size * 8)) - 1)
    if n_steps > max_steps:
        print(f"  Reducing time steps from {n_steps} to {max_steps} (memory limit)")
        n_steps = max_steps

    times = jnp.linspace(0, -T, n_steps + 1)

    print(f"Grid: {tuple(grid.pts_each_dim)}, T={T}s, {n_steps} steps")

    t0 = time.time()
    all_values = hj.solve(solver_settings, dynamics_obj, hj_grid, times, initial_values)
    elapsed = time.time() - t0

    print(f"Computation time: {elapsed:.3f}s")
    print("Finished solving\n")

    if saveAllTimeSteps:
        result = np.array(all_values)
        axes = list(range(1, result.ndim)) + [0]
        return result.transpose(axes)

    return np.array(all_values[-1])
