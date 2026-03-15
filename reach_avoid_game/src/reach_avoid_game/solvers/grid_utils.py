"""Grid creation utilities for HJ reachability solvers."""

from __future__ import annotations

import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


def create_vertical_game_grid(config: GameConfig, preset: str | None = None) -> hj.Grid:
    """Create a 3D grid for the vertical sub-game.

    Dimensions: [z_D, v_D_z, z_A]
    Domain:
      z_D in [0, room_height]
      v_D_z in [-U_D_z, U_D_z]
      z_A in [0, room_height]

    Args:
        config: Game configuration
        preset: Grid preset name to override config (e.g., "dev", "paper")

    Returns:
        hj_reachability Grid object
    """
    room_height = config.room.z_max
    u_d_z = config.defender.max_speed_vertical

    # Get grid resolution from config (preset already applied in config loading)
    grid_3d = config.grid.vertical_3d
    shape = (grid_3d.z_d_points, grid_3d.v_dz_points, grid_3d.z_a_points)

    domain = hj.sets.Box(
        lo=jnp.array([0.0, -u_d_z, 0.0]),
        hi=jnp.array([room_height, u_d_z, room_height]),
    )

    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=domain,
        shape=shape,
        boundary_conditions=(
            hj.boundary_conditions.extrapolate,  # z_D
            hj.boundary_conditions.extrapolate,  # v_D_z
            hj.boundary_conditions.extrapolate,  # z_A
        ),
    )


def create_vertical_relative_grid(config: GameConfig, preset: str | None = None) -> hj.Grid:
    """Create a 2D grid for vertical relative dynamics (for V_z_inf).

    Dimensions: [z_rel, v_D_z] where z_rel = z_D - z_A
    Domain:
      z_rel in [-room_height, room_height]
      v_D_z in [-U_D_z, U_D_z]

    Args:
        config: Game configuration
        preset: Grid preset name (unused, uses config values directly)

    Returns:
        hj_reachability Grid object
    """
    z_range = config.grid.vertical.z_rel_range
    v_range = config.grid.vertical.v_dz_range
    shape = (config.grid.vertical.z_rel_points, config.grid.vertical.v_dz_points)

    domain = hj.sets.Box(
        lo=jnp.array([z_range[0], v_range[0]]),
        hi=jnp.array([z_range[1], v_range[1]]),
    )

    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=domain,
        shape=shape,
        boundary_conditions=(
            hj.boundary_conditions.extrapolate,  # z_rel
            hj.boundary_conditions.extrapolate,  # v_D_z
        ),
    )
