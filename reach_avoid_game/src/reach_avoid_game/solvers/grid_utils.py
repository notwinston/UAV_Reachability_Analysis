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


def create_horizontal_game_grid(config: GameConfig) -> hj.Grid:
    """Create a 6D grid for the horizontal sub-game.

    Dimensions: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
    Domain:
      x_D, x_A in [room.x_min, room.x_max]
      y_D, y_A in [room.y_min, room.y_max]
      v_D_x in [-U_D_h, U_D_h], v_D_y in [-U_D_h, U_D_h]

    Grid resolution per dimension is set independently to match the paper:
      Paper: 85 x 45 x 8 x 7 x 85 x 45 (position x, y, vel x, vel y, att x, att y)

    Args:
        config: Game configuration (preset already applied)

    Returns:
        hj_reachability Grid object
    """
    h = config.grid.horizontal
    u_d_h = config.defender.max_speed_horizontal

    shape = (
        h.game_x_points,       # x_D
        h.game_y_points,       # y_D
        h.game_vel_x_points,   # v_D_x
        h.game_vel_y_points,   # v_D_y
        h.game_x_points,       # x_A (same resolution as x_D)
        h.game_y_points,       # y_A (same resolution as y_D)
    )

    domain = hj.sets.Box(
        lo=jnp.array([
            config.room.x_min, config.room.y_min,
            -u_d_h, -u_d_h,
            config.room.x_min, config.room.y_min,
        ]),
        hi=jnp.array([
            config.room.x_max, config.room.y_max,
            u_d_h, u_d_h,
            config.room.x_max, config.room.y_max,
        ]),
    )

    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=domain,
        shape=shape,
        boundary_conditions=(
            hj.boundary_conditions.extrapolate,  # x_D
            hj.boundary_conditions.extrapolate,  # y_D
            hj.boundary_conditions.extrapolate,  # v_D_x
            hj.boundary_conditions.extrapolate,  # v_D_y
            hj.boundary_conditions.extrapolate,  # x_A
            hj.boundary_conditions.extrapolate,  # y_A
        ),
    )


def create_horizontal_relative_grid(config: GameConfig) -> hj.Grid:
    """Create a 4D grid for horizontal relative dynamics (for V_h_T).

    Dimensions: [x_rel, y_rel, v_D_x, v_D_y]
      where x_rel = x_D - x_A, y_rel = y_D - y_A
    Domain:
      x_rel, y_rel in [-rel_pos_range, rel_pos_range]
        Paper uses [-3, 3] (matching capture distance d_h=3m)
      v_D_x, v_D_y in [-U_D_h, U_D_h]

    Args:
        config: Game configuration

    Returns:
        hj_reachability Grid object
    """
    h = config.grid.horizontal
    u_d_h = config.defender.max_speed_horizontal
    r = h.rel_pos_range

    shape = (h.rel_pos_points, h.rel_pos_points, h.rel_vel_points, h.rel_vel_points)

    domain = hj.sets.Box(
        lo=jnp.array([-r, -r, -u_d_h, -u_d_h]),
        hi=jnp.array([r, r, u_d_h, u_d_h]),
    )

    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=domain,
        shape=shape,
        boundary_conditions=(
            hj.boundary_conditions.extrapolate,  # x_rel
            hj.boundary_conditions.extrapolate,  # y_rel
            hj.boundary_conditions.extrapolate,  # v_D_x
            hj.boundary_conditions.extrapolate,  # v_D_y
        ),
    )


def create_attacker_reaching_grid(config: GameConfig) -> hj.Grid:
    """Create a 2D grid for attacker reaching computation.

    Dimensions: [x_A, y_A]
    Used to compute T_goal: earliest time attacker reaches target region.

    Args:
        config: Game configuration

    Returns:
        hj_reachability Grid object
    """
    h = config.grid.horizontal

    shape = (h.reach_x_points, h.reach_y_points)

    domain = hj.sets.Box(
        lo=jnp.array([config.room.x_min, config.room.y_min]),
        hi=jnp.array([config.room.x_max, config.room.y_max]),
    )

    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=domain,
        shape=shape,
        boundary_conditions=(
            hj.boundary_conditions.extrapolate,  # x_A
            hj.boundary_conditions.extrapolate,  # y_A
        ),
    )
