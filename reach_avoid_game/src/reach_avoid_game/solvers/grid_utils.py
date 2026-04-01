"""Grid creation utilities for HJ reachability solvers.

Uses the OptimizedDP-compatible Grid from reach_avoid_game.odp.
"""

from __future__ import annotations

import numpy as np

from reach_avoid_game.config import GameConfig
from reach_avoid_game.odp.grid import Grid


def create_vertical_game_grid(config: GameConfig, preset: str | None = None) -> Grid:
    """Create a 3D grid for the vertical sub-game.

    Dimensions: [z_D, v_D_z, z_A]
    """
    room_height = config.room.z_max
    u_d_z = config.defender.max_speed_vertical

    grid_3d = config.grid.vertical_3d
    shape = [grid_3d.z_d_points, grid_3d.v_dz_points, grid_3d.z_a_points]

    return Grid(
        minBounds=np.array([0.0, -u_d_z, 0.0]),
        maxBounds=np.array([room_height, u_d_z, room_height]),
        dims=3,
        pts_each_dim=np.array(shape),
    )


def create_vertical_relative_grid(config: GameConfig, preset: str | None = None) -> Grid:
    """Create a 2D grid for vertical relative dynamics (for V_z_inf).

    Dimensions: [z_rel, v_D_z] where z_rel = z_D - z_A
    """
    z_range = config.grid.vertical.z_rel_range
    v_range = config.grid.vertical.v_dz_range
    shape = [config.grid.vertical.z_rel_points, config.grid.vertical.v_dz_points]

    return Grid(
        minBounds=np.array([z_range[0], v_range[0]]),
        maxBounds=np.array([z_range[1], v_range[1]]),
        dims=2,
        pts_each_dim=np.array(shape),
    )


def create_horizontal_game_grid(config: GameConfig) -> Grid:
    """Create a 6D grid for the horizontal sub-game.

    Dimensions: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
    """
    h = config.grid.horizontal
    u_d_h = config.defender.max_speed_horizontal

    shape = [
        h.game_x_points,       # x_D
        h.game_y_points,       # y_D
        h.game_vel_x_points,   # v_D_x
        h.game_vel_y_points,   # v_D_y
        h.game_x_points,       # x_A
        h.game_y_points,       # y_A
    ]

    return Grid(
        minBounds=np.array([
            config.room.x_min, config.room.y_min,
            -u_d_h, -u_d_h,
            config.room.x_min, config.room.y_min,
        ]),
        maxBounds=np.array([
            config.room.x_max, config.room.y_max,
            u_d_h, u_d_h,
            config.room.x_max, config.room.y_max,
        ]),
        dims=6,
        pts_each_dim=np.array(shape),
    )


def create_horizontal_relative_grid(config: GameConfig) -> Grid:
    """Create a 4D grid for horizontal relative dynamics (for V_h_T).

    Dimensions: [x_rel, y_rel, v_D_x, v_D_y]
    """
    h = config.grid.horizontal
    u_d_h = config.defender.max_speed_horizontal
    r = h.rel_pos_range

    shape = [h.rel_pos_points, h.rel_pos_points, h.rel_vel_points, h.rel_vel_points]

    return Grid(
        minBounds=np.array([-r, -r, -u_d_h, -u_d_h]),
        maxBounds=np.array([r, r, u_d_h, u_d_h]),
        dims=4,
        pts_each_dim=np.array(shape),
    )


def create_attacker_reaching_grid(config: GameConfig) -> Grid:
    """Create a 2D grid for attacker reaching computation.

    Dimensions: [x_A, y_A]
    """
    h = config.grid.horizontal

    shape = [h.reach_x_points, h.reach_y_points]

    return Grid(
        minBounds=np.array([config.room.x_min, config.room.y_min]),
        maxBounds=np.array([config.room.x_max, config.room.y_max]),
        dims=2,
        pts_each_dim=np.array(shape),
    )
