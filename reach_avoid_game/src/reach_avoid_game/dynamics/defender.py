"""Defender dynamics: 6D double integrator model.

State: [x, y, z, vx, vy, vz]
Dynamics:
  dx/dt = vx
  dy/dt = vy
  dz/dt = vz
  dvx/dt = k_x * (u_x - vx)
  dvy/dt = k_y * (u_y - vy)
  dvz/dt = k_z * (u_z - vz)

Speed constraints:
  ||(u_x, u_y)|| <= U_D_h
  |u_z| <= U_D_z
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from reach_avoid_game.config import DefenderConfig


class DefenderDynamics:
    """6D double integrator dynamics for the defender UAV."""

    def __init__(self, config: DefenderConfig) -> None:
        self.k_x = config.k_x
        self.k_y = config.k_y
        self.k_z = config.k_z
        self.max_speed_horizontal = config.max_speed_horizontal
        self.max_speed_vertical = config.max_speed_vertical

    def compute_derivatives(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """Compute state derivatives given state and control input.

        Args:
            state: [x, y, z, vx, vy, vz] shape (6,)
            control: [u_x, u_y, u_z] shape (3,) — velocity commands (clamped)

        Returns:
            State derivatives shape (6,)
        """
        vx, vy, vz = state[3], state[4], state[5]
        u = self.clamp_control(control)
        u_x, u_y, u_z = u[0], u[1], u[2]

        return np.array([
            vx,
            vy,
            vz,
            self.k_x * (u_x - vx),
            self.k_y * (u_y - vy),
            self.k_z * (u_z - vz),
        ])

    def clamp_control(self, control: np.ndarray) -> np.ndarray:
        """Clamp control inputs to speed limits.

        Args:
            control: [u_x, u_y, u_z] shape (3,)

        Returns:
            Clamped control shape (3,)
        """
        u = np.array(control, dtype=float)
        # Clamp horizontal speed
        u_h_norm = np.sqrt(u[0] ** 2 + u[1] ** 2)
        if u_h_norm > self.max_speed_horizontal:
            scale = self.max_speed_horizontal / u_h_norm
            u[0] *= scale
            u[1] *= scale
        # Clamp vertical speed
        u[2] = np.clip(u[2], -self.max_speed_vertical, self.max_speed_vertical)
        return u
