"""Attacker dynamics: 3D single integrator model.

State: [x, y, z]
Dynamics:
  dx/dt = d_x
  dy/dt = d_y
  dz/dt = d_z

Speed constraints:
  ||(d_x, d_y)|| <= U_A_h
  |d_z| <= U_A_z
"""

from __future__ import annotations

import numpy as np

from reach_avoid_game.config import AttackerConfig


class AttackerDynamics:
    """3D single integrator dynamics for the attacker UAV."""

    def __init__(self, config: AttackerConfig) -> None:
        self.max_speed_horizontal = config.max_speed_horizontal
        self.max_speed_vertical = config.max_speed_vertical

    def compute_derivatives(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """Compute state derivatives given state and control input.

        Args:
            state: [x, y, z] shape (3,)
            control: [d_x, d_y, d_z] shape (3,) — velocity commands (clamped)

        Returns:
            State derivatives shape (3,)
        """
        d = self.clamp_control(control)
        return np.array([d[0], d[1], d[2]])

    def clamp_control(self, control: np.ndarray) -> np.ndarray:
        """Clamp control inputs to speed limits.

        Args:
            control: [d_x, d_y, d_z] shape (3,)

        Returns:
            Clamped control shape (3,)
        """
        d = np.array(control, dtype=float)
        # Clamp horizontal speed
        d_h_norm = np.sqrt(d[0] ** 2 + d[1] ** 2)
        if d_h_norm > self.max_speed_horizontal:
            scale = self.max_speed_horizontal / d_h_norm
            d[0] *= scale
            d[1] *= scale
        # Clamp vertical speed
        d[2] = np.clip(d[2], -self.max_speed_vertical, self.max_speed_vertical)
        return d
