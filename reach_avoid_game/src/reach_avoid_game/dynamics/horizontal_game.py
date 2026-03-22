"""Horizontal sub-game dynamics for HJ reachability solver.

6D state: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
Dynamics (Eq. 19 in paper).

Two-player game:
  - Defender (control) maximizes value (tries to capture)
  - Attacker (disturbance) minimizes value (tries to escape)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class HorizontalGameDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """6D horizontal sub-game dynamics."""

    def __init__(self, config: GameConfig) -> None:
        self.k_x = config.defender.k_x
        self.k_y = config.defender.k_y
        self.u_d_h = config.defender.max_speed_horizontal
        self.u_a_h = config.attacker.max_speed_horizontal

        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=hj.sets.Box(
                lo=jnp.array([-self.u_d_h, -self.u_d_h]),
                hi=jnp.array([self.u_d_h, self.u_d_h]),
            ),
            disturbance_space=hj.sets.Box(
                lo=jnp.array([-self.u_a_h, -self.u_a_h]),
                hi=jnp.array([self.u_a_h, self.u_a_h]),
            ),
        )

    def open_loop_dynamics(self, state, time):
        return jnp.array([
            state[2], state[3],
            -self.k_x * state[2], -self.k_y * state[3],
            0.0, 0.0,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0], [0.0, 0.0],
            [self.k_x, 0.0], [0.0, self.k_y],
            [0.0, 0.0], [0.0, 0.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
            [1.0, 0.0], [0.0, 1.0],
        ])

    # --- OptimizedDP-compatible pure-Python methods ---

    def opt_ctrl_numpy(self, state, spat_deriv):
        u_x = np.where(spat_deriv[2] * self.k_x >= 0, self.u_d_h, -self.u_d_h)
        u_y = np.where(spat_deriv[3] * self.k_y >= 0, self.u_d_h, -self.u_d_h)
        return u_x, u_y

    def opt_dstb_numpy(self, state, spat_deriv):
        d_x = np.where(spat_deriv[4] >= 0, -self.u_a_h, self.u_a_h)
        d_y = np.where(spat_deriv[5] >= 0, -self.u_a_h, self.u_a_h)
        return d_x, d_y


class HorizontalGameTrackingDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """6D horizontal tracking dynamics (reversed: ctrl min, dstb max)."""

    def __init__(self, config: GameConfig) -> None:
        self.k_x = config.defender.k_x
        self.k_y = config.defender.k_y
        self.u_d_h = config.defender.max_speed_horizontal
        self.u_a_h = config.attacker.max_speed_horizontal

        super().__init__(
            control_mode="min",
            disturbance_mode="max",
            control_space=hj.sets.Box(
                lo=jnp.array([-self.u_d_h, -self.u_d_h]),
                hi=jnp.array([self.u_d_h, self.u_d_h]),
            ),
            disturbance_space=hj.sets.Box(
                lo=jnp.array([-self.u_a_h, -self.u_a_h]),
                hi=jnp.array([self.u_a_h, self.u_a_h]),
            ),
        )

    def open_loop_dynamics(self, state, time):
        return jnp.array([
            state[2], state[3],
            -self.k_x * state[2], -self.k_y * state[3],
            0.0, 0.0,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0], [0.0, 0.0],
            [self.k_x, 0.0], [0.0, self.k_y],
            [0.0, 0.0], [0.0, 0.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
            [1.0, 0.0], [0.0, 1.0],
        ])
