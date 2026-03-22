"""Horizontal relative dynamics for computing V_h_T.

4D state: [x_rel, y_rel, v_D_x, v_D_y]

For tracking: defender minimizes, attacker maximizes.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class HorizontalRelativeDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """4D horizontal relative dynamics."""

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
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0], [0.0, 0.0],
            [self.k_x, 0.0], [0.0, self.k_y],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [-1.0, 0.0], [0.0, -1.0],
            [0.0, 0.0], [0.0, 0.0],
        ])
