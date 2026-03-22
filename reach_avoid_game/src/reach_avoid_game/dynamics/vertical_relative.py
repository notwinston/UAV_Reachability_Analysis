"""Vertical relative dynamics for computing V_z_inf.

2D state: [z_rel, v_D_z] where z_rel = z_D - z_A

For tracking: defender minimizes, attacker maximizes.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class VerticalRelativeDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """2D vertical relative dynamics."""

    def __init__(self, config: GameConfig) -> None:
        self.k_z = config.defender.k_z
        self.u_d_z = config.defender.max_speed_vertical
        self.u_a_z = config.attacker.max_speed_vertical

        super().__init__(
            control_mode="min",
            disturbance_mode="max",
            control_space=hj.sets.Box(
                lo=jnp.array([-self.u_d_z]),
                hi=jnp.array([self.u_d_z]),
            ),
            disturbance_space=hj.sets.Box(
                lo=jnp.array([-self.u_a_z]),
                hi=jnp.array([self.u_a_z]),
            ),
        )

    def open_loop_dynamics(self, state, time):
        return jnp.array([state[1], -self.k_z * state[1]])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [self.k_z]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[-1.0], [0.0]])
