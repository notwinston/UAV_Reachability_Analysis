"""Attacker reaching dynamics for T_goal computation.

2D state: [x_A, y_A]
The attacker minimizes the value function (reaches target ASAP).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class AttackerReachingDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """2D attacker reaching dynamics."""

    def __init__(self, config: GameConfig) -> None:
        self.u_a_h = config.attacker.max_speed_horizontal

        super().__init__(
            control_mode="min",
            disturbance_mode="max",
            control_space=hj.sets.Box(
                lo=jnp.array([-self.u_a_h, -self.u_a_h]),
                hi=jnp.array([self.u_a_h, self.u_a_h]),
            ),
            disturbance_space=hj.sets.Box(
                lo=jnp.array([0.0]),
                hi=jnp.array([0.0]),
            ),
        )

    def open_loop_dynamics(self, state, time):
        return jnp.array([0.0, 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0]])
