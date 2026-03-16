"""Horizontal relative dynamics for computing V_h_T (max distance value function).

4D state: [x_rel, y_rel, v_D_x, v_D_y]
  where x_rel = x_D - x_A, y_rel = y_D - y_A

Dynamics:
  x_rel_dot = v_D_x - d_x
  y_rel_dot = v_D_y - d_y
  v_D_x_dot = k_x * (u_x - v_D_x)
  v_D_y_dot = k_y * (u_y - v_D_y)

For tracking (V_h_T):
  - Defender minimizes value (wants to stay close)
  - Attacker maximizes value (wants to escape)
"""

from __future__ import annotations

import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class HorizontalRelativeDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """4D horizontal relative dynamics for tracking value function computation.

    Control: [u_x, u_y] — defender horizontal velocity commands
    Disturbance: [d_x, d_y] — attacker horizontal velocities
    """

    def __init__(self, config: GameConfig) -> None:
        self.k_x = config.defender.k_x
        self.k_y = config.defender.k_y
        self.u_d_h = config.defender.max_speed_horizontal
        self.u_a_h = config.attacker.max_speed_horizontal

        # For tracking: defender minimizes (tracks), attacker maximizes (escapes)
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
        """Open-loop dynamics:
        x_rel_dot = v_D_x, y_rel_dot = v_D_y,
        v_D_x_dot = -k_x * v_D_x, v_D_y_dot = -k_y * v_D_y.
        """
        return jnp.array([
            state[2],                # x_rel_dot = v_D_x
            state[3],                # y_rel_dot = v_D_y
            -self.k_x * state[2],    # v_D_x_dot = -k_x * v_D_x
            -self.k_y * state[3],    # v_D_y_dot = -k_y * v_D_y
        ])

    def control_jacobian(self, state, time):
        """Control affects velocities: v_D_x_dot += k_x * u_x, v_D_y_dot += k_y * u_y."""
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.k_x, 0.0],
            [0.0, self.k_y],
        ])

    def disturbance_jacobian(self, state, time):
        """Disturbance affects relative position: x_rel_dot -= d_x, y_rel_dot -= d_y."""
        return jnp.array([
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
