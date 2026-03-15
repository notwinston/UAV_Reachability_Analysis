"""Vertical relative dynamics for computing V_z_inf (max distance value function).

2D state: [z_rel, v_D_z] where z_rel = z_D - z_A
Dynamics:
  z_rel_dot = v_D_z - d_z      (d_z is attacker's vertical velocity)
  v_D_z_dot = k_z * (u_z - v_D_z)

This is used to compute the worst-case maximum vertical distance over time.
The defender (control) wants to minimize the distance (tracking),
the attacker (disturbance) wants to maximize the distance (escape).

For tracking (V_z_inf):
  - Defender minimizes value (wants to stay close)
  - Attacker maximizes value (wants to escape)
"""

from __future__ import annotations

import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class VerticalRelativeDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """2D vertical relative dynamics for tracking value function computation.

    Control: u_z (scalar) — defender vertical velocity command
    Disturbance: d_z (scalar) — attacker vertical velocity
    """

    def __init__(self, config: GameConfig) -> None:
        self.k_z = config.defender.k_z
        self.u_d_z = config.defender.max_speed_vertical
        self.u_a_z = config.attacker.max_speed_vertical

        # For tracking: defender minimizes (tracks), attacker maximizes (escapes)
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
        """Open-loop dynamics: z_rel_dot = v_D_z, v_D_z_dot = -k_z * v_D_z."""
        return jnp.array([
            state[1],              # z_rel_dot = v_D_z (no disturbance)
            -self.k_z * state[1],  # v_D_z_dot = -k_z * v_D_z
        ])

    def control_jacobian(self, state, time):
        """Control affects only v_D_z: v_D_z_dot += k_z * u_z."""
        return jnp.array([
            [0.0],
            [self.k_z],
        ])

    def disturbance_jacobian(self, state, time):
        """Disturbance affects z_rel: z_rel_dot -= d_z."""
        return jnp.array([
            [-1.0],
            [0.0],
        ])
