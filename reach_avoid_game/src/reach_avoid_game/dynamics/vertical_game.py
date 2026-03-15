"""Vertical sub-game dynamics for HJ reachability solver.

3D state: [z_D, v_D_z, z_A]
  - z_D: defender vertical position
  - v_D_z: defender vertical velocity
  - z_A: attacker vertical position

Dynamics (Eq. 25 in paper):
  z_D_dot   = v_D_z
  v_D_z_dot = k_z * (u_z - v_D_z)    [defender control u_z, |u_z| <= U_D_z]
  z_A_dot   = d_z                     [attacker control d_z, |d_z| <= U_A_z]

Two-player game:
  - Defender (control) maximizes value (tries to capture)
  - Attacker (disturbance) minimizes value (tries to escape)
"""

from __future__ import annotations

import jax.numpy as jnp

import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class VerticalGameDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """3D vertical sub-game dynamics for HJ reachability.

    Control: u_z (scalar) — defender vertical velocity command
    Disturbance: d_z (scalar) — attacker vertical velocity
    """

    def __init__(self, config: GameConfig) -> None:
        self.k_z = config.defender.k_z
        self.u_d_z = config.defender.max_speed_vertical
        self.u_a_z = config.attacker.max_speed_vertical

        super().__init__(
            control_mode="max",
            disturbance_mode="min",
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
        """Open-loop dynamics f(x, t): drift when no control/disturbance applied.

        z_D_dot   = v_D_z
        v_D_z_dot = k_z * (0 - v_D_z) = -k_z * v_D_z
        z_A_dot   = 0
        """
        return jnp.array([
            state[1],              # z_D_dot = v_D_z
            -self.k_z * state[1],  # v_D_z_dot = -k_z * v_D_z (open-loop part)
            0.0,                   # z_A_dot = 0 (no control)
        ])

    def control_jacobian(self, state, time):
        """Control Jacobian G_u(x, t): maps control u_z to state derivatives.

        Only v_D_z_dot is affected: v_D_z_dot += k_z * u_z
        """
        return jnp.array([
            [0.0],
            [self.k_z],
            [0.0],
        ])

    def disturbance_jacobian(self, state, time):
        """Disturbance Jacobian G_d(x, t): maps disturbance d_z to state derivatives.

        Only z_A_dot is affected: z_A_dot = d_z
        """
        return jnp.array([
            [0.0],
            [0.0],
            [1.0],
        ])
