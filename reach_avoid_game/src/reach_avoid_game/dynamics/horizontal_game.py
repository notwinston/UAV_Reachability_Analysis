"""Horizontal sub-game dynamics for HJ reachability solver.

6D state: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
  - x_D, y_D: defender horizontal position
  - v_D_x, v_D_y: defender horizontal velocity
  - x_A, y_A: attacker horizontal position

Dynamics (Eq. 19 in paper):
  x_D_dot   = v_D_x
  y_D_dot   = v_D_y
  v_D_x_dot = k_x * (u_x - v_D_x)    [defender control u_x]
  v_D_y_dot = k_y * (u_y - v_D_y)    [defender control u_y]
  x_A_dot   = d_x                     [attacker control d_x]
  y_A_dot   = d_y                     [attacker control d_y]

Speed constraints:
  ||(u_x, u_y)|| <= U_D_h
  ||(d_x, d_y)|| <= U_A_h

Two-player game:
  - Defender (control) maximizes value (tries to capture)
  - Attacker (disturbance) minimizes value (tries to escape)
"""

from __future__ import annotations

import jax.numpy as jnp

import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class HorizontalGameDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """6D horizontal sub-game dynamics for HJ reachability.

    Control: [u_x, u_y] — defender horizontal velocity commands
    Disturbance: [d_x, d_y] — attacker horizontal velocities
    """

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
        """Open-loop dynamics f(x, t): drift when no control/disturbance applied.

        x_D_dot   = v_D_x
        y_D_dot   = v_D_y
        v_D_x_dot = -k_x * v_D_x  (open-loop: u_x=0)
        v_D_y_dot = -k_y * v_D_y  (open-loop: u_y=0)
        x_A_dot   = 0
        y_A_dot   = 0
        """
        return jnp.array([
            state[2],                # x_D_dot = v_D_x
            state[3],                # y_D_dot = v_D_y
            -self.k_x * state[2],    # v_D_x_dot = -k_x * v_D_x
            -self.k_y * state[3],    # v_D_y_dot = -k_y * v_D_y
            0.0,                     # x_A_dot = 0
            0.0,                     # y_A_dot = 0
        ])

    def control_jacobian(self, state, time):
        """Control Jacobian G_u(x, t): maps control [u_x, u_y] to state derivatives.

        v_D_x_dot += k_x * u_x
        v_D_y_dot += k_y * u_y
        """
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.k_x, 0.0],
            [0.0, self.k_y],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

    def disturbance_jacobian(self, state, time):
        """Disturbance Jacobian G_d(x, t): maps disturbance [d_x, d_y] to state derivatives.

        x_A_dot = d_x
        y_A_dot = d_y
        """
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])


class HorizontalGameTrackingDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """6D horizontal tracking dynamics for computing V_h_T in absolute coordinates.

    Same dynamics as HorizontalGameDynamics but with reversed optimization:
      - Defender (control) minimizes value (minimizes distance to attacker)
      - Attacker (disturbance) maximizes value (maximizes distance from defender)

    Used for the 6D extension of the maximum distance value function.
    """

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
            state[2],
            state[3],
            -self.k_x * state[2],
            -self.k_y * state[3],
            0.0,
            0.0,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.k_x, 0.0],
            [0.0, self.k_y],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
