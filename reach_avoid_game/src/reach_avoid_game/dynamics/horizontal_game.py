"""Horizontal sub-game dynamics for OptimizedDP.

6D state: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
Dynamics (Eq. 19 in paper).

Two-player game:
  - Defender (control) maximizes value (tries to capture)
  - Attacker (disturbance) minimizes value (tries to escape)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reach_avoid_game.dynamics.odp_utils import make_box, require_hcl

if TYPE_CHECKING:
    from reach_avoid_game.config import GameConfig


class HorizontalGameDynamics:
    """6D horizontal sub-game dynamics."""

    def __init__(self, config: GameConfig) -> None:
        self.k_x = config.defender.k_x
        self.k_y = config.defender.k_y
        self.u_d_h = config.defender.max_speed_horizontal
        self.u_a_h = config.attacker.max_speed_horizontal
        self.uMode = "max"
        self.dMode = "min"
        self.control_space = make_box(
            [-self.u_d_h, -self.u_d_h], [self.u_d_h, self.u_d_h],
        )
        self.disturbance_space = make_box(
            [-self.u_a_h, -self.u_a_h], [self.u_a_h, self.u_a_h],
        )

    def open_loop_dynamics(self, state, time=0.0):
        return np.array([
            state[2], state[3],
            -self.k_x * state[2], -self.k_y * state[3],
            0.0, 0.0,
        ])

    def control_jacobian(self, state, time=0.0):
        return np.array([
            [0.0, 0.0], [0.0, 0.0],
            [self.k_x, 0.0], [0.0, self.k_y],
            [0.0, 0.0], [0.0, 0.0],
        ])

    def disturbance_jacobian(self, state, time=0.0):
        return np.array([
            [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
            [1.0, 0.0], [0.0, 1.0],
        ])

    def __call__(self, state, control, disturbance, time=0.0):
        return np.array(self.dynamics_inPython(
            state, control[0], control[1], disturbance[0], disturbance[1],
        ))

    def optimal_control(self, state, time, spat_deriv):
        return np.array(self.optCtrl_inPython(state, spat_deriv))

    def optimal_disturbance(self, state, time, spat_deriv):
        return np.array(self.optDstb_inPython(state, spat_deriv))

    def opt_ctrl(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_u_x = hcl.scalar(self.u_d_h, "opt_u_x")
        opt_u_y = hcl.scalar(self.u_d_h, "opt_u_y")
        dummy_1 = hcl.scalar(0, "opt_u_dummy_1")
        dummy_2 = hcl.scalar(0, "opt_u_dummy_2")
        with hcl.if_(spat_deriv[2] * self.k_x < 0):
            opt_u_x[0] = -self.u_d_h
        with hcl.if_(spat_deriv[3] * self.k_y < 0):
            opt_u_y[0] = -self.u_d_h
        return (opt_u_x[0], opt_u_y[0], dummy_1[0], dummy_2[0])

    def opt_dstb(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_d_x = hcl.scalar(self.u_a_h, "opt_d_x")
        opt_d_y = hcl.scalar(self.u_a_h, "opt_d_y")
        dummy_1 = hcl.scalar(0, "opt_d_dummy_1")
        dummy_2 = hcl.scalar(0, "opt_d_dummy_2")
        with hcl.if_(spat_deriv[4] >= 0):
            opt_d_x[0] = -self.u_a_h
        with hcl.if_(spat_deriv[5] >= 0):
            opt_d_y[0] = -self.u_a_h
        return (opt_d_x[0], opt_d_y[0], dummy_1[0], dummy_2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        hcl = require_hcl()
        x_d_dot = hcl.scalar(0, "x_d_dot")
        y_d_dot = hcl.scalar(0, "y_d_dot")
        v_dx_dot = hcl.scalar(0, "v_dx_dot")
        v_dy_dot = hcl.scalar(0, "v_dy_dot")
        x_a_dot = hcl.scalar(0, "x_a_dot")
        y_a_dot = hcl.scalar(0, "y_a_dot")
        x_d_dot[0] = state[2]
        y_d_dot[0] = state[3]
        v_dx_dot[0] = self.k_x * (uOpt[0] - state[2])
        v_dy_dot[0] = self.k_y * (uOpt[1] - state[3])
        x_a_dot[0] = dOpt[0]
        y_a_dot[0] = dOpt[1]
        return (
            x_d_dot[0], y_d_dot[0], v_dx_dot[0],
            v_dy_dot[0], x_a_dot[0], y_a_dot[0],
        )

    # --- OptimizedDP-compatible pure-Python methods ---

    def opt_ctrl_numpy(self, state, spat_deriv):
        u_x = np.where(spat_deriv[2] * self.k_x >= 0, self.u_d_h, -self.u_d_h)
        u_y = np.where(spat_deriv[3] * self.k_y >= 0, self.u_d_h, -self.u_d_h)
        return u_x, u_y

    def opt_dstb_numpy(self, state, spat_deriv):
        d_x = np.where(spat_deriv[4] >= 0, -self.u_a_h, self.u_a_h)
        d_y = np.where(spat_deriv[5] >= 0, -self.u_a_h, self.u_a_h)
        return d_x, d_y

    # --- odp.dynamics.HorizontalGame6D-compatible interface ---

    def optCtrl_inPython(self, state, spat_deriv):
        """Optimal defender control (odp-style). Returns (u_x, u_y)."""
        u_x = float(self.u_d_h if spat_deriv[2] * self.k_x >= 0 else -self.u_d_h)
        u_y = float(self.u_d_h if spat_deriv[3] * self.k_y >= 0 else -self.u_d_h)
        return u_x, u_y

    def optDstb_inPython(self, state, spat_deriv):
        """Optimal attacker disturbance (odp-style). Returns (d_x, d_y)."""
        d_x = float(-self.u_a_h if spat_deriv[4] >= 0 else self.u_a_h)
        d_y = float(-self.u_a_h if spat_deriv[5] >= 0 else self.u_a_h)
        return d_x, d_y

    def dynamics_inPython(self, state, u_x, u_y, d_x, d_y):
        """State derivatives (pure Python tuple)."""
        return (
            state[2], state[3],
            self.k_x * (u_x - state[2]),
            self.k_y * (u_y - state[3]),
            d_x, d_y,
        )


class HorizontalGameTrackingDynamics:
    """6D horizontal tracking dynamics (reversed: ctrl min, dstb max)."""

    def __init__(self, config: GameConfig) -> None:
        self.k_x = config.defender.k_x
        self.k_y = config.defender.k_y
        self.u_d_h = config.defender.max_speed_horizontal
        self.u_a_h = config.attacker.max_speed_horizontal
        self.uMode = "min"
        self.dMode = "max"
        self.control_space = make_box(
            [-self.u_d_h, -self.u_d_h], [self.u_d_h, self.u_d_h],
        )
        self.disturbance_space = make_box(
            [-self.u_a_h, -self.u_a_h], [self.u_a_h, self.u_a_h],
        )

    def open_loop_dynamics(self, state, time=0.0):
        return np.array([
            state[2], state[3],
            -self.k_x * state[2], -self.k_y * state[3],
            0.0, 0.0,
        ])

    def opt_ctrl(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_u_x = hcl.scalar(self.u_d_h, "track_u_x")
        opt_u_y = hcl.scalar(self.u_d_h, "track_u_y")
        dummy_1 = hcl.scalar(0, "track_u_dummy_1")
        dummy_2 = hcl.scalar(0, "track_u_dummy_2")
        with hcl.if_(spat_deriv[2] * self.k_x > 0):
            opt_u_x[0] = -self.u_d_h
        with hcl.if_(spat_deriv[3] * self.k_y > 0):
            opt_u_y[0] = -self.u_d_h
        return (opt_u_x[0], opt_u_y[0], dummy_1[0], dummy_2[0])

    def opt_dstb(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_d_x = hcl.scalar(self.u_a_h, "track_d_x")
        opt_d_y = hcl.scalar(self.u_a_h, "track_d_y")
        dummy_1 = hcl.scalar(0, "track_d_dummy_1")
        dummy_2 = hcl.scalar(0, "track_d_dummy_2")
        with hcl.if_(spat_deriv[4] < 0):
            opt_d_x[0] = -self.u_a_h
        with hcl.if_(spat_deriv[5] < 0):
            opt_d_y[0] = -self.u_a_h
        return (opt_d_x[0], opt_d_y[0], dummy_1[0], dummy_2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        hcl = require_hcl()
        x_d_dot = hcl.scalar(0, "track_x_d_dot")
        y_d_dot = hcl.scalar(0, "track_y_d_dot")
        v_dx_dot = hcl.scalar(0, "track_v_dx_dot")
        v_dy_dot = hcl.scalar(0, "track_v_dy_dot")
        x_a_dot = hcl.scalar(0, "track_x_a_dot")
        y_a_dot = hcl.scalar(0, "track_y_a_dot")
        x_d_dot[0] = state[2]
        y_d_dot[0] = state[3]
        v_dx_dot[0] = self.k_x * (uOpt[0] - state[2])
        v_dy_dot[0] = self.k_y * (uOpt[1] - state[3])
        x_a_dot[0] = dOpt[0]
        y_a_dot[0] = dOpt[1]
        return (
            x_d_dot[0], y_d_dot[0], v_dx_dot[0],
            v_dy_dot[0], x_a_dot[0], y_a_dot[0],
        )
