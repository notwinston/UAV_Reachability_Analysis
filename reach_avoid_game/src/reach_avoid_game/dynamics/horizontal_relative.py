"""Horizontal relative dynamics for computing V_h_T.

4D state: [x_rel, y_rel, v_D_x, v_D_y]

For tracking: defender minimizes, attacker maximizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reach_avoid_game.dynamics.odp_utils import make_box, require_hcl

if TYPE_CHECKING:
    from reach_avoid_game.config import GameConfig


class HorizontalRelativeDynamics:
    """4D horizontal relative dynamics."""

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
        return (
            state[2], state[3],
            -self.k_x * state[2], -self.k_y * state[3],
        )

    def opt_ctrl(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_u_x = hcl.scalar(self.u_d_h, "rel_u_x")
        opt_u_y = hcl.scalar(self.u_d_h, "rel_u_y")
        dummy_1 = hcl.scalar(0, "rel_u_dummy_1")
        dummy_2 = hcl.scalar(0, "rel_u_dummy_2")
        with hcl.if_(spat_deriv[2] * self.k_x > 0):
            opt_u_x[0] = -self.u_d_h
        with hcl.if_(spat_deriv[3] * self.k_y > 0):
            opt_u_y[0] = -self.u_d_h
        return (opt_u_x[0], opt_u_y[0], dummy_1[0], dummy_2[0])

    def opt_dstb(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_d_x = hcl.scalar(self.u_a_h, "rel_d_x")
        opt_d_y = hcl.scalar(self.u_a_h, "rel_d_y")
        dummy_1 = hcl.scalar(0, "rel_d_dummy_1")
        dummy_2 = hcl.scalar(0, "rel_d_dummy_2")
        with hcl.if_(-spat_deriv[0] < 0):
            opt_d_x[0] = -self.u_a_h
        with hcl.if_(-spat_deriv[1] < 0):
            opt_d_y[0] = -self.u_a_h
        return (opt_d_x[0], opt_d_y[0], dummy_1[0], dummy_2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        hcl = require_hcl()
        x_rel_dot = hcl.scalar(0, "x_rel_dot")
        y_rel_dot = hcl.scalar(0, "y_rel_dot")
        v_dx_dot = hcl.scalar(0, "rel_v_dx_dot")
        v_dy_dot = hcl.scalar(0, "rel_v_dy_dot")
        x_rel_dot[0] = state[2] - dOpt[0]
        y_rel_dot[0] = state[3] - dOpt[1]
        v_dx_dot[0] = self.k_x * (uOpt[0] - state[2])
        v_dy_dot[0] = self.k_y * (uOpt[1] - state[3])
        return (x_rel_dot[0], y_rel_dot[0], v_dx_dot[0], v_dy_dot[0])
