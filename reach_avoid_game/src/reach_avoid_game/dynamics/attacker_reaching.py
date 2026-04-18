"""Attacker reaching dynamics for T_goal computation.

2D state: [x_A, y_A]
The attacker minimizes the value function (reaches target ASAP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reach_avoid_game.dynamics.odp_utils import make_box, require_hcl

if TYPE_CHECKING:
    from reach_avoid_game.config import GameConfig


class AttackerReachingDynamics:
    """2D attacker reaching dynamics."""

    def __init__(self, config: GameConfig) -> None:
        self.u_a_h = config.attacker.max_speed_horizontal
        self.uMode = "min"
        self.dMode = "max"
        self.control_space = make_box(
            [-self.u_a_h, -self.u_a_h], [self.u_a_h, self.u_a_h],
        )
        self.disturbance_space = make_box([0.0, 0.0], [0.0, 0.0])

    def open_loop_dynamics(self, state, time=0.0):
        return (0.0, 0.0)

    def opt_ctrl(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_u_x = hcl.scalar(self.u_a_h, "reach_u_x")
        opt_u_y = hcl.scalar(self.u_a_h, "reach_u_y")
        with hcl.if_(spat_deriv[0] > 0):
            opt_u_x[0] = -self.u_a_h
        with hcl.if_(spat_deriv[1] > 0):
            opt_u_y[0] = -self.u_a_h
        return (opt_u_x[0], opt_u_y[0])

    def opt_dstb(self, t, state, spat_deriv):
        hcl = require_hcl()
        d_x = hcl.scalar(0, "reach_d_x")
        d_y = hcl.scalar(0, "reach_d_y")
        return (d_x[0], d_y[0])

    def dynamics(self, t, state, uOpt, dOpt):
        hcl = require_hcl()
        x_dot = hcl.scalar(0, "reach_x_dot")
        y_dot = hcl.scalar(0, "reach_y_dot")
        x_dot[0] = uOpt[0]
        y_dot[0] = uOpt[1]
        return (x_dot[0], y_dot[0])
