"""Vertical relative dynamics for computing V_z_inf.

2D state: [z_rel, v_D_z] where z_rel = z_D - z_A

For tracking: defender minimizes, attacker maximizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reach_avoid_game.dynamics.odp_utils import make_box, require_hcl

if TYPE_CHECKING:
    from reach_avoid_game.config import GameConfig


class VerticalRelativeDynamics:
    """2D vertical relative dynamics."""

    def __init__(self, config: GameConfig) -> None:
        self.k_z = config.defender.k_z
        self.u_d_z = config.defender.max_speed_vertical
        self.u_a_z = config.attacker.max_speed_vertical
        self.uMode = "min"
        self.dMode = "max"
        self.control_space = make_box([-self.u_d_z], [self.u_d_z])
        self.disturbance_space = make_box([-self.u_a_z], [self.u_a_z])

    def open_loop_dynamics(self, state, time=0.0):
        return (state[1], -self.k_z * state[1])

    def opt_ctrl(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_u = hcl.scalar(self.u_d_z, "rel_u_z")
        dummy = hcl.scalar(0, "rel_u_z_dummy")
        with hcl.if_(spat_deriv[1] * self.k_z > 0):
            opt_u[0] = -self.u_d_z
        return (opt_u[0], dummy[0])

    def opt_dstb(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_d = hcl.scalar(self.u_a_z, "rel_d_z")
        dummy = hcl.scalar(0, "rel_d_z_dummy")
        with hcl.if_(-spat_deriv[0] < 0):
            opt_d[0] = -self.u_a_z
        return (opt_d[0], dummy[0])

    def dynamics(self, t, state, uOpt, dOpt):
        hcl = require_hcl()
        z_rel_dot = hcl.scalar(0, "z_rel_dot")
        v_dz_dot = hcl.scalar(0, "rel_v_dz_dot")
        z_rel_dot[0] = state[1] - dOpt[0]
        v_dz_dot[0] = self.k_z * (uOpt[0] - state[1])
        return (z_rel_dot[0], v_dz_dot[0])
