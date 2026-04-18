"""Vertical sub-game dynamics for OptimizedDP.

3D state: [z_D, v_D_z, z_A]
Dynamics (Eq. 25 in paper):
  z_D_dot   = v_D_z
  v_D_z_dot = k_z * (u_z - v_D_z)
  z_A_dot   = d_z

Two-player game:
  - Defender (control) minimizes value (tries to capture)
  - Attacker (disturbance) maximizes value (tries to escape)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reach_avoid_game.dynamics.odp_utils import make_box, require_hcl

if TYPE_CHECKING:
    from reach_avoid_game.config import GameConfig


class VerticalGameDynamics:
    """3D vertical sub-game dynamics.

    Implements the OptimizedDP solver interface and pure-Python helpers for
    tests / online control.
    """

    def __init__(self, config: GameConfig) -> None:
        self.k_z = config.defender.k_z
        self.u_d_z = config.defender.max_speed_vertical
        self.u_a_z = config.attacker.max_speed_vertical
        self.uMode = "min"
        self.dMode = "max"
        self.control_space = make_box([-self.u_d_z], [self.u_d_z])
        self.disturbance_space = make_box([-self.u_a_z], [self.u_a_z])

    def open_loop_dynamics(self, state, time=0.0):
        return np.array([state[1], -self.k_z * state[1], 0.0])

    def __call__(self, state, control, disturbance, time=0.0):
        return np.array(self.dynamics_inPython(state, control[0], disturbance[0]))

    def optimal_control(self, state, time, spat_deriv):
        return np.array([self.optCtrl_inPython(state, spat_deriv)])

    def optimal_disturbance(self, state, time, spat_deriv):
        return np.array([self.optDstb_inPython(state, spat_deriv)])

    def opt_ctrl(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_u = hcl.scalar(self.u_d_z, "opt_u_z")
        dummy_1 = hcl.scalar(0, "opt_u_z_dummy_1")
        dummy_2 = hcl.scalar(0, "opt_u_z_dummy_2")
        with hcl.if_(spat_deriv[1] * self.k_z > 0):
            opt_u[0] = -self.u_d_z
        return (opt_u[0], dummy_1[0], dummy_2[0])

    def opt_dstb(self, t, state, spat_deriv):
        hcl = require_hcl()
        opt_d = hcl.scalar(-self.u_a_z, "opt_d_z")
        dummy_1 = hcl.scalar(0, "opt_d_z_dummy_1")
        dummy_2 = hcl.scalar(0, "opt_d_z_dummy_2")
        with hcl.if_(spat_deriv[2] > 0):
            opt_d[0] = self.u_a_z
        return (opt_d[0], dummy_1[0], dummy_2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        hcl = require_hcl()
        z_d_dot = hcl.scalar(0, "z_d_dot")
        v_dz_dot = hcl.scalar(0, "v_dz_dot")
        z_a_dot = hcl.scalar(0, "z_a_dot")
        z_d_dot[0] = state[1]
        v_dz_dot[0] = self.k_z * (uOpt[0] - state[1])
        z_a_dot[0] = dOpt[0]
        return (z_d_dot[0], v_dz_dot[0], z_a_dot[0])

    # --- OptimizedDP-compatible pure-Python methods (for online control) ---

    def opt_ctrl_numpy(self, state, spat_deriv):
        """Optimal defender control (NumPy), minimizing the vertical value."""
        coeff = spat_deriv[1] * self.k_z
        return np.where(coeff > 0, -self.u_d_z, self.u_d_z)

    def opt_dstb_numpy(self, state, spat_deriv):
        """Optimal attacker disturbance (NumPy), maximizing the vertical value."""
        return np.where(spat_deriv[2] > 0, self.u_a_z, -self.u_a_z)

    def dynamics_numpy(self, state, u_z, d_z):
        """State derivatives (NumPy)."""
        return (state[1], self.k_z * (u_z - state[1]), d_z)

    # --- odp.dynamics.VerticalGame3D-compatible interface ---

    def optCtrl_inPython(self, state, spat_deriv):
        """Optimal defender control (odp-style scalar), minimizing the value."""
        coeff = spat_deriv[1] * self.k_z
        return float(-self.u_d_z if coeff > 0 else self.u_d_z)

    def optDstb_inPython(self, state, spat_deriv):
        """Optimal attacker disturbance (odp-style scalar), maximizing the value."""
        return float(self.u_a_z if spat_deriv[2] > 0 else -self.u_a_z)

    def dynamics_inPython(self, state, u_z, d_z):
        """State derivatives (pure Python tuple)."""
        return (state[1], self.k_z * (u_z - state[1]), d_z)
