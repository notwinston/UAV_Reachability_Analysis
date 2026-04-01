"""Vertical sub-game dynamics for HJ reachability solver.

3D state: [z_D, v_D_z, z_A]
Dynamics (Eq. 25 in paper):
  z_D_dot   = v_D_z
  v_D_z_dot = k_z * (u_z - v_D_z)
  z_A_dot   = d_z

Two-player game:
  - Defender (control) maximizes value (tries to capture)
  - Attacker (disturbance) minimizes value (tries to escape)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from reach_avoid_game.config import GameConfig


class VerticalGameDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """3D vertical sub-game dynamics.

    Implements both hj_reachability interface (for solver) and
    OptimizedDP-style pure-Python methods (for online control).
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
        return jnp.array([
            state[1],
            -self.k_z * state[1],
            0.0,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0],
            [self.k_z],
            [0.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.0],
            [0.0],
            [1.0],
        ])

    # --- OptimizedDP-compatible pure-Python methods (for online control) ---

    def opt_ctrl_numpy(self, state, spat_deriv):
        """Optimal control (NumPy). u_z = U_D_z * sign(dV/dv_Dz * k_z)."""
        coeff = spat_deriv[1] * self.k_z
        return np.where(coeff >= 0, self.u_d_z, -self.u_d_z)

    def opt_dstb_numpy(self, state, spat_deriv):
        """Optimal disturbance (NumPy). d_z = -U_A_z * sign(dV/dz_A)."""
        return np.where(spat_deriv[2] >= 0, -self.u_a_z, self.u_a_z)

    def dynamics_numpy(self, state, u_z, d_z):
        """State derivatives (NumPy)."""
        return (state[1], self.k_z * (u_z - state[1]), d_z)

    # --- odp.dynamics.VerticalGame3D-compatible interface ---

    def optCtrl_inPython(self, state, spat_deriv):
        """Optimal defender control (odp-style scalar). u_z = U_Dz * sign(dV/dv_Dz * k_z)."""
        coeff = spat_deriv[1] * self.k_z
        return float(self.u_d_z if coeff >= 0 else -self.u_d_z)

    def optDstb_inPython(self, state, spat_deriv):
        """Optimal attacker disturbance (odp-style scalar). d_z = -U_Az * sign(dV/dz_A)."""
        return float(-self.u_a_z if spat_deriv[2] >= 0 else self.u_a_z)

    def dynamics_inPython(self, state, u_z, d_z):
        """State derivatives (pure Python tuple)."""
        return (state[1], self.k_z * (u_z - state[1]), d_z)
