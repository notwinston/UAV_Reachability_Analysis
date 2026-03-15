"""Optimal control extraction from value functions via gradient computation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from reach_avoid_game.solvers.value_function_io import ValueFunctionData


def _build_interpolator(vf_data: ValueFunctionData) -> RegularGridInterpolator:
    """Build a RegularGridInterpolator from a value function."""
    ndim = vf_data.values.ndim
    axes = []
    for i in range(ndim):
        axes.append(np.linspace(
            float(vf_data.grid_min[i]),
            float(vf_data.grid_max[i]),
            vf_data.values.shape[i],
        ))
    return RegularGridInterpolator(
        tuple(axes),
        vf_data.values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def interpolate_value(vf_data: ValueFunctionData, state: np.ndarray) -> float:
    """Interpolate value function at a given state.

    Args:
        vf_data: Value function data
        state: State vector matching the grid dimensions

    Returns:
        Interpolated value
    """
    interp = _build_interpolator(vf_data)
    return float(interp(state.reshape(1, -1))[0])


def compute_gradient(vf_data: ValueFunctionData, state: np.ndarray) -> np.ndarray:
    """Compute the spatial gradient of the value function at a state via central finite differences.

    Args:
        vf_data: Value function data
        state: State vector matching the grid dimensions

    Returns:
        Gradient vector (same dimensionality as state)
    """
    interp = _build_interpolator(vf_data)
    ndim = len(state)
    grad = np.zeros(ndim)

    # Compute grid spacing for each dimension
    spacings = []
    for i in range(ndim):
        spacing = (float(vf_data.grid_max[i]) - float(vf_data.grid_min[i])) / (vf_data.values.shape[i] - 1)
        spacings.append(spacing)

    for i in range(ndim):
        h = spacings[i]
        state_plus = state.copy()
        state_minus = state.copy()
        state_plus[i] += h
        state_minus[i] -= h

        v_plus = float(interp(state_plus.reshape(1, -1))[0])
        v_minus = float(interp(state_minus.reshape(1, -1))[0])
        grad[i] = (v_plus - v_minus) / (2 * h)

    return grad


def extract_optimal_control_vertical(
    vf_data: ValueFunctionData,
    state: np.ndarray,
    k_z: float,
    u_d_z: float,
) -> float:
    """Extract optimal vertical control from value function gradient.

    For the vertical game (defender maximizes):
    u_z = U_D_z * sign(dV/dv_Dz * k_z)

    Args:
        vf_data: Value function data (Phi_z or V_z_inf)
        state: State vector [z_D, v_D_z, z_A] (3D) or [z_rel, v_D_z] (2D)
        k_z: Vertical proportional gain
        u_d_z: Defender max vertical speed

    Returns:
        Optimal u_z control value
    """
    grad = compute_gradient(vf_data, state)

    # Index of v_D_z in the state
    v_idx = 1  # Same for both 3D [z_D, v_D_z, z_A] and 2D [z_rel, v_D_z]

    # Defender maximizes: u_z = U_D_z * sign(dV/dv_Dz * k_z)
    direction = grad[v_idx] * k_z
    if direction >= 0:
        return u_d_z
    else:
        return -u_d_z


def extract_optimal_disturbance_vertical(
    vf_data: ValueFunctionData,
    state: np.ndarray,
    u_a_z: float,
) -> float:
    """Extract optimal attacker disturbance from value function gradient.

    For the 3D vertical game (attacker minimizes):
    d_z = -U_A_z * sign(dV/dz_A)

    For the 2D relative dynamics (attacker maximizes to escape):
    d_z = U_A_z * sign(-dV/dz_rel)  (since z_rel_dot -= d_z)

    Args:
        vf_data: Value function data
        state: State vector [z_D, v_D_z, z_A] (3D) or [z_rel, v_D_z] (2D)
        u_a_z: Attacker max vertical speed

    Returns:
        Optimal d_z disturbance value
    """
    grad = compute_gradient(vf_data, state)

    if len(state) == 3:
        # 3D game: attacker minimizes, d_z = -U_A_z * sign(dV/dz_A)
        direction = grad[2]
        if direction >= 0:
            return -u_a_z
        else:
            return u_a_z
    else:
        # 2D relative: attacker maximizes distance, disturbance Jacobian is [-1, 0]
        # So attacker direction = grad @ [-1, 0] = -grad[0]
        # Attacker maximizes → d_z = U_A_z * sign(-grad[0])
        direction = -grad[0]
        if direction >= 0:
            return u_a_z
        else:
            return -u_a_z


def is_deep_inside_invariant_set(
    vf_data: ValueFunctionData,
    state: np.ndarray,
    d_z: float,
    margin_fraction: float = 0.3,
) -> bool:
    """Check if state is deep inside the invariant set B_z.

    "Deep inside" means value_function(state) < -margin where
    margin = margin_fraction * d_z.

    For V_z_inf initialized as |z_rel| - d_z:
    - value <= 0 means inside B_z
    - value < -margin means deep inside

    Args:
        vf_data: V_z_inf value function data
        state: State vector
        d_z: Capture distance
        margin_fraction: Fraction of d_z for deep-inside threshold

    Returns:
        True if deep inside B_z
    """
    value = interpolate_value(vf_data, state)
    margin = margin_fraction * d_z
    return value < -margin
