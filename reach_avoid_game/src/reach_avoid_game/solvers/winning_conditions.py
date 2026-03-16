"""Winning condition analysis for the reach-avoid game.

Implements the capture guarantee analysis from Section VI of the paper:
- Proposition 1: Attacker wins if T_goal <= T_capture
- Theorem: Defender wins if x_z in W_{D,z}, T_goal > T_capture, x_h in W_{D,h}

Key functions:
- compute_T_goal: earliest time attacker reaches target
- compute_T_capture: earliest time defender captures vertically
- check_defender_wins: combined winning condition check
- get_winning_regions: extract W_D and W_A from value function level sets
"""

from __future__ import annotations

import numpy as np

from reach_avoid_game.solvers.value_function_io import ValueFunctionData
from reach_avoid_game.solvers.control_extraction import interpolate_value


def compute_T_goal(
    phi_a_reach: ValueFunctionData,
    attacker_pos: np.ndarray,
    time_horizon: float = 10.0,
) -> float:
    """Compute T_goal: earliest time attacker can reach target region.

    Uses the attacker reaching value function. If phi_A_reach(x_A) <= 0,
    the attacker is already inside the target or can reach it within the
    time horizon. T_goal is estimated from the value function magnitude.

    Args:
        phi_a_reach: Attacker reaching value function (2D: [x_A, y_A])
        attacker_pos: Attacker position [x_A, y_A]
        time_horizon: Solver time horizon used for phi_A_reach

    Returns:
        Estimated T_goal in seconds. Returns inf if attacker cannot reach target.
    """
    value = interpolate_value(phi_a_reach, attacker_pos)

    if value <= 0:
        # Attacker can reach target. Estimate time from value.
        # The value function's magnitude gives a rough time estimate:
        # states that become reachable earlier have more negative values.
        # T_goal approx proportional to how close to zero the value is.
        params = phi_a_reach.params if isinstance(phi_a_reach.params, dict) else {}
        u_a_h = params.get("U_A_h", 3.0)
        # Value represents signed distance at final time; convert to time estimate
        # using attacker speed. A value of -x means the state was reachable
        # for about x/U_A_h seconds before the horizon.
        t_remaining = abs(value) / u_a_h
        return max(0.0, time_horizon - t_remaining)
    else:
        return float("inf")


def compute_T_capture(
    phi_z: ValueFunctionData,
    state_z: np.ndarray,
    d_z: float = 1.0,
    time_horizon: float = 15.0,
) -> float:
    """Compute T_capture: earliest time defender captures attacker vertically.

    Uses the vertical reach-avoid value function. If phi_z(x_z) <= 0,
    vertical capture is possible within the time horizon.

    Args:
        phi_z: Vertical reach-avoid value function (3D: [z_D, v_D_z, z_A])
        state_z: Vertical state [z_D, v_D_z, z_A]
        d_z: Vertical capture distance
        time_horizon: Solver time horizon used for phi_z

    Returns:
        Estimated T_capture in seconds. Returns inf if capture not possible.
    """
    value = interpolate_value(phi_z, state_z)

    if value <= 0:
        # Capture is possible. Estimate time from value magnitude.
        # More negative value means captured earlier.
        t_remaining = abs(value) / d_z
        return max(0.0, time_horizon - t_remaining)
    else:
        return float("inf")


def check_defender_wins(
    phi_h: ValueFunctionData,
    phi_z: ValueFunctionData,
    b_z: ValueFunctionData,
    state_h: np.ndarray,
    state_z: np.ndarray,
    phi_a_reach: ValueFunctionData | None = None,
    attacker_pos: np.ndarray | None = None,
) -> dict:
    """Check if defender is guaranteed to win from given states.

    Implements the paper's Theorem:
    Defender wins if:
    1. x_h in W_{D,h} (phi_h(x_h) <= 0 from defender's perspective)
    2. x_z in W_{D,z} (phi_z(x_z) <= 0 from defender's perspective)
    3. T_goal > T_capture (defender captures vertically before attacker reaches goal)

    Note: The value function convention from hj_reachability is that
    the value function level set {x : V(x) <= 0} represents the
    set of states from which the target can be reached (capture possible).

    Args:
        phi_h: Horizontal reach-avoid value function (6D)
        phi_z: Vertical reach-avoid value function (3D)
        b_z: Vertical invariant capture set B_z (2D: [z_rel, v_D_z])
        state_h: Horizontal state [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
        state_z: Vertical state [z_D, v_D_z, z_A]
        phi_a_reach: Attacker reaching value function (optional, for T_goal)
        attacker_pos: Attacker position [x_A, y_A] (required if phi_a_reach given)

    Returns:
        Dictionary with:
        - defender_wins: bool
        - in_W_D_h: bool (horizontal winning region)
        - in_W_D_z: bool (vertical winning region)
        - in_B_z: bool (vertical invariant set)
        - phi_h_value: float
        - phi_z_value: float
        - T_goal: float (if phi_a_reach provided)
        - T_capture: float (if phi_a_reach provided)
    """
    # Check horizontal winning region
    phi_h_val = interpolate_value(phi_h, state_h)
    in_w_d_h = phi_h_val <= 0  # Defender can capture horizontally

    # Check vertical winning region
    phi_z_val = interpolate_value(phi_z, state_z)
    in_w_d_z = phi_z_val <= 0  # Defender can capture vertically

    # Check vertical invariant set
    z_rel = state_z[0] - state_z[2]  # z_D - z_A
    v_d_z = state_z[1]
    state_2d = np.array([z_rel, v_d_z])
    state_2d_clamped = np.clip(state_2d, b_z.grid_min, b_z.grid_max)
    b_z_val = interpolate_value(b_z, state_2d_clamped)
    in_b_z = b_z_val > 0.5

    result = {
        "in_W_D_h": bool(in_w_d_h),
        "in_W_D_z": bool(in_w_d_z),
        "in_B_z": bool(in_b_z),
        "phi_h_value": float(phi_h_val),
        "phi_z_value": float(phi_z_val),
    }

    # Basic condition: defender wins if in both winning regions
    defender_wins = in_w_d_h and in_w_d_z

    # If timing info available, check T_goal > T_capture condition
    if phi_a_reach is not None and attacker_pos is not None:
        params_h = phi_h.params if isinstance(phi_h.params, dict) else {}
        params_z = phi_z.params if isinstance(phi_z.params, dict) else {}
        params_a = phi_a_reach.params if isinstance(phi_a_reach.params, dict) else {}

        t_goal = compute_T_goal(
            phi_a_reach, attacker_pos,
            time_horizon=params_a.get("time_horizon", 10.0),
        )
        t_capture = compute_T_capture(
            phi_z, state_z,
            time_horizon=params_z.get("time_horizon", 15.0),
        )

        result["T_goal"] = t_goal
        result["T_capture"] = t_capture

        # Full theorem: defender wins if in both winning regions AND captures before goal
        defender_wins = in_w_d_h and in_w_d_z and (t_goal > t_capture)

    result["defender_wins"] = defender_wins
    return result


def get_winning_regions(
    phi_data: ValueFunctionData,
) -> dict:
    """Extract winning regions from a value function.

    W_D (defender wins) = {x : phi(x) <= 0}  (can capture from here)
    W_A (attacker wins) = {x : phi(x) > 0}   (attacker escapes from here)

    Args:
        phi_data: Value function data (phi_h or phi_z)

    Returns:
        Dictionary with:
        - W_D_mask: boolean array (True where defender wins)
        - W_A_mask: boolean array (True where attacker wins)
        - W_D_fraction: fraction of grid in W_D
        - W_A_fraction: fraction of grid in W_A
    """
    w_d_mask = phi_data.values <= 0
    w_a_mask = phi_data.values > 0

    return {
        "W_D_mask": w_d_mask,
        "W_A_mask": w_a_mask,
        "W_D_fraction": float(np.mean(w_d_mask)),
        "W_A_fraction": float(np.mean(w_a_mask)),
    }
