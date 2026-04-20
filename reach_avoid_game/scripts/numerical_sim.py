"""Forward Euler simulation of reach-track control.

Implements:
- Algorithm 1 (vertical reach-track) from the paper
- Algorithm 2 (horizontal reach-track-avoid)
- Combined 3D simulation with both algorithms

The attacker uses optimal escape control.
"""

import argparse
import math
from pathlib import Path

import numpy as np

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.value_function_io import load_value_function
from reach_avoid_game.solvers.control_extraction import (
    interpolate_value,
    compute_gradient,
    extract_optimal_control_vertical,
    extract_optimal_disturbance_vertical,
    is_deep_inside_invariant_set,
)


GRADIENT_DEADBAND = 1e-6
VERTICAL_PID_GAIN = 8.0
HORIZONTAL_PID_GAIN = 2.0


def _clamp_horizontal_speed(cmd_x, cmd_y, speed_limit):
    """Clamp a horizontal command to the configured speed magnitude."""
    speed = math.hypot(cmd_x, cmd_y)
    if speed <= speed_limit or speed < 1e-12:
        return float(cmd_x), float(cmd_y)
    scale = speed_limit / speed
    return float(cmd_x * scale), float(cmd_y * scale)


def _is_closing_horizontal_gap(cmd_x, cmd_y, x_rel, y_rel):
    """Return True when defender horizontal command reduces defender-attacker gap."""
    return (x_rel * cmd_x + y_rel * cmd_y) < -GRADIENT_DEADBAND


def _is_closing_vertical_gap(cmd_z, z_rel):
    """Return True when defender vertical command reduces defender-attacker gap."""
    return (z_rel * cmd_z) < -GRADIENT_DEADBAND


def _apply_wall_avoidance(cmd, pos, vel, config):
    """Apply wall-avoidance safety layer to defender control commands.

    Scales down commands toward nearby walls using dynamic stopping-distance
    margins. Actively pushes away when very close (< 1m) to a wall.
    """
    bounds_min = [config.room.x_min, config.room.y_min, config.room.z_min]
    bounds_max = [config.room.x_max, config.room.y_max, config.room.z_max]
    k_gains = [config.defender.k_x, config.defender.k_y, config.defender.k_z]
    u_max = [config.defender.max_speed_horizontal, config.defender.max_speed_horizontal,
             config.defender.max_speed_vertical]
    result = list(cmd)
    for i in range(3):
        v = vel[i]
        p = pos[i]
        if k_gains[i] > 1e-6:
            d_stop = abs(v) / k_gains[i] * (1.0 - math.exp(-1.0))
        else:
            d_stop = 0.0
        margin = max(d_stop * 2.0, 1.5)
        dist_min = p - bounds_min[i]
        dist_max = bounds_max[i] - p
        if dist_min < margin and (v < 0 or p < 1.0):
            scale = max(0.0, dist_min / margin)
            if result[i] < 0:
                result[i] *= scale
            if dist_min < 1.0:
                result[i] = max(result[i], u_max[i] * 0.5)
        if dist_max < margin and (v > 0 or p > bounds_max[i] - 1.0):
            scale = max(0.0, dist_max / margin)
            if result[i] > 0:
                result[i] *= scale
            if dist_max < 1.0:
                result[i] = min(result[i], -u_max[i] * 0.5)
    return result


def _inside_obstacle_xy(x, y, config):
    for obs in config.obstacles:
        if obs.x_min < x < obs.x_max and obs.y_min < y < obs.y_max:
            return obs
    return None


def _apply_obstacle_avoidance_xy(cmd_x, cmd_y, x, y, config):
    """Project a horizontal command away from nearby box obstacles."""
    result_x, result_y = float(cmd_x), float(cmd_y)
    margin = 1.0
    for obs in config.obstacles:
        if not (obs.x_min - margin <= x <= obs.x_max + margin and obs.y_min - margin <= y <= obs.y_max + margin):
            continue
        distances = {
            "left": abs(x - obs.x_min),
            "right": abs(x - obs.x_max),
            "bottom": abs(y - obs.y_min),
            "top": abs(y - obs.y_max),
        }
        side = min(distances, key=distances.get)
        if side == "left" and result_x > 0:
            result_x = 0.0
        elif side == "right" and result_x < 0:
            result_x = 0.0
        elif side == "bottom" and result_y > 0:
            result_y = 0.0
        elif side == "top" and result_y < 0:
            result_y = 0.0
    return result_x, result_y


def _clamp_out_of_obstacle_xy(x, y, vx, vy, config):
    """If a point entered a box obstacle, move it to the nearest face."""
    obs = _inside_obstacle_xy(x, y, config)
    if obs is None:
        return x, y, vx, vy
    distances = {
        "left": abs(x - obs.x_min),
        "right": abs(x - obs.x_max),
        "bottom": abs(y - obs.y_min),
        "top": abs(y - obs.y_max),
    }
    side = min(distances, key=distances.get)
    if side == "left":
        return obs.x_min, y, min(vx, 0.0), vy
    if side == "right":
        return obs.x_max, y, max(vx, 0.0), vy
    if side == "bottom":
        return x, obs.y_min, vx, min(vy, 0.0)
    return x, obs.y_max, vx, max(vy, 0.0)


def trajectory_hits_obstacle(traj, config) -> bool:
    """Return True if any defender or attacker horizontal sample is in an obstacle."""
    for prefix in [("x_d", "y_d"), ("x_a", "y_a")]:
        if prefix[0] not in traj:
            continue
        for x, y in zip(traj[prefix[0]], traj[prefix[1]]):
            if _inside_obstacle_xy(float(x), float(y), config) is not None:
                return True
    return False


def run_vertical_sim(
    config: GameConfig,
    vf_dir: str,
    dt: float = 0.01,
    T: float = 10.0,
    check_only: bool = False,
) -> dict:
    """Run vertical reach-track simulation (Algorithm 1).

    Args:
        config: Game configuration
        vf_dir: Directory containing value function files
        dt: Time step
        T: Total simulation time
        check_only: If True, just verify loading and run 1 step

    Returns:
        Dictionary with trajectory data and capture status
    """
    vf_dir = Path(vf_dir)

    phi_z_data = load_value_function(vf_dir / "phi_z.npz")
    v_z_inf_data = load_value_function(vf_dir / "V_z_inf.npz")
    b_z_data = load_value_function(vf_dir / "B_z.npz")

    d_z = config.capture.d_z
    k_z = config.defender.k_z
    u_d_z = config.defender.max_speed_vertical
    u_a_z = config.attacker.max_speed_vertical

    pid_gain = VERTICAL_PID_GAIN
    margin_fraction = 0.3
    b_z_params = b_z_data.params
    b_z_valid = not (
        isinstance(b_z_params, dict)
        and float(b_z_params.get("d_z_effective", d_z)) > d_z + 1e-9
    )

    z_d = 8.0
    v_d_z = 0.0
    z_a = 3.0

    n_steps = int(T / dt)
    if check_only:
        n_steps = 1

    traj_z_d = np.zeros(n_steps + 1)
    traj_v_d_z = np.zeros(n_steps + 1)
    traj_z_a = np.zeros(n_steps + 1)
    traj_u_z = np.zeros(n_steps)
    traj_d_z = np.zeros(n_steps)
    traj_mode = np.zeros(n_steps, dtype=int)

    traj_z_d[0] = z_d
    traj_v_d_z[0] = v_d_z
    traj_z_a[0] = z_a

    captured = False

    for step in range(n_steps):
        state_3d = np.array([z_d, v_d_z, z_a])
        z_rel = z_d - z_a
        state_2d = np.array([z_rel, v_d_z])

        state_3d_clamped = np.clip(state_3d, phi_z_data.grid_min, phi_z_data.grid_max)
        state_2d_clamped = np.clip(state_2d, v_z_inf_data.grid_min, v_z_inf_data.grid_max)

        b_z_val = interpolate_value(b_z_data, state_2d_clamped) if b_z_valid else 0.0
        in_b_z = b_z_val > 0.5

        deep_inside = is_deep_inside_invariant_set(v_z_inf_data, state_2d_clamped, d_z, margin_fraction)

        if not in_b_z:
            u_z = extract_optimal_control_vertical(phi_z_data, state_3d_clamped, k_z, u_d_z)
            mode = 0
            if not _is_closing_vertical_gap(u_z, z_rel):
                u_z = np.clip(pid_gain * (z_a - z_d), -u_d_z, u_d_z)
                mode = 2
        elif deep_inside:
            z_error = z_a - z_d
            u_z = np.clip(pid_gain * z_error, -u_d_z, u_d_z)
            mode = 2
        else:
            u_z = extract_optimal_control_vertical(v_z_inf_data, state_2d_clamped, k_z, u_d_z)
            mode = 1
            if not _is_closing_vertical_gap(u_z, z_rel):
                u_z = np.clip(pid_gain * (z_a - z_d), -u_d_z, u_d_z)
                mode = 2

        d_z_ctrl = extract_optimal_disturbance_vertical(phi_z_data, state_3d_clamped, u_a_z)

        traj_u_z[step] = u_z
        traj_d_z[step] = d_z_ctrl
        traj_mode[step] = mode

        z_d_new = z_d + dt * v_d_z
        v_d_z_new = v_d_z + dt * k_z * (u_z - v_d_z)
        z_a_new = z_a + dt * d_z_ctrl

        z_d = np.clip(z_d_new, config.room.z_min, config.room.z_max)
        v_d_z = v_d_z_new
        z_a = np.clip(z_a_new, config.room.z_min, config.room.z_max)

        traj_z_d[step + 1] = z_d
        traj_v_d_z[step + 1] = v_d_z
        traj_z_a[step + 1] = z_a

        if abs(z_d - z_a) <= d_z:
            captured = True

    return {
        "z_d": traj_z_d[:n_steps + 1],
        "v_d_z": traj_v_d_z[:n_steps + 1],
        "z_a": traj_z_a[:n_steps + 1],
        "u_z": traj_u_z[:n_steps],
        "d_z": traj_d_z[:n_steps],
        "mode": traj_mode[:n_steps],
        "captured": captured,
        "dt": dt,
        "T": T,
    }


def _extract_horizontal_control(
    phi_h_data, v_h_t_data,
    state_h, state_rel,
    k_x, k_y, u_d_h, d_h, margin_fraction=0.3,
):
    """Extract horizontal defender control using Algorithm 2.

    Returns (u_x, u_y, mode) where mode is 0=reach, 1=track_boundary, 2=pid.
    """
    state_h_clamped = np.clip(state_h, phi_h_data.grid_min, phi_h_data.grid_max)
    x_rel, y_rel = state_rel[0], state_rel[1]
    pid_x = np.clip(-HORIZONTAL_PID_GAIN * x_rel, -u_d_h, u_d_h)
    pid_y = np.clip(-HORIZONTAL_PID_GAIN * y_rel, -u_d_h, u_d_h)
    pid_x, pid_y = _clamp_horizontal_speed(pid_x, pid_y, u_d_h)

    phi_h_val = interpolate_value(phi_h_data, state_h_clamped)
    in_winning = phi_h_val > 0

    if v_h_t_data.values.ndim != 6 or float(np.nanmin(v_h_t_data.values)) > d_h:
        return pid_x, pid_y, 2

    state_h_tracking = np.clip(state_h, v_h_t_data.grid_min, v_h_t_data.grid_max)
    v_h_val = interpolate_value(v_h_t_data, state_h_tracking)
    in_b_h = v_h_val <= d_h

    if not in_b_h:
        if in_winning:
            # Mode 0: optimal reaching control from paper Phi_h gradient.
            grad = compute_gradient(phi_h_data, state_h_clamped)
            # Defender maximizes Phi_h: u direction = sign of gradient wrt velocity dims.
            direction_x = grad[2] * k_x  # dPhi_h/dv_Dx * k_x
            direction_y = grad[3] * k_y  # dPhi_h/dv_Dy * k_y
            u_x = u_d_h if direction_x > 0 else -u_d_h
            u_y = u_d_h if direction_y > 0 else -u_d_h
            if abs(direction_x) < GRADIENT_DEADBAND:
                u_x = 0.0
            if abs(direction_y) < GRADIENT_DEADBAND:
                u_y = 0.0
            u_x, u_y = _clamp_horizontal_speed(u_x, u_y, u_d_h)
            if not _is_closing_horizontal_gap(u_x, u_y, x_rel, y_rel):
                return pid_x, pid_y, 2
            return u_x, u_y, 0

        # Outside the defender-winning horizontal set: pursue with PID fallback.
        return pid_x, pid_y, 2
    else:
        # Check if deep inside B_h
        deep_inside = v_h_val < d_h * (1 - margin_fraction)

        if deep_inside:
            # Mode 2: PID tracking
            return pid_x, pid_y, 2
        else:
            # Mode 1: optimal tracking from V_h_T gradient
            grad = compute_gradient(v_h_t_data, state_h_tracking)
            # Defender minimizes (tracking): u direction = -sign of gradient wrt velocity
            direction_x = grad[2] * k_x
            direction_y = grad[3] * k_y
            u_x = -u_d_h if direction_x > 0 else u_d_h
            u_y = -u_d_h if direction_y > 0 else u_d_h
            if abs(direction_x) < GRADIENT_DEADBAND:
                u_x = 0.0
            if abs(direction_y) < GRADIENT_DEADBAND:
                u_y = 0.0
            u_x, u_y = _clamp_horizontal_speed(u_x, u_y, u_d_h)
            if not _is_closing_horizontal_gap(u_x, u_y, x_rel, y_rel):
                return pid_x, pid_y, 2
            return u_x, u_y, 1


def _extract_horizontal_disturbance(phi_h_data, state_h, u_a_h):
    """Extract attacker optimal horizontal disturbance.

    Attacker minimizes value function.
    """
    state_h_clamped = np.clip(state_h, phi_h_data.grid_min, phi_h_data.grid_max)
    grad = compute_gradient(phi_h_data, state_h_clamped)
    # Attacker minimizes: d direction = -sign of gradient wrt attacker position
    direction_x = grad[4]  # dV/dx_A
    direction_y = grad[5]  # dV/dy_A
    if abs(direction_x) < GRADIENT_DEADBAND:
        d_x = 0.0
    else:
        d_x = -u_a_h if direction_x > 0 else u_a_h
    if abs(direction_y) < GRADIENT_DEADBAND:
        d_y = 0.0
    else:
        d_y = -u_a_h if direction_y > 0 else u_a_h
    return d_x, d_y


def run_combined_sim(
    config: GameConfig,
    vf_dir: str,
    dt: float = 0.01,
    T: float = 10.0,
    check_only: bool = False,
    initial_defender_pos=None,
    initial_defender_vel=None,
    initial_attacker_pos=None,
) -> dict:
    """Run combined 3D simulation with Algorithm 1 + Algorithm 2.

    Simulates both vertical and horizontal sub-games simultaneously.

    Args:
        config: Game configuration
        vf_dir: Directory containing all value function files
        dt: Time step
        T: Total simulation time
        check_only: If True, just verify loading and run 1 step
        initial_defender_pos: Optional [x, y, z] defender start.
        initial_defender_vel: Optional [vx, vy, vz] defender initial velocity.
        initial_attacker_pos: Optional [x, y, z] attacker start.

    Returns:
        Dictionary with full 3D trajectory data
    """
    vf_dir = Path(vf_dir)

    # Load all value functions
    phi_z_data = load_value_function(vf_dir / "phi_z.npz")
    v_z_inf_data = load_value_function(vf_dir / "V_z_inf.npz")
    b_z_data = load_value_function(vf_dir / "B_z.npz")
    phi_h_data = load_value_function(vf_dir / "phi_h.npz")
    v_h_t_data = load_value_function(vf_dir / "V_h_T_6d.npz")

    k_x, k_y, k_z = config.defender.k_x, config.defender.k_y, config.defender.k_z
    u_d_h = config.defender.max_speed_horizontal
    u_d_z = config.defender.max_speed_vertical
    u_a_h = config.attacker.max_speed_horizontal
    u_a_z = config.attacker.max_speed_vertical
    d_z = config.capture.d_z
    d_h = config.capture.d_h

    b_z_params = b_z_data.params
    b_z_valid = not (
        isinstance(b_z_params, dict)
        and float(b_z_params.get("d_z_effective", d_z)) > d_z + 1e-9
    )

    # Initial states
    # Defender: upper-right near target, attacker: lower-left behind obstacle
    if initial_defender_pos is None:
        initial_defender_pos = [35.0, 20.0, 8.0]
    if initial_defender_vel is None:
        initial_defender_vel = [0.0, 0.0, 0.0]
    if initial_attacker_pos is None:
        initial_attacker_pos = [5.0, 3.0, 3.0]

    x_d, y_d, z_d = map(float, initial_defender_pos)
    v_dx, v_dy, v_dz = map(float, initial_defender_vel)
    # Attacker: [x_A, y_A, z_A]
    x_a, y_a, z_a = map(float, initial_attacker_pos)

    n_steps = int(T / dt)
    if check_only:
        n_steps = 1

    # Trajectory storage
    traj = {
        "x_d": np.zeros(n_steps + 1), "y_d": np.zeros(n_steps + 1), "z_d": np.zeros(n_steps + 1),
        "v_dx": np.zeros(n_steps + 1), "v_dy": np.zeros(n_steps + 1), "v_dz": np.zeros(n_steps + 1),
        "x_a": np.zeros(n_steps + 1), "y_a": np.zeros(n_steps + 1), "z_a": np.zeros(n_steps + 1),
        "u_x": np.zeros(n_steps), "u_y": np.zeros(n_steps), "u_z": np.zeros(n_steps),
        "d_x": np.zeros(n_steps), "d_y": np.zeros(n_steps), "d_z_ctrl": np.zeros(n_steps),
        "mode_z": np.zeros(n_steps, dtype=int),
        "mode_h": np.zeros(n_steps, dtype=int),
    }

    traj["x_d"][0], traj["y_d"][0], traj["z_d"][0] = x_d, y_d, z_d
    traj["v_dx"][0], traj["v_dy"][0], traj["v_dz"][0] = v_dx, v_dy, v_dz
    traj["x_a"][0], traj["y_a"][0], traj["z_a"][0] = x_a, y_a, z_a

    captured_h = False
    captured_z = False
    captured_3d = False
    attacker_reached_target = False
    first_capture_step = None
    first_target_step = None

    for step in range(n_steps):
        # Vertical states
        state_z_3d = np.array([z_d, v_dz, z_a])
        z_rel = z_d - z_a
        state_z_2d = np.array([z_rel, v_dz])

        state_z_3d_c = np.clip(state_z_3d, phi_z_data.grid_min, phi_z_data.grid_max)
        state_z_2d_c = np.clip(state_z_2d, v_z_inf_data.grid_min, v_z_inf_data.grid_max)

        # Horizontal states
        state_h = np.array([x_d, y_d, v_dx, v_dy, x_a, y_a])
        x_rel, y_rel = x_d - x_a, y_d - y_a
        state_h_rel = np.array([x_rel, y_rel, v_dx, v_dy])

        # --- Vertical control (Algorithm 1) ---
        b_z_val = interpolate_value(b_z_data, state_z_2d_c) if b_z_valid else 0.0
        in_b_z = b_z_val > 0.5
        deep_inside_z = is_deep_inside_invariant_set(v_z_inf_data, state_z_2d_c, d_z, 0.3)

        if not in_b_z:
            u_z = extract_optimal_control_vertical(phi_z_data, state_z_3d_c, k_z, u_d_z)
            mode_z = 0
            if not _is_closing_vertical_gap(u_z, z_rel):
                u_z = np.clip(VERTICAL_PID_GAIN * (z_a - z_d), -u_d_z, u_d_z)
                mode_z = 2
        elif deep_inside_z:
            u_z = np.clip(VERTICAL_PID_GAIN * (z_a - z_d), -u_d_z, u_d_z)
            mode_z = 2
        else:
            u_z = extract_optimal_control_vertical(v_z_inf_data, state_z_2d_c, k_z, u_d_z)
            mode_z = 1
            if not _is_closing_vertical_gap(u_z, z_rel):
                u_z = np.clip(VERTICAL_PID_GAIN * (z_a - z_d), -u_d_z, u_d_z)
                mode_z = 2

        d_z_ctrl = extract_optimal_disturbance_vertical(phi_z_data, state_z_3d_c, u_a_z)

        # --- Horizontal control (Algorithm 2) ---
        u_x, u_y, mode_h = _extract_horizontal_control(
            phi_h_data, v_h_t_data,
            state_h, state_h_rel,
            k_x, k_y, u_d_h, d_h,
        )
        d_x, d_y = _extract_horizontal_disturbance(phi_h_data, state_h, u_a_h)

        # Apply wall-avoidance safety layer to defender controls
        wa = _apply_wall_avoidance(
            [u_x, u_y, u_z], [x_d, y_d, z_d], [v_dx, v_dy, v_dz], config,
        )
        u_x, u_y, u_z = wa[0], wa[1], wa[2]
        u_x, u_y = _apply_obstacle_avoidance_xy(u_x, u_y, x_d, y_d, config)
        d_x, d_y = _apply_obstacle_avoidance_xy(d_x, d_y, x_a, y_a, config)

        # Store
        traj["u_x"][step], traj["u_y"][step], traj["u_z"][step] = u_x, u_y, u_z
        traj["d_x"][step], traj["d_y"][step], traj["d_z_ctrl"][step] = d_x, d_y, d_z_ctrl
        traj["mode_z"][step] = mode_z
        traj["mode_h"][step] = mode_h

        # Forward Euler integration
        x_d += dt * v_dx
        y_d += dt * v_dy
        z_d += dt * v_dz
        v_dx += dt * k_x * (u_x - v_dx)
        v_dy += dt * k_y * (u_y - v_dy)
        v_dz += dt * k_z * (u_z - v_dz)

        x_a += dt * d_x
        y_a += dt * d_y
        z_a += dt * d_z_ctrl

        # Clamp to room bounds
        x_d = np.clip(x_d, config.room.x_min, config.room.x_max)
        y_d = np.clip(y_d, config.room.y_min, config.room.y_max)
        z_d = np.clip(z_d, config.room.z_min, config.room.z_max)
        x_a = np.clip(x_a, config.room.x_min, config.room.x_max)
        y_a = np.clip(y_a, config.room.y_min, config.room.y_max)
        z_a = np.clip(z_a, config.room.z_min, config.room.z_max)
        x_d, y_d, v_dx, v_dy = _clamp_out_of_obstacle_xy(x_d, y_d, v_dx, v_dy, config)
        x_a, y_a, d_x, d_y = _clamp_out_of_obstacle_xy(x_a, y_a, d_x, d_y, config)

        # Zero wall-normal velocity on wall contact
        for pos_val, vel_ref, lo, hi in [
            (x_d, 'v_dx', config.room.x_min, config.room.x_max),
            (y_d, 'v_dy', config.room.y_min, config.room.y_max),
            (z_d, 'v_dz', config.room.z_min, config.room.z_max),
        ]:
            if vel_ref == 'v_dx':
                if pos_val <= lo and v_dx < 0: v_dx = 0.0
                if pos_val >= hi and v_dx > 0: v_dx = 0.0
            elif vel_ref == 'v_dy':
                if pos_val <= lo and v_dy < 0: v_dy = 0.0
                if pos_val >= hi and v_dy > 0: v_dy = 0.0
            elif vel_ref == 'v_dz':
                if pos_val <= lo and v_dz < 0: v_dz = 0.0
                if pos_val >= hi and v_dz > 0: v_dz = 0.0

        traj["x_d"][step + 1], traj["y_d"][step + 1], traj["z_d"][step + 1] = x_d, y_d, z_d
        traj["v_dx"][step + 1], traj["v_dy"][step + 1], traj["v_dz"][step + 1] = v_dx, v_dy, v_dz
        traj["x_a"][step + 1], traj["y_a"][step + 1], traj["z_a"][step + 1] = x_a, y_a, z_a

        # Check captures
        h_dist = np.sqrt((x_d - x_a)**2 + (y_d - y_a)**2)
        z_dist = abs(z_d - z_a)

        if h_dist <= d_h:
            captured_h = True
        if z_dist <= d_z:
            captured_z = True
        if h_dist <= d_h and z_dist <= d_z:
            if not captured_3d:
                first_capture_step = step
            captured_3d = True

        tr = config.target_region
        in_target = (tr.x_min <= x_a <= tr.x_max and tr.y_min <= y_a <= tr.y_max)
        if in_target:
            if not attacker_reached_target:
                first_target_step = step
            attacker_reached_target = True

    # Determine outcome: first event wins
    if first_capture_step is not None and first_target_step is not None:
        if first_capture_step <= first_target_step:
            outcome = "Defender wins (attacker captured)"
        else:
            outcome = "Attacker wins (reached target)"
    elif first_capture_step is not None:
        outcome = "Defender wins (attacker captured)"
    elif first_target_step is not None:
        outcome = "Attacker wins (reached target)"
    else:
        outcome = "Timeout — no capture, target not reached"

    traj["captured_h"] = captured_h
    traj["captured_z"] = captured_z
    traj["captured_3d"] = captured_3d
    traj["attacker_reached_target"] = attacker_reached_target
    traj["first_capture_step"] = first_capture_step
    traj["first_target_step"] = first_target_step
    traj["outcome"] = outcome
    traj["obstacle_violation"] = trajectory_hits_obstacle(traj, config)
    traj["dt"] = dt
    traj["T"] = T

    return traj


def main():
    parser = argparse.ArgumentParser(description="Reach-track numerical simulation")
    parser.add_argument("--mode", default="vertical", choices=["vertical", "combined"],
                        help="Simulation mode (default: vertical)")
    parser.add_argument("--value-function-dir", default="data/value_functions",
                        help="Directory containing value function .npz files")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (default: 0.01)")
    parser.add_argument("--T", type=float, default=10.0, help="Simulation time (default: 10s)")
    parser.add_argument("--check-only", action="store_true",
                        help="Just verify loading and run 1 step")
    parser.add_argument("--config", default="config/game_params.yaml",
                        help="Path to game configuration YAML")
    parser.add_argument("--output-dir", default="data/simulations",
                        help="Directory for saved simulation .npz files")
    args = parser.parse_args()

    config = GameConfig.from_yaml(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "vertical":
        result = run_vertical_sim(
            config=config, vf_dir=args.value_function_dir,
            dt=args.dt, T=args.T, check_only=args.check_only,
        )

        if args.check_only:
            print("Check-only mode: loaded value functions and ran 1 step successfully.")
            return

        output_path = output_dir / "vertical_sim.npz"
        np.savez(output_path,
                 z_d=result["z_d"], v_d_z=result["v_d_z"], z_a=result["z_a"],
                 u_z=result["u_z"], d_z_ctrl=result["d_z"], mode=result["mode"],
                 dt=result["dt"], T=result["T"])
        print(f"Trajectory saved to {output_path}")

        if result["captured"]:
            z_diff = np.abs(result["z_d"] - result["z_a"])
            capture_idx = np.argmax(z_diff <= config.capture.d_z)
            print(f"CAPTURE at t={capture_idx * result['dt']:.2f}s")
        else:
            print("NO CAPTURE")
        z_diff = np.abs(result["z_d"] - result["z_a"])
        print(f"Final |z_D - z_A| = {z_diff[-1]:.3f}m")

    elif args.mode == "combined":
        result = run_combined_sim(
            config=config, vf_dir=args.value_function_dir,
            dt=args.dt, T=args.T, check_only=args.check_only,
        )

        if args.check_only:
            print("Check-only mode: loaded all value functions and ran 1 step successfully.")
            return

        output_path = output_dir / "combined_sim.npz"
        save_keys = [
            "x_d", "y_d", "z_d", "v_dx", "v_dy", "v_dz",
            "x_a", "y_a", "z_a",
            "u_x", "u_y", "u_z", "d_x", "d_y", "d_z_ctrl",
            "mode_z", "mode_h",
        ]
        save_data = {k: result[k] for k in save_keys}
        save_data["dt"] = result["dt"]
        save_data["T"] = result["T"]
        np.savez(output_path, **save_data)
        print(f"Trajectory saved to {output_path}")

        h_dist = np.sqrt((result["x_d"] - result["x_a"])**2 + (result["y_d"] - result["y_a"])**2)
        z_dist = np.abs(result["z_d"] - result["z_a"])

        print(f"Horizontal capture: {result['captured_h']}")
        print(f"Vertical capture: {result['captured_z']}")
        print(f"3D capture: {result['captured_3d']}")
        print(f"Final horizontal distance: {h_dist[-1]:.3f}m")
        print(f"Final vertical distance: {z_dist[-1]:.3f}m")
        print(f"Min horizontal distance: {h_dist.min():.3f}m at t={np.argmin(h_dist) * result['dt']:.2f}s")
        print(f"Min vertical distance: {z_dist.min():.3f}m at t={np.argmin(z_dist) * result['dt']:.2f}s")


if __name__ == "__main__":
    main()
