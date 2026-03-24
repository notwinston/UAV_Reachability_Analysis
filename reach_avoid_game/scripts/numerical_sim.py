"""Forward Euler simulation of reach-track control.

Implements:
- Algorithm 1 (vertical reach-track) from the paper
- Algorithm 2 (horizontal reach-track-avoid)
- Combined 3D simulation with both algorithms

The attacker uses optimal escape control.
"""

import argparse
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

    pid_gain = 2.0
    margin_fraction = 0.3
    b_z_params = b_z_data.params
    d_z_effective = b_z_params.get("d_z_effective", d_z) if isinstance(b_z_params, dict) else d_z

    z_d = 5.0
    v_d_z = 0.0
    z_a = 2.0

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

        b_z_val = interpolate_value(b_z_data, state_2d_clamped)
        in_b_z = b_z_val > 0.5

        deep_inside = is_deep_inside_invariant_set(v_z_inf_data, state_2d_clamped, d_z_effective, margin_fraction)

        if not in_b_z:
            u_z = extract_optimal_control_vertical(phi_z_data, state_3d_clamped, k_z, u_d_z)
            mode = 0
        elif deep_inside:
            z_error = z_a - z_d
            u_z = np.clip(pid_gain * z_error, -u_d_z, u_d_z)
            mode = 2
        else:
            u_z = extract_optimal_control_vertical(v_z_inf_data, state_2d_clamped, k_z, u_d_z)
            mode = 1

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
    phi_h_data, v_h_t_data, b_h_data,
    state_h, state_rel,
    k_x, k_y, u_d_h, margin_fraction=0.3,
):
    """Extract horizontal defender control using Algorithm 2.

    Returns (u_x, u_y, mode) where mode is 0=reach, 1=track_boundary, 2=pid.
    """
    state_h_clamped = np.clip(state_h, phi_h_data.grid_min, phi_h_data.grid_max)
    state_rel_clamped = np.clip(state_rel, v_h_t_data.grid_min, v_h_t_data.grid_max)

    # Check if in B_h
    b_h_val = interpolate_value(b_h_data, state_rel_clamped)
    in_b_h = b_h_val > 0.5

    b_h_params = b_h_data.params
    d_h_effective = b_h_params.get("d_h_effective", 3.0) if isinstance(b_h_params, dict) else 3.0

    if not in_b_h:
        # Mode 0: optimal reaching control from phi_h gradient
        grad = compute_gradient(phi_h_data, state_h_clamped)
        # Defender maximizes: u direction = sign of gradient wrt velocity dims
        direction_x = grad[2] * k_x  # dV/dv_Dx * k_x
        direction_y = grad[3] * k_y  # dV/dv_Dy * k_y
        u_x = u_d_h if direction_x >= 0 else -u_d_h
        u_y = u_d_h if direction_y >= 0 else -u_d_h
        return u_x, u_y, 0
    else:
        # Check if deep inside B_h
        v_h_val = interpolate_value(v_h_t_data, state_rel_clamped)
        deep_inside = v_h_val < d_h_effective * (1 - margin_fraction)

        if deep_inside:
            # Mode 2: PID tracking
            x_rel, y_rel = state_rel[0], state_rel[1]
            u_x = np.clip(-2.0 * x_rel, -u_d_h, u_d_h)
            u_y = np.clip(-2.0 * y_rel, -u_d_h, u_d_h)
            return u_x, u_y, 2
        else:
            # Mode 1: optimal tracking from V_h_T gradient
            grad = compute_gradient(v_h_t_data, state_rel_clamped)
            # Defender minimizes (tracking): u direction = -sign of gradient wrt velocity
            direction_x = grad[2] * k_x
            direction_y = grad[3] * k_y
            u_x = -u_d_h if direction_x >= 0 else u_d_h
            u_y = -u_d_h if direction_y >= 0 else u_d_h
            return u_x, u_y, 1


def _attacker_waypoint(x_a, y_a, target_center, obstacles, margin=2.0):
    """Return the immediate waypoint for the attacker, routing around obstacles.

    Phase 1 — head for the near corner of the wall gap (clears y-range).
    Phase 2 — once y is outside the wall, proceed past it in x.
    """
    tx, ty = target_center
    for obs in obstacles:
        if tx <= obs.x_max or x_a > obs.x_max + margin:
            continue

        dist_top = abs(y_a - obs.y_max)
        dist_bot = abs(y_a - obs.y_min)
        if dist_top < dist_bot:
            gap_y = obs.y_max + margin
            y_cleared = y_a > obs.y_max
        else:
            gap_y = obs.y_min - margin
            y_cleared = y_a < obs.y_min

        if y_cleared:
            return (obs.x_max + margin, gap_y)
        return (obs.x_min, gap_y)
    return target_center


def _extract_horizontal_disturbance(phi_h_data, state_h, u_a_h,
                                    target_center=None, obstacles=()):
    """Extract attacker horizontal control as goal-seeking toward target.

    The attacker's objective is to reach the target region, not merely
    to evade the defender (which is what minimising phi_h would do).
    When a target_center is provided the attacker moves at max speed
    toward it, routing around obstacles; otherwise falls back to the
    phi_h gradient.
    """
    x_a, y_a = state_h[4], state_h[5]

    if target_center is not None:
        wp = _attacker_waypoint(x_a, y_a, target_center, obstacles)
        dx = wp[0] - x_a
        dy = wp[1] - y_a
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 1e-6:
            d_x = (dx / dist) * u_a_h
            d_y = (dy / dist) * u_a_h
        else:
            d_x, d_y = 0.0, 0.0
        return d_x, d_y

    state_h_clamped = np.clip(state_h, phi_h_data.grid_min, phi_h_data.grid_max)
    grad = compute_gradient(phi_h_data, state_h_clamped)
    d_x = -u_a_h if grad[4] >= 0 else u_a_h
    d_y = -u_a_h if grad[5] >= 0 else u_a_h
    return d_x, d_y


def run_combined_sim(
    config: GameConfig,
    vf_dir: str,
    dt: float = 0.01,
    T: float = 10.0,
    check_only: bool = False,
) -> dict:
    """Run combined 3D simulation with Algorithm 1 + Algorithm 2.

    Simulates both vertical and horizontal sub-games simultaneously.

    Args:
        config: Game configuration
        vf_dir: Directory containing all value function files
        dt: Time step
        T: Total simulation time
        check_only: If True, just verify loading and run 1 step

    Returns:
        Dictionary with full 3D trajectory data
    """
    vf_dir = Path(vf_dir)

    # Load all value functions
    phi_z_data = load_value_function(vf_dir / "phi_z.npz")
    v_z_inf_data = load_value_function(vf_dir / "V_z_inf.npz")
    b_z_data = load_value_function(vf_dir / "B_z.npz")
    phi_h_data = load_value_function(vf_dir / "phi_h.npz")
    v_h_t_data = load_value_function(vf_dir / "V_h_T.npz")
    b_h_data = load_value_function(vf_dir / "B_h.npz")

    k_x, k_y, k_z = config.defender.k_x, config.defender.k_y, config.defender.k_z
    u_d_h = config.defender.max_speed_horizontal
    u_d_z = config.defender.max_speed_vertical
    u_a_h = config.attacker.max_speed_horizontal
    u_a_z = config.attacker.max_speed_vertical
    d_z = config.capture.d_z
    d_h = config.capture.d_h

    b_z_params = b_z_data.params
    d_z_effective = b_z_params.get("d_z_effective", d_z) if isinstance(b_z_params, dict) else d_z

    # Initial states
    # Defender: [x_D, y_D, z_D, v_Dx, v_Dy, v_Dz]
    x_d, y_d, z_d = 10.0, 12.0, 5.0
    v_dx, v_dy, v_dz = 0.0, 0.0, 0.0
    # Attacker: [x_A, y_A, z_A] — starts behind the obstacle wall
    x_a, y_a, z_a = 5.0, 12.0, 2.0

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

    target_center = None
    if hasattr(config, 'target_region') and config.target_region is not None:
        tr = config.target_region
        target_center = ((tr.x_min + tr.x_max) / 2, (tr.y_min + tr.y_max) / 2)

    captured_h = False
    captured_z = False
    captured_3d = False

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
        b_z_val = interpolate_value(b_z_data, state_z_2d_c)
        in_b_z = b_z_val > 0.5
        deep_inside_z = is_deep_inside_invariant_set(v_z_inf_data, state_z_2d_c, d_z_effective, 0.3)

        if not in_b_z:
            u_z = extract_optimal_control_vertical(phi_z_data, state_z_3d_c, k_z, u_d_z)
            mode_z = 0
        elif deep_inside_z:
            u_z = np.clip(2.0 * (z_a - z_d), -u_d_z, u_d_z)
            mode_z = 2
        else:
            u_z = extract_optimal_control_vertical(v_z_inf_data, state_z_2d_c, k_z, u_d_z)
            mode_z = 1

        d_z_ctrl = extract_optimal_disturbance_vertical(phi_z_data, state_z_3d_c, u_a_z)

        # --- Horizontal control (Algorithm 2) ---
        u_x, u_y, mode_h = _extract_horizontal_control(
            phi_h_data, v_h_t_data, b_h_data,
            state_h, state_h_rel,
            k_x, k_y, u_d_h,
        )
        d_x, d_y = _extract_horizontal_disturbance(
            phi_h_data, state_h, u_a_h, target_center, config.obstacles)

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

        # Enforce obstacle collisions: revert position if inside any obstacle
        for obs in config.obstacles:
            if obs.x_min <= x_d <= obs.x_max and obs.y_min <= y_d <= obs.y_max:
                x_d, y_d = traj["x_d"][step], traj["y_d"][step]
                v_dx, v_dy = 0.0, 0.0
            if obs.x_min <= x_a <= obs.x_max and obs.y_min <= y_a <= obs.y_max:
                x_a, y_a = traj["x_a"][step], traj["y_a"][step]

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
            captured_3d = True

    traj["captured_h"] = captured_h
    traj["captured_z"] = captured_z
    traj["captured_3d"] = captured_3d
    traj["dt"] = dt
    traj["T"] = T

    return traj


def main():
    parser = argparse.ArgumentParser(description="Reach-track numerical simulation")
    parser.add_argument("--mode", default="vertical", choices=["vertical", "combined"],
                        help="Simulation mode (default: vertical)")
    parser.add_argument("--value-function-dir", default="/workspace/data/value_functions/",
                        help="Directory containing value function .npz files")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (default: 0.01)")
    parser.add_argument("--T", type=float, default=10.0, help="Simulation time (default: 10s)")
    parser.add_argument("--check-only", action="store_true",
                        help="Just verify loading and run 1 step")
    parser.add_argument("--config", default="/workspace/config/game_params.yaml",
                        help="Path to game configuration YAML")
    args = parser.parse_args()

    config = GameConfig.from_yaml(args.config)

    output_dir = Path("/workspace/data/simulations")
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
