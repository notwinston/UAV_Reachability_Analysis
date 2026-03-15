"""Forward Euler simulation of vertical reach-track control.

Implements Algorithm 1 (vertical reach-track) from the paper:
1. If NOT in B_z: apply optimal reaching control from Phi_z gradient
2. If in B_z but near boundary: apply optimal tracking from V_z_inf gradient
3. If deep inside B_z: apply PID tracking

The attacker uses optimal escape control.
"""

import argparse
from pathlib import Path

import numpy as np

from reach_avoid_game.config import GameConfig
from reach_avoid_game.dynamics.defender import DefenderDynamics
from reach_avoid_game.dynamics.attacker import AttackerDynamics
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
    """Run vertical reach-track simulation.

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

    # Load value functions
    phi_z_data = load_value_function(vf_dir / "phi_z.npz")
    v_z_inf_data = load_value_function(vf_dir / "V_z_inf.npz")
    b_z_data = load_value_function(vf_dir / "B_z.npz")

    d_z = config.capture.d_z
    k_z = config.defender.k_z
    u_d_z = config.defender.max_speed_vertical
    u_a_z = config.attacker.max_speed_vertical

    # PID tracking gain
    pid_gain = 2.0

    # Deep-inside margin
    margin_fraction = 0.3
    # Use effective d_z from B_z params if available
    b_z_params = b_z_data.params
    d_z_effective = b_z_params.get("d_z_effective", d_z) if isinstance(b_z_params, dict) else d_z

    # Initialize states
    # Defender: z_D=5m, v_D_z=0
    z_d = 5.0
    v_d_z = 0.0
    # Attacker: z_A=2m
    z_a = 2.0

    n_steps = int(T / dt)
    if check_only:
        n_steps = 1

    # Trajectory storage
    traj_z_d = np.zeros(n_steps + 1)
    traj_v_d_z = np.zeros(n_steps + 1)
    traj_z_a = np.zeros(n_steps + 1)
    traj_u_z = np.zeros(n_steps)
    traj_d_z = np.zeros(n_steps)
    traj_mode = np.zeros(n_steps, dtype=int)  # 0=reach, 1=track_boundary, 2=pid

    traj_z_d[0] = z_d
    traj_v_d_z[0] = v_d_z
    traj_z_a[0] = z_a

    captured = False

    for step in range(n_steps):
        # Current state for value function lookups
        state_3d = np.array([z_d, v_d_z, z_a])
        z_rel = z_d - z_a
        state_2d = np.array([z_rel, v_d_z])

        # Clamp states to grid bounds for interpolation
        state_3d_clamped = np.clip(state_3d, phi_z_data.grid_min, phi_z_data.grid_max)
        state_2d_clamped = np.clip(state_2d, v_z_inf_data.grid_min, v_z_inf_data.grid_max)

        # Check if in B_z (invariant set)
        b_z_val = interpolate_value(b_z_data, state_2d_clamped)
        in_b_z = b_z_val > 0.5  # B_z is stored as 1.0/0.0 mask

        # Check if deep inside B_z
        deep_inside = is_deep_inside_invariant_set(v_z_inf_data, state_2d_clamped, d_z_effective, margin_fraction)

        # Algorithm 1: Vertical Reach-Track
        if not in_b_z:
            # Mode 0: Not in B_z — apply optimal reaching control from Phi_z
            u_z = extract_optimal_control_vertical(phi_z_data, state_3d_clamped, k_z, u_d_z)
            mode = 0
        elif deep_inside:
            # Mode 2: Deep inside B_z — PID tracking
            # Target: z_D should track z_A
            z_error = z_a - z_d
            u_z = np.clip(pid_gain * z_error, -u_d_z, u_d_z)
            mode = 2
        else:
            # Mode 1: In B_z but near boundary — optimal tracking from V_z_inf
            u_z = extract_optimal_control_vertical(v_z_inf_data, state_2d_clamped, k_z, u_d_z)
            mode = 1

        # Attacker uses optimal escape control from Phi_z
        d_z_ctrl = extract_optimal_disturbance_vertical(phi_z_data, state_3d_clamped, u_a_z)

        # Store control
        traj_u_z[step] = u_z
        traj_d_z[step] = d_z_ctrl
        traj_mode[step] = mode

        # Forward Euler integration
        # Defender: z_D_dot = v_D_z, v_D_z_dot = k_z * (u_z - v_D_z)
        z_d_new = z_d + dt * v_d_z
        v_d_z_new = v_d_z + dt * k_z * (u_z - v_d_z)
        # Attacker: z_A_dot = d_z
        z_a_new = z_a + dt * d_z_ctrl

        # Clamp to room bounds
        z_d = np.clip(z_d_new, config.room.z_min, config.room.z_max)
        v_d_z = v_d_z_new
        z_a = np.clip(z_a_new, config.room.z_min, config.room.z_max)

        traj_z_d[step + 1] = z_d
        traj_v_d_z[step + 1] = v_d_z
        traj_z_a[step + 1] = z_a

        # Check capture
        if abs(z_d - z_a) <= d_z:
            captured = True
            if not check_only:
                # Continue simulation to show tracking behavior
                pass

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


def main():
    parser = argparse.ArgumentParser(description="Vertical reach-track numerical simulation")
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

    result = run_vertical_sim(
        config=config,
        vf_dir=args.value_function_dir,
        dt=args.dt,
        T=args.T,
        check_only=args.check_only,
    )

    if args.check_only:
        print("Check-only mode: loaded value functions and ran 1 step successfully.")
        return

    # Save trajectory
    output_dir = Path("/workspace/data/simulations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vertical_sim.npz"
    np.savez(
        output_path,
        z_d=result["z_d"],
        v_d_z=result["v_d_z"],
        z_a=result["z_a"],
        u_z=result["u_z"],
        d_z_ctrl=result["d_z"],
        mode=result["mode"],
        dt=result["dt"],
        T=result["T"],
    )
    print(f"Trajectory saved to {output_path}")

    # Report
    if result["captured"]:
        # Find first capture time
        z_diff = np.abs(result["z_d"] - result["z_a"])
        capture_idx = np.argmax(z_diff <= config.capture.d_z)
        capture_time = capture_idx * result["dt"]
        print(f"CAPTURE at t={capture_time:.2f}s (|z_D - z_A| <= {config.capture.d_z}m)")
    else:
        print("NO CAPTURE")

    # Print summary
    z_diff = np.abs(result["z_d"] - result["z_a"])
    print(f"Final |z_D - z_A| = {z_diff[-1]:.3f}m")
    print(f"Min |z_D - z_A| = {z_diff.min():.3f}m at t={np.argmin(z_diff) * result['dt']:.2f}s")


if __name__ == "__main__":
    main()
