#!/usr/bin/env python3
"""Offline closed-loop simulation of the reach-avoid game.

Uses the REAL value functions and defender control logic (DefenderControlLogic)
plus the attacker's scripted waypoint controller to simulate drone trajectories
without Gazebo/ROS2/PX4.

Models first-order velocity dynamics with drag:
  v_dot = k * (u - v)    (same as PX4 velocity controller model)

Outputs:
  - Terminal table of timestamped positions
  - Trajectory plot saved to data/plots/drone_tracker.png
  - Game outcome (CAPTURED / ATTACKER_REACHED / TIMEOUT)
"""

import math
import os
import sys

import numpy as np
import yaml

# Import the real control logic
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'reach_avoid_ws', 'src', 'reach_avoid_controller'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'reach_avoid_game', 'src'))

from reach_avoid_controller.value_function_loader import ValueFunctionLoader
from reach_avoid_controller.defender_node import DefenderControlLogic

# ── Configuration ──────────────────────────────────────────────────────────
VF_DIR = "/workspace/data/value_functions/"
PARAMS_FILE = "/workspace/config/game_params.yaml"
DT = 0.02            # 50 Hz control loop
T_MAX = 15.0         # max simulation time (seconds)
PLOT_PATH = "/workspace/data/plots/drone_tracker.png"

# Load game params
with open(PARAMS_FILE) as f:
    gp = yaml.safe_load(f)

# Dynamics drag coefficients (from game_params)
K_X = gp["defender"]["k_x"]
K_Y = gp["defender"]["k_y"]
K_Z = gp["defender"]["k_z"]

U_D_H = gp["defender"]["max_speed_horizontal"]
U_D_Z = gp["defender"]["max_speed_vertical"]
U_A_H = gp["attacker"]["max_speed_horizontal"]
U_A_Z = gp["attacker"]["max_speed_vertical"]

D_H = gp["capture"]["d_h"]
D_Z = gp["capture"]["d_z"]

# Target region
TARGET = gp["target_region"]
TARGET_CENTER = np.array([
    (TARGET["x_min"] + TARGET["x_max"]) / 2,
    (TARGET["y_min"] + TARGET["y_max"]) / 2,
])

# Obstacle
OBS = gp["obstacles"][0]

# Room bounds for clamping
ROOM = gp["room"]

# ── Initial conditions ──
# Defender starts 4m behind attacker (just outside d_h=3), same altitude.
# Attacker navigates around obstacle toward target. Tests:
# - Vertical: z_mode should be tracking/pid_deep (same altitude → inside B_z)
# - Horizontal: h_mode should transition reaching → tracking → pid_deep as gap closes
DEFENDER_POS_0 = np.array([10.0, 12.5, 10.0])     # [x, y, z]
DEFENDER_VEL_0 = np.array([0.0, 0.0, 0.0])        # [vx, vy, vz]

# x_rel = 10-12.5 = -2.5, y_rel = 12.5-14.5 = -2.0
# h_dist = sqrt(6.25+4.0) = 3.2 → outside capture but inside B_h grid [-3,3]
# z_rel = -0.5 → inside B_z
# Both components within B_h grid range → should start in tracking mode
ATTACKER_POS_0 = np.array([12.5, 14.5, 10.5])     # [x, y, z]
ATTACKER_VEL_0 = np.array([3.0, -2.0, 0.0])       # evading fast

T_MAX = 25.0

# ── Attacker scripted controller (waypoints toward target) ─────────────
# Attacker tries to break away — short evasive hops
ATTACKER_WAYPOINTS = [
    np.array([14.0, 4.0, 10.5]),    # dodge below obstacle
    np.array([22.0, 3.0, 11.0]),    # past obstacle
    np.array([25.0, 10.0, 10.0]),   # back toward center
    np.array([32.0, 12.5, 10.0]),   # approach target
    np.array([41.5, 12.5, 10.0]),   # target center
]
ATTACKER_SPEED_FRAC = 1.0   # attacker at MAX speed
WAYPOINT_RADIUS = 1.5


def attacker_control(pos, wp_idx):
    """Scripted attacker: navigate to waypoint, advance when close."""
    if wp_idx >= len(ATTACKER_WAYPOINTS):
        wp_idx = len(ATTACKER_WAYPOINTS) - 1
    wp = ATTACKER_WAYPOINTS[wp_idx]
    diff = wp - pos
    dist = np.linalg.norm(diff)
    if dist < WAYPOINT_RADIUS and wp_idx < len(ATTACKER_WAYPOINTS) - 1:
        wp_idx += 1
        wp = ATTACKER_WAYPOINTS[wp_idx]
        diff = wp - pos
        dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros(3), wp_idx
    direction = diff / dist
    speed = U_A_H * ATTACKER_SPEED_FRAC
    # Split into horizontal and vertical
    h_dir = diff[:2]
    h_dist = np.linalg.norm(h_dir)
    v_dir = diff[2]
    u = np.zeros(3)
    if h_dist > 0.01:
        u[:2] = (h_dir / h_dist) * min(speed, h_dist / DT)
    u[2] = np.clip(v_dir * 2.0, -U_A_Z, U_A_Z)
    # Clamp total horizontal speed
    h_speed = np.linalg.norm(u[:2])
    if h_speed > U_A_H:
        u[:2] *= U_A_H / h_speed
    return u, wp_idx


def step_dynamics(pos, vel, u_cmd, k_xy, k_z, dt):
    """First-order velocity dynamics: v_dot = k * (u - v), pos_dot = v.

    Clamps position to room boundaries.
    """
    vel_new = vel.copy()
    vel_new[0] += k_xy * (u_cmd[0] - vel[0]) * dt
    vel_new[1] += k_xy * (u_cmd[1] - vel[1]) * dt
    vel_new[2] += k_z * (u_cmd[2] - vel[2]) * dt
    pos_new = pos + vel_new * dt
    # Clamp to room and zero velocity at walls
    for i, (lo, hi) in enumerate([(ROOM["x_min"], ROOM["x_max"]),
                                   (ROOM["y_min"], ROOM["y_max"]),
                                   (ROOM["z_min"], ROOM["z_max"])]):
        if pos_new[i] < lo:
            pos_new[i] = lo
            vel_new[i] = max(vel_new[i], 0.0)
        elif pos_new[i] > hi:
            pos_new[i] = hi
            vel_new[i] = min(vel_new[i], 0.0)
    return pos_new, vel_new


def in_target(pos_2d):
    """Check if position is inside target region."""
    return (TARGET["x_min"] <= pos_2d[0] <= TARGET["x_max"] and
            TARGET["y_min"] <= pos_2d[1] <= TARGET["y_max"])


def is_captured(d_pos, a_pos):
    """Check capture condition."""
    h_dist = np.linalg.norm(d_pos[:2] - a_pos[:2])
    z_dist = abs(d_pos[2] - a_pos[2])
    return h_dist <= D_H and z_dist <= D_Z


# ── Main simulation ────────────────────────────────────────────────────
def run_simulation():
    # Load value functions
    print("Loading value functions...")
    loader = ValueFunctionLoader(VF_DIR)
    print(f"  Loaded: {loader.loaded_names}")
    assert loader.all_loaded, f"Missing VFs: {set(['phi_z','V_z_inf','B_z','phi_h','V_h_T','B_h','phi_A_reach']) - set(loader.loaded_names)}"

    logic = DefenderControlLogic(loader)
    print(f"  Game params: d_h={logic.d_h}, d_z={logic.d_z}, U_D_h={logic.U_D_h}, U_D_z={logic.U_D_z}")

    # State
    d_pos = DEFENDER_POS_0.copy()
    d_vel = DEFENDER_VEL_0.copy()
    a_pos = ATTACKER_POS_0.copy()
    a_vel = ATTACKER_VEL_0.copy()
    wp_idx = 0

    # History
    t_hist = []
    d_pos_hist = []
    a_pos_hist = []
    d_vel_hist = []
    a_vel_hist = []
    h_dist_hist = []
    z_dist_hist = []
    z_mode_hist = []
    h_mode_hist = []
    status_hist = []

    outcome = "TIMEOUT"
    n_steps = int(T_MAX / DT)

    print(f"\nRunning simulation: {T_MAX}s at {1/DT:.0f}Hz ({n_steps} steps)")
    print(f"  Defender start: {DEFENDER_POS_0}")
    print(f"  Attacker start: {ATTACKER_POS_0}")
    print(f"  Target center:  {TARGET_CENTER}")
    print()

    # Header
    print(f"{'t':>6s} | {'Defender (x,y,z)':>24s} | {'Attacker (x,y,z)':>24s} | {'h_dist':>7s} {'z_dist':>7s} | {'z_mode':>12s} {'h_mode':>12s} | {'status':>20s}")
    print("-" * 135)

    for step in range(n_steps):
        t = step * DT

        # ── Defender control (using real value functions) ──
        d_cmd, d_status = logic.compute_control(d_pos, d_vel, a_pos)

        # ── Attacker control (scripted waypoints) ──
        a_cmd, wp_idx = attacker_control(a_pos, wp_idx)

        # ── Step dynamics ──
        d_pos, d_vel = step_dynamics(d_pos, d_vel, d_cmd, K_X, K_Z, DT)
        a_pos, a_vel = step_dynamics(a_pos, a_vel, a_cmd, K_X, K_Z, DT)

        # ── Record ──
        h_dist = np.linalg.norm(d_pos[:2] - a_pos[:2])
        z_dist = abs(d_pos[2] - a_pos[2])

        t_hist.append(t)
        d_pos_hist.append(d_pos.copy())
        a_pos_hist.append(a_pos.copy())
        d_vel_hist.append(d_vel.copy())
        a_vel_hist.append(a_vel.copy())
        h_dist_hist.append(h_dist)
        z_dist_hist.append(z_dist)
        z_mode_hist.append(d_status.get("z_mode", "?"))
        h_mode_hist.append(d_status.get("h_mode", "?"))
        status_hist.append(d_status.get("game_status", "?"))

        # Print every 0.5s
        if step % int(0.5 / DT) == 0:
            print(f"{t:6.2f} | ({d_pos[0]:7.2f},{d_pos[1]:7.2f},{d_pos[2]:7.2f}) "
                  f"| ({a_pos[0]:7.2f},{a_pos[1]:7.2f},{a_pos[2]:7.2f}) "
                  f"| {h_dist:7.2f} {z_dist:7.2f} "
                  f"| {d_status.get('z_mode','?'):>12s} {d_status.get('h_mode','?'):>12s} "
                  f"| {d_status.get('game_status','?'):>20s}")

        # ── Check termination ──
        if is_captured(d_pos, a_pos):
            outcome = "CAPTURED"
            print(f"\n  >>> CAPTURED at t={t:.2f}s! h_dist={h_dist:.3f} z_dist={z_dist:.3f}")
            break

        if in_target(a_pos[:2]):
            outcome = "ATTACKER_REACHED_TARGET"
            print(f"\n  >>> ATTACKER REACHED TARGET at t={t:.2f}s! pos=({a_pos[0]:.1f},{a_pos[1]:.1f})")
            break

    print(f"\n{'='*60}")
    print(f"  OUTCOME: {outcome}")
    print(f"  Duration: {t_hist[-1]:.2f}s  ({len(t_hist)} steps)")
    print(f"  Final defender: ({d_pos[0]:.2f}, {d_pos[1]:.2f}, {d_pos[2]:.2f})")
    print(f"  Final attacker: ({a_pos[0]:.2f}, {a_pos[1]:.2f}, {a_pos[2]:.2f})")
    print(f"  Final h_dist={h_dist_hist[-1]:.3f}  z_dist={z_dist_hist[-1]:.3f}")
    print(f"  Min h_dist={min(h_dist_hist):.3f} at t={t_hist[h_dist_hist.index(min(h_dist_hist))]:.2f}s")
    print(f"{'='*60}")

    # ── Plot ──
    plot_trajectories(
        t_hist, d_pos_hist, a_pos_hist,
        h_dist_hist, z_dist_hist,
        z_mode_hist, h_mode_hist, status_hist,
        outcome,
    )

    return outcome


def plot_trajectories(t_hist, d_pos_hist, a_pos_hist,
                      h_dist_hist, z_dist_hist,
                      z_mode_hist, h_mode_hist, status_hist,
                      outcome):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    d_pos = np.array(d_pos_hist)
    a_pos = np.array(a_pos_hist)
    t = np.array(t_hist)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Reach-Avoid Game Simulation — Outcome: {outcome}", fontsize=14, fontweight="bold")

    # ── Top-left: XY trajectory (top-down view) ──
    ax = axes[0, 0]
    ax.set_title("Top-Down View (XY Plane)")

    # Draw obstacle
    obs_w = OBS["x_max"] - OBS["x_min"]
    obs_h = OBS["y_max"] - OBS["y_min"]
    ax.add_patch(patches.Rectangle(
        (OBS["x_min"], OBS["y_min"]), obs_w, obs_h,
        linewidth=1, edgecolor='gray', facecolor='lightcoral', alpha=0.5, label='Obstacle'))

    # Draw target
    tgt_w = TARGET["x_max"] - TARGET["x_min"]
    tgt_h = TARGET["y_max"] - TARGET["y_min"]
    ax.add_patch(patches.Rectangle(
        (TARGET["x_min"], TARGET["y_min"]), tgt_w, tgt_h,
        linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.5, label='Target'))

    # Trajectories
    ax.plot(d_pos[:, 0], d_pos[:, 1], 'b-', linewidth=1.5, label='Defender', alpha=0.8)
    ax.plot(a_pos[:, 0], a_pos[:, 1], 'r-', linewidth=1.5, label='Attacker', alpha=0.8)

    # Start/end markers
    ax.plot(d_pos[0, 0], d_pos[0, 1], 'bo', markersize=10, label='D start')
    ax.plot(d_pos[-1, 0], d_pos[-1, 1], 'bs', markersize=10, label='D end')
    ax.plot(a_pos[0, 0], a_pos[0, 1], 'ro', markersize=10, label='A start')
    ax.plot(a_pos[-1, 0], a_pos[-1, 1], 'rs', markersize=10, label='A end')

    # Time markers every 5s
    for mark_t in range(5, int(t[-1]) + 1, 5):
        idx = np.argmin(np.abs(t - mark_t))
        ax.annotate(f'{mark_t}s', (d_pos[idx, 0], d_pos[idx, 1]),
                    fontsize=7, color='blue', ha='center')
        ax.annotate(f'{mark_t}s', (a_pos[idx, 0], a_pos[idx, 1]),
                    fontsize=7, color='red', ha='center')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(gp["room"]["x_min"] - 1, gp["room"]["x_max"] + 1)
    ax.set_ylim(gp["room"]["y_min"] - 1, gp["room"]["y_max"] + 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Top-right: Altitude (Z) over time ──
    ax = axes[0, 1]
    ax.set_title("Altitude Over Time")
    ax.plot(t, d_pos[:, 2], 'b-', linewidth=1.5, label='Defender z')
    ax.plot(t, a_pos[:, 2], 'r-', linewidth=1.5, label='Attacker z')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude z (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: Inter-drone distances ──
    ax = axes[1, 0]
    ax.set_title("Inter-Drone Distance")
    ax.plot(t, h_dist_hist, 'purple', linewidth=1.5, label=f'Horizontal (capture={D_H}m)')
    ax.plot(t, z_dist_hist, 'orange', linewidth=1.5, label=f'Vertical (capture={D_Z}m)')
    ax.axhline(y=D_H, color='purple', linestyle='--', alpha=0.5)
    ax.axhline(y=D_Z, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: Control modes ──
    ax = axes[1, 1]
    ax.set_title("Control Modes Over Time")
    mode_map = {"reaching": 0, "tracking": 1, "pid_deep": 2, "pid_fallback": 3}
    z_modes_num = [mode_map.get(m, -1) for m in z_mode_hist]
    h_modes_num = [mode_map.get(m, -1) for m in h_mode_hist]
    ax.step(t, z_modes_num, 'b-', linewidth=1.5, label='Z mode', where='post')
    ax.step(t, h_modes_num, 'r-', linewidth=1.5, label='H mode', where='post')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["reaching", "tracking", "pid_deep", "pid_fallback"])
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to: {PLOT_PATH}")
    plt.close()


if __name__ == "__main__":
    outcome = run_simulation()
    sys.exit(0 if outcome == "CAPTURED" else 1)
