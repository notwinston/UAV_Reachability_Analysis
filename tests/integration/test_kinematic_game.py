"""Standalone end-to-end kinematic game test.

Replicates the full ROS2 kinematic_game launch without ROS2:
  - KinematicSim: simple velocity integration (matching kinematic_sim_node.py)
  - AttackerController: scripted waypoint following (matching attacker_node.py)
  - DefenderController: reach-track control with value functions (matching defender_node.py)

This validates the complete interception pipeline.
"""

import math
import sys
import numpy as np

sys.path.insert(0, '/workspace/reach_avoid_ws/src/reach_avoid_controller')
from reach_avoid_controller.value_function_loader import ValueFunctionLoader
from reach_avoid_controller.defender_node import DefenderControlLogic


# ---------- Kinematic simulator (matches kinematic_sim_node.py) ----------
class KinematicSim:
    """Simple velocity integration with arena bounds clamping."""

    def __init__(self, defender_pos, attacker_pos, dt=0.02):
        self.d_pos = np.array(defender_pos, dtype=float)
        self.d_vel = np.zeros(3)
        self.a_pos = np.array(attacker_pos, dtype=float)
        self.a_vel = np.zeros(3)
        self.dt = dt
        # Defender double-integrator gains (from game_params.yaml)
        self.k_x = 0.7
        self.k_y = 0.7
        self.k_z = 1.5

    def step(self, defender_cmd_vel, attacker_cmd_vel):
        """Advance one timestep.

        Defender: double integrator (velocity with lag).
        Attacker: single integrator (direct velocity control).
        """
        # Defender: double integrator dynamics
        # v_dot = k * (u - v)
        self.d_vel[0] += self.dt * self.k_x * (defender_cmd_vel[0] - self.d_vel[0])
        self.d_vel[1] += self.dt * self.k_y * (defender_cmd_vel[1] - self.d_vel[1])
        self.d_vel[2] += self.dt * self.k_z * (defender_cmd_vel[2] - self.d_vel[2])
        self.d_pos += self.dt * self.d_vel

        # Attacker: single integrator (direct velocity)
        self.a_vel = np.array(attacker_cmd_vel, dtype=float)
        self.a_pos += self.dt * self.a_vel

        # Clamp to arena [0,45] x [0,25] x [0,20]
        self.d_pos[0] = np.clip(self.d_pos[0], 0, 45)
        self.d_pos[1] = np.clip(self.d_pos[1], 0, 25)
        self.d_pos[2] = np.clip(self.d_pos[2], 0, 20)
        self.a_pos[0] = np.clip(self.a_pos[0], 0, 45)
        self.a_pos[1] = np.clip(self.a_pos[1], 0, 25)
        self.a_pos[2] = np.clip(self.a_pos[2], 0, 20)


# ---------- Attacker waypoint controller (matches attacker_node.py) ----------
class AttackerScriptedController:
    """Follow waypoints at constant speed."""

    def __init__(self, waypoints, max_speed=2.0, speed_fraction=0.8):
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.speed = max_speed * speed_fraction
        self.wp_idx = 0

    def compute(self, position):
        """Return velocity command toward current waypoint."""
        if self.wp_idx >= len(self.waypoints):
            return np.zeros(3)

        wp = self.waypoints[self.wp_idx]
        diff = wp - position
        dist = np.linalg.norm(diff)

        if dist < 0.5 and self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            wp = self.waypoints[self.wp_idx]
            diff = wp - position
            dist = np.linalg.norm(diff)

        if dist > 0.1:
            return (diff / dist) * self.speed
        return np.zeros(3)


def run_game(scenario_name, d_start, a_start, waypoints, max_time=30.0, dt=0.02):
    """Run a complete game simulation and return results."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"  Defender start: ({d_start[0]:.1f}, {d_start[1]:.1f}, {d_start[2]:.1f})")
    print(f"  Attacker start: ({a_start[0]:.1f}, {a_start[1]:.1f}, {a_start[2]:.1f})")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"{'='*70}")

    # Initialize
    loader = ValueFunctionLoader('/workspace/data/value_functions/')
    defender = DefenderControlLogic(loader)
    attacker = AttackerScriptedController(waypoints, max_speed=2.0, speed_fraction=0.8)
    sim = KinematicSim(d_start, a_start, dt=dt)

    n_steps = int(max_time / dt)
    captured = False
    capture_time = None

    # Capture params
    d_h_capture = 3.0  # meters
    d_z_capture = 1.0  # meters

    for step in range(n_steps):
        t = step * dt

        # Attacker control
        a_cmd = attacker.compute(sim.a_pos)

        # Defender control
        d_cmd, status = defender.compute_control(sim.d_pos, sim.d_vel, sim.a_pos)

        # Simulate
        sim.step(d_cmd, a_cmd)

        # Check capture
        h_dist = np.sqrt((sim.d_pos[0]-sim.a_pos[0])**2 + (sim.d_pos[1]-sim.a_pos[1])**2)
        z_dist = abs(sim.d_pos[2] - sim.a_pos[2])

        # Print every 2 seconds
        if step % int(2.0 / dt) == 0:
            print(f"  t={t:5.1f}s  D=({sim.d_pos[0]:5.1f},{sim.d_pos[1]:5.1f},{sim.d_pos[2]:5.1f})  "
                  f"A=({sim.a_pos[0]:5.1f},{sim.a_pos[1]:5.1f},{sim.a_pos[2]:5.1f})  "
                  f"d_h={h_dist:5.1f} d_z={z_dist:4.1f}  "
                  f"mode={status['h_mode']}/{status['z_mode']}  "
                  f"game={status.get('game_status','?')}")

        if h_dist <= d_h_capture and z_dist <= d_z_capture:
            captured = True
            capture_time = t
            print(f"\n  >>> CAPTURED at t={t:.2f}s <<<")
            print(f"      Defender: ({sim.d_pos[0]:.2f}, {sim.d_pos[1]:.2f}, {sim.d_pos[2]:.2f})")
            print(f"      Attacker: ({sim.a_pos[0]:.2f}, {sim.a_pos[1]:.2f}, {sim.a_pos[2]:.2f})")
            print(f"      h_dist={h_dist:.2f}m  z_dist={z_dist:.2f}m")
            break

        # Check if attacker reached target region [38,45] x [10,15]
        if sim.a_pos[0] >= 38.0 and sim.a_pos[1] >= 10.0 and sim.a_pos[1] <= 15.0:
            print(f"\n  >>> ATTACKER REACHED TARGET at t={t:.2f}s <<<")
            print(f"      Attacker: ({sim.a_pos[0]:.2f}, {sim.a_pos[1]:.2f}, {sim.a_pos[2]:.2f})")
            break

    if not captured and capture_time is None:
        print(f"\n  >>> TIMEOUT after {max_time:.0f}s <<<")
        print(f"      Final d_h={h_dist:.2f}m  d_z={z_dist:.2f}m")

    return captured, capture_time


def main():
    print("=" * 70)
    print("END-TO-END KINEMATIC GAME SIMULATION")
    print("Replicates: ros2 launch reach_avoid_bringup kinematic_game.launch.py")
    print("=" * 70)

    # Paper attacker waypoints (from simulation_params.yaml / attacker_node.py)
    paper_waypoints = [
        [5.0, 12.5, 10.0],    # First: move toward center, gain altitude
        [12.0, 12.5, 10.0],   # Past obstacle (x=15-20 is obstacle, go through y<5 gap)
        [12.0, 3.0, 10.0],    # Below obstacle
        [25.0, 3.0, 10.0],    # Past obstacle
        [25.0, 12.5, 10.0],   # Back to center
        [41.5, 12.5, 10.0],   # Target region
    ]

    results = []

    # Scenario 1: Paper initial conditions
    ok, t = run_game(
        "Paper initial conditions (defender at (5,12.5,3), attacker at (5,20,3))",
        d_start=[5.0, 12.5, 3.0],
        a_start=[5.0, 20.0, 3.0],
        waypoints=paper_waypoints,
    )
    results.append(("Paper scenario", ok, t))

    # Scenario 2: Attacker ahead, moving straight to target
    ok, t = run_game(
        "Attacker ahead — straight line to target",
        d_start=[5.0, 12.5, 10.0],
        a_start=[20.0, 12.5, 10.0],
        waypoints=[[41.5, 12.5, 10.0]],
    )
    results.append(("Attacker ahead", ok, t))

    # Scenario 3: Attacker going around obstacle
    ok, t = run_game(
        "Attacker navigating around obstacle",
        d_start=[5.0, 12.5, 10.0],
        a_start=[12.0, 12.5, 10.0],
        waypoints=[
            [12.0, 3.0, 10.0],
            [25.0, 3.0, 10.0],
            [25.0, 12.5, 10.0],
            [41.5, 12.5, 10.0],
        ],
    )
    results.append(("Around obstacle", ok, t))

    # Scenario 4: Different altitude
    ok, t = run_game(
        "Altitude difference (defender at z=3, attacker at z=15)",
        d_start=[10.0, 12.5, 3.0],
        a_start=[10.0, 12.5, 15.0],
        waypoints=[[41.5, 12.5, 15.0]],
    )
    results.append(("Altitude diff", ok, t))

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    all_passed = True
    for name, captured, t in results:
        status = f"CAPTURED at {t:.1f}s" if captured else "FAILED"
        icon = "PASS" if captured else "FAIL"
        print(f"  [{icon}] {name}: {status}")
        if not captured:
            all_passed = False

    if all_passed:
        print(f"\nAll {len(results)} scenarios passed — defender intercepts attacker in all cases.")
    else:
        print(f"\nSome scenarios failed!")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
