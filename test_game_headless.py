"""Headless test of the reach-avoid game with kinematic simulation.

Runs the defender (reach-track) and attacker (optimal/scripted) controllers
against each other using simple kinematic integration — no ROS2 needed.
"""

import sys
import os
import math
import numpy as np

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reach_avoid_game', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reach_avoid_ws', 'src', 'reach_avoid_controller'))

from reach_avoid_controller.value_function_loader import ValueFunctionLoader
# Import the pure control logic from defender_node (no ROS2)
# We need to extract it without ROS2 imports

VF_DIR = os.path.join(os.path.dirname(__file__), 'data', 'value_functions')

# Game parameters
K = [0.7, 0.7, 1.5]  # k_x, k_y, k_z
DT = 0.02  # 50 Hz
U_D_H = 6.0
U_D_Z = 4.0
U_A_H = 3.0
U_A_Z = 2.0
D_H = 3.0
D_Z = 1.0
BOUNDS_LO = [0.0, 0.0, 0.0]
BOUNDS_HI = [45.0, 25.0, 20.0]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SimpleDefender:
    """Minimal defender using reach-track logic from value functions."""

    def __init__(self, loader):
        self.loader = loader
        bz_params = loader.get_params("B_z") if "B_z" in loader.loaded_names else {}
        self.d_z_eff = bz_params.get("d_z_effective", D_Z)
        bh_params = loader.get_params("B_h") if "B_h" in loader.loaded_names else {}
        self.d_h_eff = bh_params.get("d_h_effective", D_H)

    def compute(self, d_pos, d_vel, a_pos):
        x_D, y_D, z_D = d_pos
        vx_D, vy_D, vz_D = d_vel
        x_A, y_A, z_A = a_pos

        z_rel = z_D - z_A
        x_rel = x_D - x_A
        y_rel = y_D - y_A

        # --- Vertical ---
        u_z, z_mode = self._vertical(z_D, vz_D, z_A, z_rel)

        # --- Horizontal ---
        u_x, u_y, h_mode = self._horizontal(
            x_D, y_D, vx_D, vy_D, x_A, y_A, x_rel, y_rel)

        # Clamp
        u_z = clamp(u_z, -U_D_Z, U_D_Z)
        h_speed = math.sqrt(u_x**2 + u_y**2)
        if h_speed > U_D_H:
            scale = U_D_H / h_speed
            u_x *= scale
            u_y *= scale

        return [u_x, u_y, u_z], z_mode, h_mode

    def _vertical(self, z_D, vz_D, z_A, z_rel):
        bz_state = np.array([z_rel, vz_D])

        if "B_z" not in self.loader.loaded_names:
            return self._pid_z(z_D, z_A), "pid_fallback"

        b_z_val = self.loader.get_value("B_z", bz_state)
        in_B_z = b_z_val > 0.5

        if not in_B_z:
            # Use V_z_inf gradient to steer toward B_z
            if "V_z_inf" in self.loader.loaded_names:
                grad = self.loader.get_gradient("V_z_inf", bz_state)
                direction = grad[1] * K[2]
                if abs(direction) < 1e-8:
                    return self._pid_z(z_D, z_A), "pid_pursuit"
                return (-U_D_Z if direction >= 0 else U_D_Z), "reaching_via_Vzinf"
            return self._pid_z(z_D, z_A), "pid_pursuit"

        # Inside B_z — tracking
        margin_z = 0.3 * D_Z
        if "V_z_inf" in self.loader.loaded_names:
            v_z_inf_val = self.loader.get_value("V_z_inf", bz_state)
            if v_z_inf_val > (self.d_z_eff - margin_z):
                grad = self.loader.get_gradient("V_z_inf", bz_state)
                direction = grad[1] * K[2]
                return (-U_D_Z if direction >= 0 else U_D_Z), "tracking"

        return self._pid_z(z_D, z_A), "pid_deep"

    def _horizontal(self, x_D, y_D, vx_D, vy_D, x_A, y_A, x_rel, y_rel):
        if "B_h" not in self.loader.loaded_names:
            return *self._pid_h(x_D, y_D, x_A, y_A), "pid_fallback"

        h_state = np.array([x_D, y_D, vx_D, vy_D, x_A, y_A])
        bh_state = np.array([x_rel, y_rel, vx_D, vy_D])

        # Check winning region
        in_winning = False
        if "phi_h" in self.loader.loaded_names:
            phi_h_val = self.loader.get_value("phi_h", h_state)
            in_winning = phi_h_val > 0

        b_h_val = self.loader.get_value("B_h", bh_state)
        in_B_h = b_h_val > 0.5

        if not in_B_h:
            if in_winning and "phi_h" in self.loader.loaded_names:
                # Optimal reaching from phi_h gradient
                grad = self.loader.get_gradient("phi_h", h_state)
                dir_x = grad[2] * K[0]
                dir_y = grad[3] * K[1]
                grad_vel_mag = math.sqrt(dir_x**2 + dir_y**2)
                if grad_vel_mag < 0.01:
                    return *self._pid_h(x_D, y_D, x_A, y_A), "reaching_pid"
                u_x = U_D_H if dir_x >= 0 else -U_D_H
                u_y = U_D_H if dir_y >= 0 else -U_D_H
                return u_x, u_y, "reaching"
            return *self._pid_h(x_D, y_D, x_A, y_A), "pid_pursuit"

        # Inside B_h — tracking
        margin_h = 0.3 * D_H
        if "V_h_T" in self.loader.loaded_names:
            v_h_val = self.loader.get_value("V_h_T", bh_state)
            if v_h_val > (self.d_h_eff - margin_h):
                grad = self.loader.get_gradient("V_h_T", bh_state)
                dir_x = grad[2] * K[0]
                dir_y = grad[3] * K[1]
                u_x = -U_D_H if dir_x >= 0 else U_D_H
                u_y = -U_D_H if dir_y >= 0 else U_D_H
                return u_x, u_y, "tracking"

        return *self._pid_h(x_D, y_D, x_A, y_A), "pid_deep"

    def _pid_z(self, z_D, z_A):
        error = z_A - z_D
        return clamp(2.0 * error, -U_D_Z, U_D_Z)

    def _pid_h(self, x_D, y_D, x_A, y_A):
        ex = x_A - x_D
        ey = y_A - y_D
        return clamp(2.0 * ex, -U_D_H, U_D_H), clamp(2.0 * ey, -U_D_H, U_D_H)


class ScriptedAttacker:
    """Follow waypoints at constant speed."""

    def __init__(self, waypoints, speed=2.0):
        self.waypoints = waypoints
        self.speed = speed
        self.wp_idx = 0

    def compute(self, a_pos):
        if self.wp_idx >= len(self.waypoints):
            self.wp_idx = len(self.waypoints) - 1
        wp = self.waypoints[self.wp_idx]
        dx = wp[0] - a_pos[0]
        dy = wp[1] - a_pos[1]
        dz = wp[2] - a_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 0.5 and self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            wp = self.waypoints[self.wp_idx]
            dx = wp[0] - a_pos[0]
            dy = wp[1] - a_pos[1]
            dz = wp[2] - a_pos[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0.1:
            return [dx/dist * self.speed, dy/dist * self.speed, dz/dist * self.speed]
        return [0.0, 0.0, 0.0]


class OptimalAttacker:
    """Game-theoretic optimal attacker using value function gradients."""

    def __init__(self, loader, target_center=(41.5, 12.5), target_alt=10.0):
        self.loader = loader
        self.target_center = target_center
        self.target_alt = target_alt

    def compute(self, a_pos, d_pos, d_vel):
        x_A, y_A, z_A = a_pos
        x_D, y_D, z_D = d_pos
        vx_D, vy_D, vz_D = d_vel
        cmd = [0.0, 0.0, 0.0]

        try:
            # Horizontal
            h_state = np.array([x_D, y_D, vx_D, vy_D, x_A, y_A])
            h_grad = self.loader.get_gradient('phi_h', h_state)
            grad_xa, grad_ya = h_grad[4], h_grad[5]
            grad_a_mag = math.sqrt(grad_xa**2 + grad_ya**2)
            if grad_a_mag < 0.01:
                # Gradient too small — goal-seek toward target
                dx = self.target_center[0] - x_A
                dy = self.target_center[1] - y_A
                dist_h = math.sqrt(dx*dx + dy*dy)
                if dist_h > 0.1:
                    cmd[0] = (dx/dist_h) * U_A_H
                    cmd[1] = (dy/dist_h) * U_A_H
            else:
                # Attacker minimizes phi_h (wants phi_h <= 0)
                cmd[0] = -U_A_H if grad_xa >= 0 else U_A_H
                cmd[1] = -U_A_H if grad_ya >= 0 else U_A_H

            # Vertical
            v_state = np.array([z_D, vz_D, z_A])
            v_grad = self.loader.get_gradient('phi_z', v_state)
            grad_za = v_grad[2]
            grad_z_mag = np.sqrt(v_grad[0]**2 + v_grad[1]**2 + v_grad[2]**2)
            if grad_z_mag < 0.01:
                # phi_z is flat — hold target altitude
                dz = self.target_alt - z_A
                cmd[2] = clamp(2.0 * dz, -U_A_Z, U_A_Z)
            else:
                # Attacker maximizes phi_z
                cmd[2] = U_A_Z if grad_za >= 0 else -U_A_Z
        except Exception:
            # Fallback: go to target
            dx = self.target_center[0] - x_A
            dy = self.target_center[1] - y_A
            dist_h = math.sqrt(dx*dx + dy*dy)
            if dist_h > 0.1:
                cmd[0] = (dx/dist_h) * U_A_H
                cmd[1] = (dy/dist_h) * U_A_H
            dz = self.target_alt - z_A
            cmd[2] = clamp(2.0 * dz, -U_A_Z, U_A_Z)

        return cmd


def simulate(attacker_mode="scripted", duration=30.0, print_interval=1.0):
    """Run headless kinematic simulation."""
    print(f"Loading value functions from {VF_DIR}...")
    loader = ValueFunctionLoader(VF_DIR)
    print(f"  Loaded: {loader.loaded_names}")

    defender = SimpleDefender(loader)

    if attacker_mode == "scripted":
        waypoints = [
            [5.0, 12.5, 10.0],
            [12.0, 12.5, 10.0],
            [12.0, 3.0, 10.0],
            [25.0, 3.0, 10.0],
            [25.0, 12.5, 10.0],
            [41.5, 12.5, 10.0],
        ]
        attacker = ScriptedAttacker(waypoints, speed=2.0)
    else:
        attacker = OptimalAttacker(loader)

    # Initial positions
    d_pos = [5.0, 12.5, 3.0]
    d_vel = [0.0, 0.0, 0.0]
    a_pos = [5.0, 20.0, 3.0]

    # Target region
    target = {"x_min": 38.0, "x_max": 45.0, "y_min": 10.0, "y_max": 15.0}

    t = 0.0
    captured = False
    attacker_reached_target = False
    last_print = -print_interval

    print(f"\nSimulation: attacker_mode={attacker_mode}, duration={duration}s")
    print(f"  Defender start: {d_pos}")
    print(f"  Attacker start: {a_pos}")
    print(f"  Target region: x=[{target['x_min']}, {target['x_max']}] "
          f"y=[{target['y_min']}, {target['y_max']}]")
    print("-" * 90)
    print(f"{'t':>5s} | {'D_pos':^25s} | {'A_pos':^25s} | {'h_dist':>6s} {'z_dist':>6s} | z_mode  h_mode")
    print("-" * 90)

    while t < duration:
        # Compute controls
        d_cmd, z_mode, h_mode = defender.compute(d_pos, d_vel, a_pos)

        if attacker_mode == "scripted":
            a_cmd = attacker.compute(a_pos)
        else:
            a_cmd = attacker.compute(a_pos, d_pos, d_vel)

        # Defender: double integrator
        for i in range(3):
            d_vel[i] += DT * K[i] * (d_cmd[i] - d_vel[i])
            d_pos[i] += d_vel[i] * DT

        # Attacker: single integrator
        for i in range(3):
            a_pos[i] += a_cmd[i] * DT

        # Clamp to bounds
        for i in range(3):
            d_pos[i] = clamp(d_pos[i], BOUNDS_LO[i], BOUNDS_HI[i])
            a_pos[i] = clamp(a_pos[i], BOUNDS_LO[i], BOUNDS_HI[i])
            if d_pos[i] <= BOUNDS_LO[i] and d_vel[i] < 0:
                d_vel[i] = 0.0
            if d_pos[i] >= BOUNDS_HI[i] and d_vel[i] > 0:
                d_vel[i] = 0.0

        # Check distances
        h_dist = math.sqrt((d_pos[0]-a_pos[0])**2 + (d_pos[1]-a_pos[1])**2)
        z_dist = abs(d_pos[2] - a_pos[2])

        if h_dist <= D_H and z_dist <= D_Z and not captured:
            captured = True
            print(f"\n*** CAPTURED at t={t:.2f}s! h_dist={h_dist:.2f}, z_dist={z_dist:.2f} ***\n")

        # Check if attacker reached target
        in_target = (target["x_min"] <= a_pos[0] <= target["x_max"] and
                     target["y_min"] <= a_pos[1] <= target["y_max"])
        if in_target and not attacker_reached_target:
            attacker_reached_target = True
            print(f"\n*** ATTACKER REACHED TARGET at t={t:.2f}s! ***\n")

        # Print status
        if t - last_print >= print_interval - 0.001:
            d_str = f"({d_pos[0]:5.1f}, {d_pos[1]:5.1f}, {d_pos[2]:5.1f})"
            a_str = f"({a_pos[0]:5.1f}, {a_pos[1]:5.1f}, {a_pos[2]:5.1f})"
            print(f"{t:5.1f} | {d_str:^25s} | {a_str:^25s} | "
                  f"{h_dist:6.2f} {z_dist:6.2f} | {z_mode:8s} {h_mode}")
            last_print = t

        t += DT

    print("-" * 90)
    print(f"\nResult after {duration}s:")
    print(f"  Captured: {captured}")
    print(f"  Attacker reached target: {attacker_reached_target}")
    h_dist = math.sqrt((d_pos[0]-a_pos[0])**2 + (d_pos[1]-a_pos[1])**2)
    z_dist = abs(d_pos[2] - a_pos[2])
    print(f"  Final h_dist={h_dist:.2f}, z_dist={z_dist:.2f}")
    print(f"  Defender final: ({d_pos[0]:.1f}, {d_pos[1]:.1f}, {d_pos[2]:.1f})")
    print(f"  Attacker final: ({a_pos[0]:.1f}, {a_pos[1]:.1f}, {a_pos[2]:.1f})")

    return captured, attacker_reached_target


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scripted"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    simulate(attacker_mode=mode, duration=dur)
