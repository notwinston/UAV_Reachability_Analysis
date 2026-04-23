"""Defender controller ROS2 node implementing reach-track control.

Implements Algorithm 1 (vertical reach-track) and Algorithm 2 (horizontal
reach-track-avoid) from the paper to intercept the attacker using
precomputed value functions.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# Import ValueFunctionLoader (works with or without reach_avoid_game)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reach_avoid_controller.value_function_loader import ValueFunctionLoader


def _clamp(value: float, limit: float) -> float:
    """Clamp value to [-limit, limit]."""
    return max(-limit, min(limit, value))


def _clamp_horizontal_speed(u_x: float, u_y: float, speed_limit: float) -> tuple[float, float]:
    """Clamp a horizontal command to the configured speed magnitude."""
    speed = math.hypot(u_x, u_y)
    if speed <= speed_limit or speed < 1e-12:
        return float(u_x), float(u_y)
    scale = speed_limit / speed
    return float(u_x * scale), float(u_y * scale)


class DefenderControlLogic:
    """Pure control logic separated from ROS2 for testability.

    Implements Algorithm 1 (vertical) and Algorithm 2 (horizontal)
    reach-track control using precomputed value functions.
    """

    def __init__(
        self,
        loader: ValueFunctionLoader,
        pid_gain_z: float = 8.0,
        pid_gain_h: float = 2.0,
        margin_z_factor: float = 0.3,
        margin_h_factor: float = 0.3,
        gradient_deadband: float = 1e-6,
        min_hj_closure_fraction: float = 0.85,
        capture_distance_horizontal: float | None = None,
        capture_distance_vertical: float | None = None,
    ):
        self.loader = loader
        self.pid_gain_z = pid_gain_z
        self.pid_gain_h = pid_gain_h
        self.margin_z_factor = margin_z_factor
        self.margin_h_factor = margin_h_factor
        self.gradient_deadband = gradient_deadband
        self.min_hj_closure_fraction = min_hj_closure_fraction

        # Extract game parameters from value functions
        z_params = loader.get_params("phi_z") if "phi_z" in loader.loaded_names else {}
        h_params = loader.get_params("phi_h") if "phi_h" in loader.loaded_names else {}

        self.d_z = z_params.get("d_z", 1.0)
        self.k_z = z_params.get("k_z", 1.5)
        self.U_D_z = z_params.get("U_D_z", 4.0)
        self.U_A_z = z_params.get("U_A_z", 2.0)

        self.d_h = h_params.get("d_h", 3.0)
        self.k_x = h_params.get("k_x", 0.7)
        self.k_y = h_params.get("k_y", 0.7)
        self.U_D_h = h_params.get("U_D_h", 6.0)
        self.U_A_h = h_params.get("U_A_h", 3.0)
        self.capture_d_h = float(capture_distance_horizontal) if capture_distance_horizontal is not None else float(self.d_h)
        self.capture_d_z = float(capture_distance_vertical) if capture_distance_vertical is not None else float(self.d_z)

        self.b_z_valid = self._valid_invariant_mask("B_z", "d_z_effective", self.d_z)

        self.b_h_valid = self._valid_invariant_mask("B_h", "d_h_effective", self.d_h)
        self.h_tracking_vf_name = self._select_horizontal_tracking_vf()

    def _valid_invariant_mask(self, name: str, effective_key: str, threshold: float) -> bool:
        """Reject threshold-expanded or empty invariant masks."""
        if name not in self.loader.loaded_names:
            return False
        params = self.loader.get_params(name)
        if float(params.get(effective_key, threshold)) > threshold + 1e-9:
            return False
        values = getattr(self.loader.vf_data[name], "values", None)
        return (
            values is not None
            and bool(params.get("paper_valid", False))
            and bool(params.get("subset_valid", True))
            and bool(np.any(np.asarray(values) > 0.5))
        )

    def _value_function_reaches_threshold(self, name: str, threshold: float) -> bool:
        if name not in self.loader.loaded_names:
            return False
        values = getattr(self.loader.vf_data[name], "values", None)
        return values is not None and float(np.nanmin(values)) <= threshold

    def _select_horizontal_tracking_vf(self) -> str | None:
        """Use only obstacle-aware 6D tracking data for paper Algorithm 2."""
        vf = self.loader.vf_data.get("V_h_T_6d") if "V_h_T_6d" in self.loader.loaded_names else None
        if (
            vf is not None
            and getattr(vf, "values", np.array([])).ndim == 6
            and self._value_function_reaches_threshold("V_h_T_6d", self.d_h)
        ):
            return "V_h_T_6d"
        return None

    def compute_control(
        self,
        defender_pos: np.ndarray,
        defender_vel: np.ndarray,
        attacker_pos: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Compute defender velocity command.

        Args:
            defender_pos: [x_D, y_D, z_D]
            defender_vel: [vx_D, vy_D, vz_D]
            attacker_pos: [x_A, y_A, z_A]

        Returns:
            (cmd_vel [ux, uy, uz], status_info dict)
        """
        # Decompose states
        x_D, y_D, z_D = defender_pos
        vx_D, vy_D, vz_D = defender_vel
        x_A, y_A, z_A = attacker_pos

        # Vertical state: [z_D, v_D_z, z_A]
        vertical_state = np.array([z_D, vz_D, z_A])

        # Horizontal state: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
        horizontal_state = np.array([x_D, y_D, vx_D, vy_D, x_A, y_A])

        # Relative states for invariant set lookups
        z_rel = z_D - z_A
        x_rel = x_D - x_A
        y_rel = y_D - y_A

        status = {}

        # --- Algorithm 1: Vertical Reach-Track ---
        u_z, z_mode = self._vertical_reach_track(vertical_state, z_rel, vz_D, z_A, z_D)
        status["z_mode"] = z_mode

        # --- Algorithm 2: Horizontal Reach-Track-Avoid ---
        u_x, u_y, h_mode = self._horizontal_reach_track(
            horizontal_state, x_rel, y_rel, vx_D, vy_D, x_A, y_A, x_D, y_D,
        )
        status["h_mode"] = h_mode

        # Clamp combined command to speed limits
        u_z = _clamp(u_z, self.U_D_z)

        # Clamp horizontal speed
        h_speed = math.sqrt(u_x**2 + u_y**2)
        if h_speed > self.U_D_h:
            scale = self.U_D_h / h_speed
            u_x *= scale
            u_y *= scale

        # Check winning conditions
        status.update(self._check_game_status(
            vertical_state, horizontal_state,
            np.array([x_A, y_A]), z_rel, x_rel, y_rel,
            defender_pos, attacker_pos,
        ))
        if status.get("captured", False):
            return np.zeros(3, dtype=float), status

        cmd_vel = np.array([u_x, u_y, u_z])

        # Apply wall-avoidance safety layer
        cmd_vel = self._apply_wall_avoidance(cmd_vel, defender_pos, defender_vel)
        cmd_vel = self._apply_obstacle_avoidance(cmd_vel, defender_pos)

        return cmd_vel, status

    def _apply_obstacle_avoidance(self, cmd_vel: np.ndarray, defender_pos: np.ndarray) -> np.ndarray:
        """Project horizontal commands away from configured box obstacles.

        The controller does not own the full YAML config, so this mirrors the
        default obstacle used by the reachability configuration.
        """
        result = cmd_vel.copy()
        obstacles = [(15.0, 20.0, 5.0, 20.0)]
        margin = 1.0
        x, y = float(defender_pos[0]), float(defender_pos[1])
        for x_min, x_max, y_min, y_max in obstacles:
            near_x = x_min - margin <= x <= x_max + margin
            near_y = y_min - margin <= y <= y_max + margin
            if not (near_x and near_y):
                continue
            distances = {
                "left": abs(x - x_min),
                "right": abs(x - x_max),
                "bottom": abs(y - y_min),
                "top": abs(y - y_max),
            }
            side = min(distances, key=distances.get)
            if side == "left" and result[0] > 0:
                result[0] = min(result[0], 0.0)
            elif side == "right" and result[0] < 0:
                result[0] = max(result[0], 0.0)
            elif side == "bottom" and result[1] > 0:
                result[1] = min(result[1], 0.0)
            elif side == "top" and result[1] < 0:
                result[1] = max(result[1], 0.0)
        return result

    def _apply_wall_avoidance(
        self,
        cmd_vel: np.ndarray,
        defender_pos: np.ndarray,
        defender_vel: np.ndarray,
    ) -> np.ndarray:
        """Post-processing safety layer to prevent wall collisions.

        Scales down velocity commands toward nearby walls using a dynamic
        safety margin based on stopping distance for double-integrator
        dynamics. Actively pushes away when very close (< 1m) to a wall.
        """
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([45.0, 25.0, 20.0])
        k = [self.k_x, self.k_y, self.k_z]
        u_max = [self.U_D_h, self.U_D_h, self.U_D_z]

        result = cmd_vel.copy()

        for i in range(3):
            v = defender_vel[i]
            pos = defender_pos[i]

            # Stopping distance for v_dot = k*(u - v) dynamics.
            # Conservative estimate using exponential decay over one time constant.
            if k[i] > 1e-6:
                d_stop = abs(v) / k[i] * (1.0 - math.exp(-1.0))
            else:
                d_stop = 0.0

            # Dynamic safety margin: at least 1.5m from any wall
            margin = max(d_stop * 2.0, 1.5)

            dist_min = pos - bounds_min[i]
            dist_max = bounds_max[i] - pos

            # Near min wall and moving toward it (or very close)
            if dist_min < margin and (v < 0 or pos < 1.0):
                scale = max(0.0, dist_min / margin)
                if result[i] < 0:
                    result[i] *= scale
                # Very close: actively push away
                if dist_min < 1.0:
                    result[i] = max(result[i], u_max[i] * 0.5)

            # Near max wall and moving toward it (or very close)
            if dist_max < margin and (v > 0 or pos > bounds_max[i] - 1.0):
                scale = max(0.0, dist_max / margin)
                if result[i] > 0:
                    result[i] *= scale
                # Very close: actively push away
                if dist_max < 1.0:
                    result[i] = min(result[i], -u_max[i] * 0.5)

        return result

    def _vertical_reach_track(
        self,
        vertical_state: np.ndarray,
        z_rel: float,
        vz_D: float,
        z_A: float,
        z_D: float,
    ) -> tuple[float, str]:
        """Algorithm 1 -- Vertical Reach-Track controller.

        Per the paper, optimal HJ control is only valid inside the
        defender's winning region (phi_z <= 0). Outside, we use PID pursuit.

        Modes:
        1. Outside winning region: PID pursuit toward attacker
        2. In winning region, NOT in B_z: optimal reaching from phi_z gradient
        3. In B_z, near boundary: optimal tracking from V_z_inf gradient
        4. Deep in B_z: PID tracking

        Returns:
            (u_z control value, mode string)
        """
        if "B_z" not in self.loader.loaded_names:
            return self._pid_vertical(z_D, z_A), "pid_fallback"
        if not self.b_z_valid:
            return self._pid_vertical(z_D, z_A), "pid_invalid_bz"

        # First check: are we in the defender's winning region?
        in_winning = False
        if "phi_z" in self.loader.loaded_names:
            phi_z_val = self.loader.get_value("phi_z", vertical_state)
            in_winning = phi_z_val <= 0

        # Check if inside invariant set B_z (B_z is stored with z_rel, v_D_z coords)
        bz_state = np.array([z_rel, vz_D])
        b_z_val = self.loader.get_value("B_z", bz_state)
        in_B_z = b_z_val > 0.5

        if not in_B_z:
            if in_winning and "phi_z" in self.loader.loaded_names:
                # Mode 2: In winning region, optimal reaching from phi_z gradient
                u_z = self._optimal_reaching_vertical(vertical_state)
                if self._vertical_command_closes_gap(u_z, z_rel):
                    return u_z, "reaching"
                return self._pid_vertical(z_D, z_A), "pid_pursuit"
            # Outside winning region or no VF: PID pursuit
            return self._pid_vertical(z_D, z_A), "pid_pursuit"

        # Inside B_z -- check if near boundary or deep inside
        margin_z = self.margin_z_factor * self.d_z
        if "V_z_inf" in self.loader.loaded_names:
            v_z_inf_val = self.loader.get_value("V_z_inf", bz_state)
            near_boundary = v_z_inf_val > (self.d_z - margin_z)

            if near_boundary:
                # Mode 3: Optimal tracking from V_z_inf gradient
                u_z = self._optimal_tracking_vertical(bz_state)
                if self._vertical_command_closes_gap(u_z, z_rel):
                    return u_z, "tracking"
                return self._pid_vertical(z_D, z_A), "pid_pursuit"

        # Mode 4: Deep inside B_z, use PID
        return self._pid_vertical(z_D, z_A), "pid_deep"

    def _optimal_reaching_vertical(self, vertical_state: np.ndarray) -> float:
        """Extract optimal vertical reaching control from phi_z gradient.

        Defender minimizes phi_z: u_z = -U_D_z when dPhi_z/dv_Dz * k_z > 0.
        """
        grad = self.loader.get_gradient("phi_z", vertical_state)
        # v_D_z is at index 1 in [z_D, v_D_z, z_A]
        direction = grad[1] * self.k_z
        if abs(direction) < self.gradient_deadband:
            return 0.0
        return -self.U_D_z if direction > 0 else self.U_D_z

    def _optimal_tracking_vertical(self, bz_state: np.ndarray) -> float:
        """Extract optimal vertical tracking control from V_z_inf gradient.

        Defender minimizes V_z_inf (wants to stay inside B_z):
        u_z = -U_D_z * sign(dV/dv_Dz * k_z)
        """
        grad = self.loader.get_gradient("V_z_inf", bz_state)
        # v_D_z is at index 1 in [z_rel, v_D_z]
        direction = grad[1] * self.k_z
        if abs(direction) < self.gradient_deadband:
            return 0.0
        return -self.U_D_z if direction > 0 else self.U_D_z

    def _pid_vertical(self, z_D: float, z_A: float) -> float:
        """Simple PID tracking: drive z_D toward z_A."""
        error = z_A - z_D
        return _clamp(self.pid_gain_z * error, self.U_D_z)

    def _vertical_command_closes_gap(self, u_z: float, z_rel: float) -> bool:
        """Check whether a command moves defender toward attacker vertically."""
        return (z_rel * u_z) < -self.gradient_deadband

    def _horizontal_reach_track(
        self,
        horizontal_state: np.ndarray,
        x_rel: float,
        y_rel: float,
        vx_D: float,
        vy_D: float,
        x_A: float,
        y_A: float,
        x_D: float,
        y_D: float,
    ) -> tuple[float, float, str]:
        """Algorithm 2 -- Horizontal Reach-Track-Avoid controller.

        Per the paper, optimal HJ control is only valid inside the
        defender's winning region (paper Phi_h > 0). Outside, we use PID pursuit.

        Modes:
        1. Outside winning region: PID pursuit toward attacker
        2. In winning region, NOT in B_h: optimal reaching from phi_h gradient
        3. In B_h, near boundary: optimal tracking from V_h_T gradient
        4. Deep in B_h: PID tracking

        Returns:
            (u_x, u_y, mode string)
        """
        if self.h_tracking_vf_name is None:
            return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_invalid_bh"
        if math.sqrt(x_rel**2 + y_rel**2) <= self.d_h:
            return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_deep"

        # First check: are we in the defender's winning region?
        in_winning = False
        if "phi_h" in self.loader.loaded_names:
            phi_h_val = self.loader.get_value("phi_h", horizontal_state)
            in_winning = phi_h_val > 0

        # Paper Algorithm 2 uses obstacle-aware horizontal invariant tracking.
        h_tracking_state = horizontal_state
        v_h_val_current = self.loader.get_value(self.h_tracking_vf_name, h_tracking_state)
        in_B_h = v_h_val_current <= self.d_h

        if not in_B_h:
            if in_winning and "phi_h" in self.loader.loaded_names:
                # Mode 2: In winning region, optimal reaching from phi_h
                u_x, u_y = self._optimal_reaching_horizontal(horizontal_state)
                if self._horizontal_command_is_useful(u_x, u_y, x_rel, y_rel, x_D, y_D, x_A, y_A):
                    return u_x, u_y, "reaching"
                return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_pursuit"
            # Outside winning region or no VF: PID pursuit toward attacker
            return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_pursuit"

        # Inside B_h
        margin_h = self.margin_h_factor * self.d_h
        if self.h_tracking_vf_name is not None:
            near_boundary = v_h_val_current > (self.d_h - margin_h)

            if near_boundary:
                # Mode 3: Optimal tracking from V_h_T gradient
                u_x, u_y = self._optimal_tracking_horizontal(h_tracking_state)
                if self._horizontal_command_is_useful(u_x, u_y, x_rel, y_rel, x_D, y_D, x_A, y_A):
                    return u_x, u_y, "tracking"
                return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_pursuit"

        # Mode 4: Deep inside B_h, PID
        return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_deep"

    def _optimal_reaching_horizontal(self, horizontal_state: np.ndarray) -> tuple[float, float]:
        """Extract optimal horizontal reaching control from phi_h gradient.

        Defender maximizes phi_h. State: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
        Control enters through velocity dynamics:
        v_D_x_dot = k_x * (u_x - v_D_x)
        So: u_x = U_D_h * sign(dPhi_h/dv_Dx * k_x)
        u_y = U_D_h * sign(dPhi_h/dv_Dy * k_y)
        """
        grad = self.loader.get_gradient("phi_h", horizontal_state)
        # v_D_x at index 2, v_D_y at index 3
        dir_x = grad[2] * self.k_x
        dir_y = grad[3] * self.k_y

        u_x = self.U_D_h if dir_x > 0 else -self.U_D_h
        u_y = self.U_D_h if dir_y > 0 else -self.U_D_h
        if abs(dir_x) < self.gradient_deadband:
            u_x = 0.0
        if abs(dir_y) < self.gradient_deadband:
            u_y = 0.0
        return _clamp_horizontal_speed(u_x, u_y, self.U_D_h)

    def _optimal_tracking_horizontal(self, h_tracking_state: np.ndarray) -> tuple[float, float]:
        """Extract optimal horizontal tracking control from V_h_T gradient.

        Defender minimizes obstacle-aware V_h_T to stay inside B_h.
        State: [x_D, y_D, vx_D, vy_D, x_A, y_A]
        u_x = -U_D_h * sign(dV/dv_Dx * k_x)
        """
        grad = self.loader.get_gradient(self.h_tracking_vf_name, h_tracking_state)
        # v_D_x at index 2, v_D_y at index 3
        dir_x = grad[2] * self.k_x
        dir_y = grad[3] * self.k_y

        u_x = -self.U_D_h if dir_x > 0 else self.U_D_h
        u_y = -self.U_D_h if dir_y > 0 else self.U_D_h
        if abs(dir_x) < self.gradient_deadband:
            u_x = 0.0
        if abs(dir_y) < self.gradient_deadband:
            u_y = 0.0
        return _clamp_horizontal_speed(u_x, u_y, self.U_D_h)

    def _pid_horizontal(self, x_D: float, y_D: float, x_A: float, y_A: float) -> tuple[float, float]:
        """Simple PID tracking: drive defender toward attacker horizontally."""
        ex = x_A - x_D
        ey = y_A - y_D
        ux = _clamp(self.pid_gain_h * ex, self.U_D_h)
        uy = _clamp(self.pid_gain_h * ey, self.U_D_h)
        return _clamp_horizontal_speed(ux, uy, self.U_D_h)

    def _horizontal_command_closes_gap(
        self,
        u_x: float,
        u_y: float,
        x_rel: float,
        y_rel: float,
    ) -> bool:
        """Check whether a command moves defender toward attacker in horizontal projection."""
        return (x_rel * u_x + y_rel * u_y) < -self.gradient_deadband

    def _horizontal_closing_speed(
        self,
        u_x: float,
        u_y: float,
        x_rel: float,
        y_rel: float,
    ) -> float:
        """Projected closure rate along the defender-attacker line."""
        dist = math.hypot(x_rel, y_rel)
        if dist < self.gradient_deadband:
            return 0.0
        return -(x_rel * u_x + y_rel * u_y) / dist

    def _horizontal_command_is_useful(
        self,
        u_x: float,
        u_y: float,
        x_rel: float,
        y_rel: float,
        x_D: float,
        y_D: float,
        x_A: float,
        y_A: float,
    ) -> bool:
        """Accept HJ commands only when they are competitive with pursuit.

        Coarse 6D gradients can point diagonally even when the attacker is
        straight ahead. Keeping the HJ command only when its line-of-sight
        closure is close to the PID pursuit closure preserves useful HJ
        behavior while avoiding arbitrary lateral drift.
        """
        hj_closure = self._horizontal_closing_speed(u_x, u_y, x_rel, y_rel)
        if hj_closure <= self.gradient_deadband:
            return False
        pid_x, pid_y = self._pid_horizontal(x_D, y_D, x_A, y_A)
        pid_closure = self._horizontal_closing_speed(pid_x, pid_y, x_rel, y_rel)
        if pid_closure <= self.gradient_deadband:
            return True
        return hj_closure >= self.min_hj_closure_fraction * pid_closure

    def _check_game_status(
        self,
        vertical_state: np.ndarray,
        horizontal_state: np.ndarray,
        attacker_pos_h: np.ndarray,
        z_rel: float,
        x_rel: float,
        y_rel: float,
        defender_pos: np.ndarray,
        attacker_pos: np.ndarray,
    ) -> dict:
        """Check winning conditions and capture status."""
        status = {}

        # Capture distances
        h_dist = math.sqrt(x_rel**2 + y_rel**2)
        z_dist = abs(z_rel)
        status["h_dist"] = h_dist
        status["z_dist"] = z_dist

        # Captured?
        captured = h_dist <= self.capture_d_h and z_dist <= self.capture_d_z
        status["captured"] = captured

        if captured:
            status["game_status"] = "CAPTURED"
            return status

        # Check winning regions if value functions available
        if "phi_z" in self.loader.loaded_names:
            phi_z_val = self.loader.get_value("phi_z", vertical_state)
            status["in_W_D_z"] = phi_z_val <= 0

        if "phi_h" in self.loader.loaded_names:
            phi_h_val = self.loader.get_value("phi_h", horizontal_state)
            status["in_W_D_h"] = phi_h_val > 0

        # Determine overall game status string
        in_w_d_z = status.get("in_W_D_z", False)
        in_w_d_h = status.get("in_W_D_h", False)

        if in_w_d_z and in_w_d_h:
            status["game_status"] = "DEFENDER_WINNING"
        elif in_w_d_z or in_w_d_h:
            status["game_status"] = "UNCERTAIN"
        else:
            status["game_status"] = "ATTACKER_ADVANTAGE"

        return status


def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
        from std_msgs.msg import String

        class DefenderControllerNode(Node):
            """ROS2 node wrapping DefenderControlLogic with reach-track control."""

            def __init__(self):
                super().__init__("defender_controller")

                # Parameters
                self.declare_parameter("value_function_dir", "/workspace/data/value_functions/")
                self.declare_parameter("control_rate", 50.0)
                self.declare_parameter("pid_gain_z", 2.0)
                self.declare_parameter("pid_gain_h", 2.0)
                self.declare_parameter("margin_z_factor", 0.3)
                self.declare_parameter("margin_h_factor", 0.3)
                self.declare_parameter("gradient_deadband", 1e-6)
                self.declare_parameter("min_hj_closure_fraction", 0.85)
                self.declare_parameter("command_filter_alpha", 0.35)
                self.declare_parameter("max_accel_horizontal", 2.0)
                self.declare_parameter("max_accel_vertical", 1.5)
                self.declare_parameter("capture_distance_horizontal", -1.0)
                self.declare_parameter("capture_distance_vertical", -1.0)

                vf_dir = self.get_parameter("value_function_dir").value
                rate = self.get_parameter("control_rate").value
                pid_z = self.get_parameter("pid_gain_z").value
                pid_h = self.get_parameter("pid_gain_h").value
                margin_z = self.get_parameter("margin_z_factor").value
                margin_h = self.get_parameter("margin_h_factor").value
                gradient_deadband = self.get_parameter("gradient_deadband").value
                min_hj_closure_fraction = self.get_parameter("min_hj_closure_fraction").value
                self._filter_alpha = float(self.get_parameter("command_filter_alpha").value)
                self._max_accel_h = float(self.get_parameter("max_accel_horizontal").value)
                self._max_accel_z = float(self.get_parameter("max_accel_vertical").value)
                capture_d_h = float(self.get_parameter("capture_distance_horizontal").value)
                capture_d_z = float(self.get_parameter("capture_distance_vertical").value)

                # Load value functions
                self._logic = None
                try:
                    loader = ValueFunctionLoader(vf_dir)
                    if loader.all_loaded:
                        self._logic = DefenderControlLogic(
                            loader,
                            pid_gain_z=pid_z,
                            pid_gain_h=pid_h,
                            margin_z_factor=margin_z,
                            margin_h_factor=margin_h,
                            gradient_deadband=gradient_deadband,
                            min_hj_closure_fraction=min_hj_closure_fraction,
                            capture_distance_horizontal=(capture_d_h if capture_d_h > 0.0 else None),
                            capture_distance_vertical=(capture_d_z if capture_d_z > 0.0 else None),
                        )
                        self.get_logger().info(
                            f"Value functions loaded from {vf_dir}: {loader.loaded_names}"
                        )
                        if capture_d_h > 0.0 or capture_d_z > 0.0:
                            self.get_logger().info(
                                "Capture overrides enabled: "
                                f"d_h={self._logic.capture_d_h:.3f}, d_z={self._logic.capture_d_z:.3f}"
                            )
                    else:
                        self.get_logger().error(
                            f"Not all value functions loaded. Have: {loader.loaded_names}"
                        )
                except Exception as e:
                    self.get_logger().error(f"Failed to load value functions: {e}")

                # State storage
                self._defender_pos = None  # [x, y, z]
                self._defender_vel = None  # [vx, vy, vz]
                self._attacker_pos = None  # [x, y, z]
                self._attacker_vel = None  # [vx, vy, vz]
                self._filtered_cmd = np.array([0.0, 0.0, 0.0], dtype=float)
                self._last_filter_time = None
                self._terminal_stop = False
                self._terminal_status = "UNKNOWN"

                # Subscribers
                self.create_subscription(
                    PoseStamped, "/defender/state", self._defender_state_cb, 10
                )
                self.create_subscription(
                    TwistStamped, "/defender/velocity", self._defender_vel_cb, 10
                )
                self.create_subscription(
                    PoseStamped, "/attacker/state", self._attacker_state_cb, 10
                )
                self.create_subscription(
                    TwistStamped, "/attacker/velocity", self._attacker_vel_cb, 10
                )
                self.create_subscription(
                    String, "/game/status", self._game_status_cb, 10
                )

                # Publishers
                self._cmd_pub = self.create_publisher(Twist, "/defender/cmd_vel", 10)
                self._status_pub = self.create_publisher(String, "/game/status", 10)

                # Control timer
                self._timer = self.create_timer(1.0 / rate, self._control_loop)

                self.get_logger().info("Defender controller node started (reach-track)")

            def _defender_state_cb(self, msg: PoseStamped):
                self._defender_pos = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ])

            def _defender_vel_cb(self, msg: TwistStamped):
                self._defender_vel = np.array([
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.linear.z,
                ])

            def _attacker_state_cb(self, msg: PoseStamped):
                self._attacker_pos = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ])

            def _attacker_vel_cb(self, msg: TwistStamped):
                self._attacker_vel = np.array([
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.linear.z,
                ])

            def _game_status_cb(self, msg: String):
                status = msg.data.split("|", 1)[0].strip()
                if status in ("CAPTURED", "ATTACKER_REACHED_TARGET"):
                    self._terminal_stop = True
                    self._terminal_status = status

            def _control_loop(self):
                """50Hz control loop implementing Algorithm 1 + Algorithm 2."""
                if self._terminal_stop:
                    self._publish_hover()
                    status_msg = String()
                    status_msg.data = self._terminal_status
                    self._status_pub.publish(status_msg)
                    return

                # Safe fallback: if value functions not loaded, hover
                if self._logic is None:
                    self.get_logger().warn(
                        "Value functions not loaded, hovering", throttle_duration_sec=5.0
                    )
                    self._publish_hover()
                    return

                # Wait for all state data
                if (
                    self._defender_pos is None
                    or self._defender_vel is None
                    or self._attacker_pos is None
                ):
                    return

                # Compute control using reach-track logic
                cmd_vel, status = self._logic.compute_control(
                    self._defender_pos,
                    self._defender_vel,
                    self._attacker_pos,
                )
                cmd_vel = self._condition_command(cmd_vel)

                # Publish cmd_vel
                msg = Twist()
                msg.linear.x = float(cmd_vel[0])
                msg.linear.y = float(cmd_vel[1])
                msg.linear.z = float(cmd_vel[2])
                self._cmd_pub.publish(msg)

                # Publish game status
                status_msg = String()
                game_status = status.get("game_status", "UNKNOWN")
                z_mode = status.get("z_mode", "?")
                h_mode = status.get("h_mode", "?")
                h_dist = status.get("h_dist", -1.0)
                z_dist = status.get("z_dist", -1.0)
                status_msg.data = (
                    f"{game_status} | "
                    f"z:{z_mode} h:{h_mode} | "
                    f"d_h={h_dist:.2f} d_z={z_dist:.2f}"
                )
                self._status_pub.publish(status_msg)

            def _condition_command(self, cmd_vel):
                """Rate-limit and smooth outgoing ROS commands before PX4 sees them."""
                now = self.get_clock().now().nanoseconds / 1e9
                if self._last_filter_time is None:
                    self._last_filter_time = now
                dt = max(0.001, min(0.2, now - self._last_filter_time))
                self._last_filter_time = now

                target = np.array(cmd_vel, dtype=float)
                target[0], target[1] = _clamp_horizontal_speed(
                    float(target[0]), float(target[1]), self._logic.U_D_h
                )
                target[2] = _clamp(float(target[2]), self._logic.U_D_z)

                max_delta = np.array([
                    self._max_accel_h * dt,
                    self._max_accel_h * dt,
                    self._max_accel_z * dt,
                ])
                delta = np.clip(target - self._filtered_cmd, -max_delta, max_delta)
                limited = self._filtered_cmd + delta
                alpha = max(0.0, min(1.0, self._filter_alpha))
                self._filtered_cmd = (1.0 - alpha) * self._filtered_cmd + alpha * limited
                self._filtered_cmd[0], self._filtered_cmd[1] = _clamp_horizontal_speed(
                    float(self._filtered_cmd[0]), float(self._filtered_cmd[1]), self._logic.U_D_h
                )
                self._filtered_cmd[2] = _clamp(float(self._filtered_cmd[2]), self._logic.U_D_z)
                return self._filtered_cmd.copy()

            def _publish_hover(self):
                """Publish zero velocity (hover)."""
                msg = Twist()
                self._cmd_pub.publish(msg)

        rclpy.init(args=args)
        node = DefenderControllerNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print("defender_controller: rclpy not available, running in stub mode")


if __name__ == "__main__":
    main()
