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


class DefenderControlLogic:
    """Pure control logic separated from ROS2 for testability.

    Implements Algorithm 1 (vertical) and Algorithm 2 (horizontal)
    reach-track control using precomputed value functions.
    """

    def __init__(
        self,
        loader: ValueFunctionLoader,
        pid_gain_z: float = 2.0,
        pid_gain_h: float = 2.0,
        margin_z_factor: float = 0.3,
        margin_h_factor: float = 0.3,
    ):
        self.loader = loader
        self.pid_gain_z = pid_gain_z
        self.pid_gain_h = pid_gain_h
        self.margin_z_factor = margin_z_factor
        self.margin_h_factor = margin_h_factor

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

        # B_z effective threshold from B_z params
        bz_params = loader.get_params("B_z") if "B_z" in loader.loaded_names else {}
        self.d_z_eff = bz_params.get("d_z_effective", self.d_z)

        bh_params = loader.get_params("B_h") if "B_h" in loader.loaded_names else {}
        self.d_h_eff = bh_params.get("d_h_effective", self.d_h)

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

        cmd_vel = np.array([u_x, u_y, u_z])

        # Apply wall-avoidance safety layer
        cmd_vel = self._apply_wall_avoidance(cmd_vel, defender_pos, defender_vel)

        return cmd_vel, status

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
                return self._optimal_reaching_vertical(vertical_state), "reaching"
            # Outside winning region or no VF: PID pursuit
            return self._pid_vertical(z_D, z_A), "pid_pursuit"

        # Inside B_z -- check if near boundary or deep inside
        margin_z = self.margin_z_factor * self.d_z
        if "V_z_inf" in self.loader.loaded_names:
            v_z_inf_val = self.loader.get_value("V_z_inf", bz_state)
            near_boundary = v_z_inf_val > (self.d_z_eff - margin_z)

            if near_boundary:
                # Mode 3: Optimal tracking from V_z_inf gradient
                return self._optimal_tracking_vertical(bz_state), "tracking"

        # Mode 4: Deep inside B_z, use PID
        return self._pid_vertical(z_D, z_A), "pid_deep"

    def _optimal_reaching_vertical(self, vertical_state: np.ndarray) -> float:
        """Extract optimal vertical reaching control from phi_z gradient.

        Defender maximizes phi_z: u_z = U_D_z * sign(dPhi_z/dv_Dz * k_z)
        """
        grad = self.loader.get_gradient("phi_z", vertical_state)
        # v_D_z is at index 1 in [z_D, v_D_z, z_A]
        direction = grad[1] * self.k_z
        return self.U_D_z if direction >= 0 else -self.U_D_z

    def _optimal_tracking_vertical(self, bz_state: np.ndarray) -> float:
        """Extract optimal vertical tracking control from V_z_inf gradient.

        Defender minimizes V_z_inf (wants to stay inside B_z):
        u_z = -U_D_z * sign(dV/dv_Dz * k_z)
        """
        grad = self.loader.get_gradient("V_z_inf", bz_state)
        # v_D_z is at index 1 in [z_rel, v_D_z]
        direction = grad[1] * self.k_z
        return -self.U_D_z if direction >= 0 else self.U_D_z

    def _pid_vertical(self, z_D: float, z_A: float) -> float:
        """Simple PID tracking: drive z_D toward z_A."""
        error = z_A - z_D
        return _clamp(self.pid_gain_z * error, self.U_D_z)

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
        defender's winning region (phi_h <= 0). Outside, we use PID pursuit.

        Modes:
        1. Outside winning region: PID pursuit toward attacker
        2. In winning region, NOT in B_h: optimal reaching from phi_h gradient
        3. In B_h, near boundary: optimal tracking from V_h_T gradient
        4. Deep in B_h: PID tracking

        Returns:
            (u_x, u_y, mode string)
        """
        if "B_h" not in self.loader.loaded_names:
            return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_fallback"

        # First check: are we in the defender's winning region?
        in_winning = False
        if "phi_h" in self.loader.loaded_names:
            phi_h_val = self.loader.get_value("phi_h", horizontal_state)
            in_winning = phi_h_val <= 0

        # B_h is in relative coords: [x_rel, y_rel, vx_D, vy_D]
        bh_state = np.array([x_rel, y_rel, vx_D, vy_D])
        b_h_val = self.loader.get_value("B_h", bh_state)
        in_B_h = b_h_val > 0.5

        if not in_B_h:
            if in_winning and "phi_h" in self.loader.loaded_names:
                # Mode 2: In winning region, optimal reaching from phi_h
                return *self._optimal_reaching_horizontal(horizontal_state), "reaching"
            # Outside winning region or no VF: PID pursuit toward attacker
            return *self._pid_horizontal(x_D, y_D, x_A, y_A), "pid_pursuit"

        # Inside B_h
        margin_h = self.margin_h_factor * self.d_h
        if "V_h_T" in self.loader.loaded_names:
            v_h_val = self.loader.get_value("V_h_T", bh_state)
            near_boundary = v_h_val > (self.d_h_eff - margin_h)

            if near_boundary:
                # Mode 3: Optimal tracking from V_h_T gradient
                return *self._optimal_tracking_horizontal(bh_state), "tracking"

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

        u_x = self.U_D_h if dir_x >= 0 else -self.U_D_h
        u_y = self.U_D_h if dir_y >= 0 else -self.U_D_h
        return u_x, u_y

    def _optimal_tracking_horizontal(self, bh_state: np.ndarray) -> tuple[float, float]:
        """Extract optimal horizontal tracking control from V_h_T gradient.

        Defender minimizes V_h_T to stay inside B_h.
        State: [x_rel, y_rel, vx_D, vy_D]
        u_x = -U_D_h * sign(dV/dv_Dx * k_x)
        """
        grad = self.loader.get_gradient("V_h_T", bh_state)
        # v_D_x at index 2, v_D_y at index 3
        dir_x = grad[2] * self.k_x
        dir_y = grad[3] * self.k_y

        u_x = -self.U_D_h if dir_x >= 0 else self.U_D_h
        u_y = -self.U_D_h if dir_y >= 0 else self.U_D_h
        return u_x, u_y

    def _pid_horizontal(self, x_D: float, y_D: float, x_A: float, y_A: float) -> tuple[float, float]:
        """Simple PID tracking: drive defender toward attacker horizontally."""
        ex = x_A - x_D
        ey = y_A - y_D
        ux = _clamp(self.pid_gain_h * ex, self.U_D_h)
        uy = _clamp(self.pid_gain_h * ey, self.U_D_h)
        return ux, uy

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
        captured = h_dist <= self.d_h and z_dist <= self.d_z
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
            status["in_W_D_h"] = phi_h_val <= 0

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

                vf_dir = self.get_parameter("value_function_dir").value
                rate = self.get_parameter("control_rate").value
                pid_z = self.get_parameter("pid_gain_z").value
                pid_h = self.get_parameter("pid_gain_h").value
                margin_z = self.get_parameter("margin_z_factor").value
                margin_h = self.get_parameter("margin_h_factor").value

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
                        )
                        self.get_logger().info(
                            f"Value functions loaded from {vf_dir}: {loader.loaded_names}"
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

            def _control_loop(self):
                """50Hz control loop implementing Algorithm 1 + Algorithm 2."""
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
