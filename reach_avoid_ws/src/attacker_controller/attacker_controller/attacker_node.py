"""Attacker controller ROS2 node.

Supports four control modes:
  - scripted:   Follow waypoints toward target region at constant speed.
  - keyboard:   Republish teleop_twist_keyboard input.
  - optimal:    Game-theoretic HJ value function bang-bang control.
  - switchable: Start in scripted mode, switch via ROS2 service at runtime.
"""

import math

import numpy as np


DEFAULT_HJ_GRADIENT_DEADBAND = 1e-4
DEFAULT_HJ_DESCENT_TOLERANCE = 1e-4


def _clamp(value, low, high):
    return max(low, min(high, value))


def _clamp_horizontal_speed(x, y, limit):
    speed = math.hypot(x, y)
    if speed <= limit or speed < 1e-9:
        return x, y
    scale = limit / speed
    return x * scale, y * scale


def _phi_a_grid_spacing(vf_data):
    shape = np.asarray(vf_data.values.shape, dtype=float)
    denom = np.maximum(shape - 1.0, 1.0)
    return (np.asarray(vf_data.grid_max, dtype=float) - np.asarray(vf_data.grid_min, dtype=float)) / denom


def _hj_local_descent_delta(loader, vf_name, state, tolerance):
    vf_data = loader.vf_data[vf_name]
    grid_min = np.asarray(vf_data.grid_min, dtype=float)
    grid_max = np.asarray(vf_data.grid_max, dtype=float)
    spacing = _phi_a_grid_spacing(vf_data)
    current = loader.get_value(vf_name, state)
    best_value = current
    best_delta = None

    for dx in (-spacing[0], 0.0, spacing[0]):
        for dy in (-spacing[1], 0.0, spacing[1]):
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                continue
            candidate = np.clip(state + np.array([dx, dy], dtype=float), grid_min, grid_max)
            value = loader.get_value(vf_name, candidate)
            if value < best_value - tolerance:
                best_value = value
                best_delta = candidate - state

    return best_delta


def _scale_direction_to_speed(dx, dy, speed_limit):
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return 0.0, 0.0
    scale = float(speed_limit) / norm
    return float(dx * scale), float(dy * scale)


def _project_obstacle_velocity(vx, vy, position, obstacle, lookahead, margin, mode='inflated'):
    """Project a horizontal command away from a box obstacle.

    ``inflated`` preserves the legacy behavior and activates within an inflated
    margin. ``hard_barrier`` only intervenes when the vehicle is inside the box
    or the lookahead step would cross into it, which avoids overriding obstacle-
    aware HJ commands that already account for the obstacle.
    """
    if position is None:
        return vx, vy

    x, y = float(position[0]), float(position[1])
    px = x + float(vx) * float(lookahead)
    py = y + float(vy) * float(lookahead)

    inside_now = (
        obstacle['x_min'] <= x <= obstacle['x_max']
        and obstacle['y_min'] <= y <= obstacle['y_max']
    )
    inside_next = (
        obstacle['x_min'] <= px <= obstacle['x_max']
        and obstacle['y_min'] <= py <= obstacle['y_max']
    )

    entry_side = None
    if not inside_now and inside_next:
        if x < obstacle['x_min'] <= px:
            entry_side = 'left'
        elif x > obstacle['x_max'] >= px:
            entry_side = 'right'
        elif y < obstacle['y_min'] <= py:
            entry_side = 'bottom'
        elif y > obstacle['y_max'] >= py:
            entry_side = 'top'

    if mode == 'hard_barrier':
        if not (inside_now or inside_next):
            return vx, vy
        ref_x, ref_y = (x, y) if inside_now else (px, py)
    else:
        near_now = (
            obstacle['x_min'] - margin <= x <= obstacle['x_max'] + margin
            and obstacle['y_min'] - margin <= y <= obstacle['y_max'] + margin
        )
        near_next = (
            obstacle['x_min'] - margin <= px <= obstacle['x_max'] + margin
            and obstacle['y_min'] - margin <= py <= obstacle['y_max'] + margin
        )
        if not (near_now or near_next):
            return vx, vy
        ref_x, ref_y = x, y

    distances = {
        'left': abs(ref_x - obstacle['x_min']),
        'right': abs(ref_x - obstacle['x_max']),
        'bottom': abs(ref_y - obstacle['y_min']),
        'top': abs(ref_y - obstacle['y_max']),
    }
    side = entry_side if entry_side is not None else min(distances, key=distances.get)
    push = 1.0 if inside_now else 0.0

    if side == 'left':
        if vx > 0.0:
            vx = 0.0
        vx = min(vx, -push)
    elif side == 'right':
        if vx < 0.0:
            vx = 0.0
        vx = max(vx, push)
    elif side == 'bottom':
        if vy > 0.0:
            vy = 0.0
        vy = min(vy, -push)
    elif side == 'top':
        if vy < 0.0:
            vy = 0.0
        vy = max(vy, push)
    return vx, vy


def extract_attacker_hj_reaching_command(
    loader,
    vf_name,
    x_a,
    y_a,
    u_a_h,
    gradient_deadband=DEFAULT_HJ_GRADIENT_DEADBAND,
    descent_tolerance=DEFAULT_HJ_DESCENT_TOLERANCE,
):
    """Extract online HJ target-reaching control from ``phi_A_reach``.

    The attacker minimizes the reaching value function, so the bang-bang HJ
    control is ``u_i = -U_A_h * sign(dV/dx_i)``. When the coarse interpolated
    gradient is too small to be informative, use local value descent on the
    neighboring grid cells instead of abandoning the HJ policy entirely.
    """
    state = np.array([x_a, y_a], dtype=float)
    grad = np.asarray(loader.get_gradient(vf_name, state), dtype=float)

    cmd_x = 0.0
    cmd_y = 0.0
    if grad[0] > gradient_deadband:
        cmd_x = -u_a_h
    elif grad[0] < -gradient_deadband:
        cmd_x = u_a_h
    if grad[1] > gradient_deadband:
        cmd_y = -u_a_h
    elif grad[1] < -gradient_deadband:
        cmd_y = u_a_h

    if abs(grad[0]) < gradient_deadband or abs(grad[1]) < gradient_deadband:
        delta = _hj_local_descent_delta(loader, vf_name, state, descent_tolerance)
        if delta is not None:
            fallback_x, fallback_y = _scale_direction_to_speed(delta[0], delta[1], u_a_h)
            if abs(grad[0]) < gradient_deadband:
                cmd_x = fallback_x
            if abs(grad[1]) < gradient_deadband:
                cmd_y = fallback_y

    return _clamp_horizontal_speed(float(cmd_x), float(cmd_y), float(u_a_h))


def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
        from std_msgs.msg import String
        from std_srvs.srv import SetBool
        import yaml
        import numpy as np
        import sys as _sys
        import os as _os
        for _candidate in [
            '/home/simuser/ws/src/reach_avoid_controller',
            '/workspace/reach_avoid_ws/src/reach_avoid_controller',
            '/workspaces/UAV_Reachability_Analysis/reach_avoid_ws/src/reach_avoid_controller',
        ]:
            if _os.path.isdir(_candidate) and _candidate not in _sys.path:
                _sys.path.insert(0, _candidate)
        from reach_avoid_controller.value_function_loader import ValueFunctionLoader

        class AttackerControllerNode(Node):
            """Attacker controller with multiple control modes."""

            def __init__(self):
                super().__init__('attacker_controller')

                # Parameters
                self.declare_parameter('mode', 'optimal')
                self.declare_parameter('max_speed', 0.5)
                self.declare_parameter('speed_fraction', 0.8)
                self.declare_parameter('target_x', 7.0)
                self.declare_parameter('target_y', 4.0)
                self.declare_parameter('target_z', 2.0)
                self.declare_parameter('waypoints', [0.0])
                self.declare_parameter('value_function_dir', '/home/simuser/ws/data/value_functions/')
                self.declare_parameter('target_altitude', 10.0)
                self.declare_parameter('room_x_min', 0.0)
                self.declare_parameter('room_x_max', 45.0)
                self.declare_parameter('room_y_min', 0.0)
                self.declare_parameter('room_y_max', 25.0)
                self.declare_parameter('room_z_min', 0.5)
                self.declare_parameter('room_z_max', 20.0)
                self.declare_parameter('obstacle_x_min', 15.0)
                self.declare_parameter('obstacle_x_max', 20.0)
                self.declare_parameter('obstacle_y_min', 5.0)
                self.declare_parameter('obstacle_y_max', 20.0)
                self.declare_parameter('safety_margin', 1.5)
                self.declare_parameter('obstacle_margin', 3.0)
                self.declare_parameter('safety_lookahead', 1.0)
                self.declare_parameter('command_filter_alpha', 0.35)
                self.declare_parameter('max_accel_horizontal', 1.0)
                self.declare_parameter('max_accel_vertical', 1.0)
                self.declare_parameter('obstacle_projection_mode', 'auto')
                self.declare_parameter('hj_gradient_deadband', DEFAULT_HJ_GRADIENT_DEADBAND)
                self.declare_parameter('hj_descent_tolerance', DEFAULT_HJ_DESCENT_TOLERANCE)

                self._mode = self.get_parameter('mode').value
                self._max_speed = self.get_parameter('max_speed').value
                self._speed_fraction = self.get_parameter('speed_fraction').value
                self._target = [
                    self.get_parameter('target_x').value,
                    self.get_parameter('target_y').value,
                    self.get_parameter('target_z').value,
                ]
                self._room_min = [
                    float(self.get_parameter('room_x_min').value),
                    float(self.get_parameter('room_y_min').value),
                    float(self.get_parameter('room_z_min').value),
                ]
                self._room_max = [
                    float(self.get_parameter('room_x_max').value),
                    float(self.get_parameter('room_y_max').value),
                    float(self.get_parameter('room_z_max').value),
                ]
                self._obstacle = {
                    'x_min': float(self.get_parameter('obstacle_x_min').value),
                    'x_max': float(self.get_parameter('obstacle_x_max').value),
                    'y_min': float(self.get_parameter('obstacle_y_min').value),
                    'y_max': float(self.get_parameter('obstacle_y_max').value),
                }
                self._safety_margin = float(self.get_parameter('safety_margin').value)
                self._obstacle_margin = float(self.get_parameter('obstacle_margin').value)
                self._safety_lookahead = float(self.get_parameter('safety_lookahead').value)
                self._filter_alpha = float(self.get_parameter('command_filter_alpha').value)
                self._max_accel_h = float(self.get_parameter('max_accel_horizontal').value)
                self._max_accel_z = float(self.get_parameter('max_accel_vertical').value)
                self._obstacle_projection_mode = str(
                    self.get_parameter('obstacle_projection_mode').value
                ).strip() or 'auto'
                self._hj_gradient_deadband = float(self.get_parameter('hj_gradient_deadband').value)
                self._hj_descent_tolerance = float(self.get_parameter('hj_descent_tolerance').value)
                waypoints_param = self.get_parameter('waypoints').value
                self._waypoints = self._parse_waypoints(waypoints_param)

                # Current state
                self._position = None
                self._velocity = None
                self._defender_position = None
                self._defender_velocity = None
                self._current_waypoint_idx = 0
                self._terminal_stop = False

                # For switchable mode, track active sub-mode
                self._active_submode = 'scripted'
                self._filtered_cmd = [0.0, 0.0, 0.0]
                self._last_control_time = None

                # Subscribers
                self.create_subscription(
                    PoseStamped, '/attacker/state',
                    self._state_callback, 10
                )
                self.create_subscription(
                    TwistStamped, '/attacker/velocity',
                    self._velocity_callback, 10
                )
                self.create_subscription(
                    PoseStamped, '/defender/state',
                    self._defender_state_callback, 10
                )
                self.create_subscription(
                    TwistStamped, '/defender/velocity',
                    self._defender_velocity_callback, 10
                )
                self.create_subscription(
                    String, '/game/status',
                    self._game_status_callback, 10,
                )

                # Keyboard mode subscriber
                if self._mode in ('keyboard', 'switchable'):
                    self.create_subscription(
                        Twist, '/attacker/teleop',
                        self._teleop_callback, 10
                    )
                self._teleop_cmd = Twist()

                # Publisher
                self.cmd_vel_pub = self.create_publisher(
                    Twist, '/attacker/cmd_vel', 10
                )

                # Service for switchable mode
                if self._mode == 'switchable':
                    self.create_service(
                        SetBool, '/attacker/set_mode',
                        self._set_mode_callback
                    )

                # 20Hz control timer
                self._timer = self.create_timer(0.05, self._control_loop)

                # Value function loading for optimal mode
                self._vf_loader = None
                self._U_A_h = 3.0  # attacker max horizontal speed
                self._U_A_z = 2.0  # attacker max vertical speed
                self._target_center = [self._target[0], self._target[1]]
                self._phi_a_reach_name = 'phi_A_reach'

                if self._mode == 'optimal':
                    vf_dir = self.get_parameter('value_function_dir').value
                    try:
                        loader = ValueFunctionLoader(vf_dir)
                        self._vf_loader = loader if loader.loaded_names else None
                        if self._vf_loader is not None:
                            h_params = loader.get_params('phi_h') if 'phi_h' in loader.loaded_names else {}
                            z_params = loader.get_params('phi_z') if 'phi_z' in loader.loaded_names else {}
                            a_params = (
                                loader.get_params(self._phi_a_reach_name)
                                if self._phi_a_reach_name in loader.loaded_names else {}
                            )
                            self._U_A_h = h_params.get(
                                'U_A_h',
                                a_params.get('U_A_h', self._U_A_h),
                            )
                            self._U_A_z = z_params.get('U_A_z', self._U_A_z)
                            self.get_logger().info(
                                f'Optimal mode: loaded VFs from {vf_dir}, '
                                f'available={loader.loaded_names}, '
                                f'U_A_h={self._U_A_h}, U_A_z={self._U_A_z}'
                            )
                        else:
                            self.get_logger().warn(
                                f'Optimal mode using default attacker limits; '
                                f'no value functions available in {vf_dir}'
                            )
                    except Exception as e:
                        self.get_logger().error(f'Failed to load value functions: {e}')

                self.get_logger().info(
                    f'Attacker controller started in "{self._mode}" mode, '
                    f'max_speed={self._max_speed}, target={self._target}'
                )

            def _parse_waypoints(self, waypoints_param):
                """Parse waypoints parameter into list of [x, y, z] points."""
                if not waypoints_param:
                    # Default: go directly to target
                    return [self._target[:]]
                # Expect flat list [x1, y1, z1, x2, y2, z2, ...]
                if len(waypoints_param) % 3 != 0:
                    self.get_logger().warn('Waypoints must have 3 coords each, using target only')
                    return [self._target[:]]
                return [
                    [waypoints_param[i], waypoints_param[i+1], waypoints_param[i+2]]
                    for i in range(0, len(waypoints_param), 3)
                ]

            def _state_callback(self, msg: PoseStamped):
                """Update current position from ground truth."""
                self._position = [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ]

            def _velocity_callback(self, msg: TwistStamped):
                """Update current velocity from ground truth."""
                self._velocity = [
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.linear.z,
                ]

            def _teleop_callback(self, msg: Twist):
                """Store latest teleop command."""
                self._teleop_cmd = msg

            def _defender_state_callback(self, msg: PoseStamped):
                """Update defender position from ground truth."""
                self._defender_position = [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ]

            def _defender_velocity_callback(self, msg: TwistStamped):
                """Update defender velocity from ground truth."""
                self._defender_velocity = [
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.linear.z,
                ]

            def _game_status_callback(self, msg: String):
                """Latch terminal game status so PX4 receives hover commands."""
                status = msg.data.split('|', 1)[0].strip()
                if status in ('CAPTURED', 'ATTACKER_REACHED_TARGET'):
                    self._terminal_stop = True

            def _set_mode_callback(self, request, response):
                """Service callback for switchable mode.

                SetBool: data=True -> switch to keyboard, data=False -> switch to scripted.
                """
                if request.data:
                    self._active_submode = 'keyboard'
                    response.success = True
                    response.message = 'Switched to keyboard mode'
                else:
                    self._active_submode = 'scripted'
                    response.success = True
                    response.message = 'Switched to scripted mode'
                self.get_logger().info(response.message)
                return response

            def _control_loop(self):
                """Main control loop at 20Hz."""
                cmd = Twist()
                if self._terminal_stop:
                    self.cmd_vel_pub.publish(cmd)
                    return

                if self._mode == 'scripted':
                    cmd = self._scripted_control()
                elif self._mode == 'keyboard':
                    cmd = self._keyboard_control()
                elif self._mode == 'optimal':
                    cmd = self._optimal_control()
                elif self._mode == 'switchable':
                    if self._active_submode == 'keyboard':
                        cmd = self._keyboard_control()
                    else:
                        cmd = self._scripted_control()
                else:
                    self.get_logger().warn(f'Unknown mode: {self._mode}', throttle_duration_sec=5.0)

                cmd = self._condition_command(cmd)
                self.cmd_vel_pub.publish(cmd)

            def _condition_command(self, cmd):
                vx = float(cmd.linear.x)
                vy = float(cmd.linear.y)
                vz = float(cmd.linear.z)
                vx, vy = _clamp_horizontal_speed(vx, vy, self._U_A_h)
                vz = _clamp(vz, -self._U_A_z, self._U_A_z)
                vx, vy, vz = self._apply_geofence_projection(vx, vy, vz)
                vx, vy = self._apply_obstacle_projection(vx, vy)
                vx, vy = _clamp_horizontal_speed(vx, vy, self._U_A_h)
                vx, vy, vz = self._smooth(vx, vy, vz)
                vx, vy, vz = self._apply_geofence_projection(vx, vy, vz)
                vx, vy = self._apply_obstacle_projection(vx, vy)
                vx, vy = _clamp_horizontal_speed(vx, vy, self._U_A_h)
                vz = _clamp(vz, -self._U_A_z, self._U_A_z)
                filtered = Twist()
                filtered.linear.x = vx
                filtered.linear.y = vy
                filtered.linear.z = vz
                return filtered

            def _smooth(self, vx, vy, vz):
                now = self.get_clock().now().nanoseconds / 1e9
                if self._last_control_time is None:
                    self._last_control_time = now
                dt = max(0.001, min(0.2, now - self._last_control_time))
                self._last_control_time = now
                max_delta_h = self._max_accel_h * dt
                max_delta_z = self._max_accel_z * dt
                targets = [vx, vy, vz]
                limits = [max_delta_h, max_delta_h, max_delta_z]
                limited = []
                for target, previous, delta in zip(targets, self._filtered_cmd, limits):
                    limited.append(previous + _clamp(target - previous, -delta, delta))
                alpha = _clamp(self._filter_alpha, 0.0, 1.0)
                self._filtered_cmd = [
                    (1.0 - alpha) * self._filtered_cmd[0] + alpha * limited[0],
                    (1.0 - alpha) * self._filtered_cmd[1] + alpha * limited[1],
                    (1.0 - alpha) * self._filtered_cmd[2] + alpha * limited[2],
                ]
                self._filtered_cmd[0], self._filtered_cmd[1] = _clamp_horizontal_speed(
                    self._filtered_cmd[0], self._filtered_cmd[1], self._U_A_h
                )
                self._filtered_cmd[2] = _clamp(self._filtered_cmd[2], -self._U_A_z, self._U_A_z)
                return tuple(self._filtered_cmd)

            def _apply_geofence_projection(self, vx, vy, vz):
                if self._position is None:
                    return vx, vy, vz
                cmd = [vx, vy, vz]
                for i, p in enumerate(self._position):
                    lo = self._room_min[i]
                    hi = self._room_max[i]
                    margin = self._safety_margin
                    if p <= lo + margin and cmd[i] < 0.0:
                        cmd[i] *= max(0.0, (p - lo) / margin)
                    if p >= hi - margin and cmd[i] > 0.0:
                        cmd[i] *= max(0.0, (hi - p) / margin)
                    projected = p + cmd[i] * self._safety_lookahead
                    if projected < lo + 0.25:
                        cmd[i] = max(cmd[i], (lo + 0.25 - p) / self._safety_lookahead)
                    if projected > hi - 0.25:
                        cmd[i] = min(cmd[i], (hi - 0.25 - p) / self._safety_lookahead)
                    if p <= lo + 0.25:
                        cmd[i] = max(cmd[i], 0.5)
                    if p >= hi - 0.25:
                        cmd[i] = min(cmd[i], -0.5)
                return cmd[0], cmd[1], cmd[2]

            def _apply_obstacle_projection(self, vx, vy):
                mode = self._current_obstacle_projection_mode()
                if mode == 'disabled':
                    return vx, vy
                return _project_obstacle_velocity(
                    vx,
                    vy,
                    self._position,
                    self._obstacle,
                    self._safety_lookahead,
                    self._obstacle_margin,
                    mode=mode,
                )

            def _current_obstacle_projection_mode(self):
                if self._obstacle_projection_mode != 'auto':
                    return self._obstacle_projection_mode
                if (
                    self._mode == 'optimal'
                    and self._vf_loader is not None
                    and self._phi_a_reach_name in self._vf_loader.loaded_names
                ):
                    return 'hard_barrier'
                return 'inflated'

            def _scripted_control(self):
                """Follow waypoints toward target at constant speed.

                Navigate to each waypoint in sequence, then loop on the last one
                (the target). Speed is max_speed * speed_fraction.
                """
                cmd = Twist()
                if self._position is None:
                    return cmd

                # Current waypoint
                if self._current_waypoint_idx >= len(self._waypoints):
                    self._current_waypoint_idx = len(self._waypoints) - 1

                wp = self._waypoints[self._current_waypoint_idx]
                dx = wp[0] - self._position[0]
                dy = wp[1] - self._position[1]
                dz = wp[2] - self._position[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                # Advance to next waypoint if close enough
                if dist < 0.5 and self._current_waypoint_idx < len(self._waypoints) - 1:
                    self._current_waypoint_idx += 1
                    wp = self._waypoints[self._current_waypoint_idx]
                    dx = wp[0] - self._position[0]
                    dy = wp[1] - self._position[1]
                    dz = wp[2] - self._position[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                if dist > 0.1:
                    speed = self._max_speed * self._speed_fraction
                    cmd.linear.x = (dx / dist) * speed
                    cmd.linear.y = (dy / dist) * speed
                    cmd.linear.z = (dz / dist) * speed

                return cmd

            def _keyboard_control(self):
                """Republish teleop_twist_keyboard commands."""
                return self._teleop_cmd

            def _optimal_control(self):
                """Target-reaching control for the attacker.

                Use the online HJ reaching value function ``phi_A_reach`` when
                available. Fall back only if that artifact is missing.
                """
                cmd = Twist()
                if self._position is None:
                    return cmd

                x_A, y_A, z_A = self._position
                if (
                    self._vf_loader is not None
                    and self._phi_a_reach_name in self._vf_loader.loaded_names
                ):
                    cmd.linear.x, cmd.linear.y = extract_attacker_hj_reaching_command(
                        self._vf_loader,
                        self._phi_a_reach_name,
                        x_A,
                        y_A,
                        self._U_A_h,
                        gradient_deadband=self._hj_gradient_deadband,
                        descent_tolerance=self._hj_descent_tolerance,
                    )
                    target_alt = self.get_parameter('target_altitude').value
                    dz = target_alt - z_A
                    cmd.linear.z = max(-self._U_A_z, min(self._U_A_z, 2.0 * dz))
                    return cmd
                return self._goal_seeking_fallback(x_A, y_A, z_A)

            def _goal_seeking_fallback(self, x_A, y_A, z_A):
                """Simple goal-seeking: fly toward target center at max speed."""
                cmd = Twist()
                target_x, target_y = self._obstacle_aware_target_point(x_A, y_A)
                dx = target_x - x_A
                dy = target_y - y_A
                dist_h = math.sqrt(dx * dx + dy * dy)
                if dist_h > 0.1:
                    cmd.linear.x = (dx / dist_h) * self._U_A_h
                    cmd.linear.y = (dy / dist_h) * self._U_A_h
                target_alt = self.get_parameter('target_altitude').value
                dz = target_alt - z_A
                cmd.linear.z = max(-self._U_A_z, min(self._U_A_z, 2.0 * dz))
                return cmd

            def _obstacle_aware_target_point(self, x_A, y_A):
                """Choose a target-directed intermediate point around the box obstacle."""
                target_x, target_y = self._target_center
                obs = self._obstacle
                margin = self._obstacle_margin
                pre_x = obs['x_min'] - margin - 0.5
                post_x = obs['x_max'] + margin + 0.5
                if not (x_A < post_x and target_x > obs['x_max']):
                    return target_x, target_y

                bottom_y = max(self._room_min[1] + self._safety_margin, obs['y_min'] - margin - 0.5)
                top_y = min(self._room_max[1] - self._safety_margin, obs['y_max'] + margin + 0.5)
                # The paper target lies below the obstacle's top half, so the
                # lower corridor keeps the initial command pointed toward the
                # target y-coordinate instead of away from it.
                corridor_y = bottom_y if target_y <= 0.5 * (obs['y_min'] + obs['y_max']) else top_y

                if x_A < pre_x:
                    return pre_x, corridor_y
                return post_x, corridor_y

        rclpy.init(args=args)
        node = AttackerControllerNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('attacker_controller: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
