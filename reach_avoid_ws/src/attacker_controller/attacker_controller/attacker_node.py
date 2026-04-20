"""Attacker controller ROS2 node.

Supports four control modes:
  - scripted:   Follow waypoints toward target region at constant speed.
  - keyboard:   Republish teleop_twist_keyboard input.
  - optimal:    Game-theoretic HJ value function bang-bang control.
  - switchable: Start in scripted mode, switch via ROS2 service at runtime.
"""

import math


def _clamp(value, low, high):
    return max(low, min(high, value))


def _clamp_horizontal_speed(x, y, limit):
    speed = math.hypot(x, y)
    if speed <= limit or speed < 1e-9:
        return x, y
    scale = limit / speed
    return x * scale, y * scale


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

                if self._mode == 'optimal':
                    vf_dir = self.get_parameter('value_function_dir').value
                    try:
                        loader = ValueFunctionLoader(vf_dir)
                        if 'phi_h' in loader.loaded_names and 'phi_z' in loader.loaded_names:
                            self._vf_loader = loader
                            h_params = loader.get_params('phi_h')
                            z_params = loader.get_params('phi_z')
                            self._U_A_h = h_params.get('U_A_h', 3.0)
                            self._U_A_z = z_params.get('U_A_z', 2.0)
                            self.get_logger().info(
                                f'Optimal mode: loaded VFs from {vf_dir}, '
                                f'U_A_h={self._U_A_h}, U_A_z={self._U_A_z}'
                            )
                        else:
                            self.get_logger().warn(
                                f'Optimal mode using default attacker limits; '
                                f'phi_h/phi_z not both available. Loaded: {loader.loaded_names}'
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
                if self._position is None:
                    return vx, vy
                x, y = self._position[0], self._position[1]
                obs = self._obstacle
                margin = self._obstacle_margin
                px = x + vx * self._safety_lookahead
                py = y + vy * self._safety_lookahead
                near_now = (
                    obs['x_min'] - margin <= x <= obs['x_max'] + margin
                    and obs['y_min'] - margin <= y <= obs['y_max'] + margin
                )
                near_next = (
                    obs['x_min'] - margin <= px <= obs['x_max'] + margin
                    and obs['y_min'] - margin <= py <= obs['y_max'] + margin
                )
                if not (near_now or near_next):
                    return vx, vy
                distances = {
                    'left': abs(x - obs['x_min']),
                    'right': abs(x - obs['x_max']),
                    'bottom': abs(y - obs['y_min']),
                    'top': abs(y - obs['y_max']),
                }
                side = min(distances, key=distances.get)
                inside = obs['x_min'] <= x <= obs['x_max'] and obs['y_min'] <= y <= obs['y_max']
                push = 1.0 if inside else 0.0
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
                """Target-reaching optimal control for the attacker.

                The checked-in horizontal value functions are too coarse to use
                as a reliable online attacker policy. In optimal mode the
                attacker therefore saturates toward the target center, which is
                the target-reaching objective used by the reach-avoid game.
                """
                cmd = Twist()
                if self._position is None:
                    return cmd

                x_A, y_A, z_A = self._position
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
