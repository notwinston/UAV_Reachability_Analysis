"""Attacker controller ROS2 node.

Supports four control modes:
  - scripted:   Follow waypoints toward target region at constant speed.
  - keyboard:   Republish teleop_twist_keyboard input.
  - optimal:    Game-theoretic HJ value function bang-bang control.
  - switchable: Start in scripted mode, switch via ROS2 service at runtime.
"""

import math

def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
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

                self._mode = self.get_parameter('mode').value
                self._max_speed = self.get_parameter('max_speed').value
                self._speed_fraction = self.get_parameter('speed_fraction').value
                self._target = [
                    self.get_parameter('target_x').value,
                    self.get_parameter('target_y').value,
                    self.get_parameter('target_z').value,
                ]
                waypoints_param = self.get_parameter('waypoints').value
                self._waypoints = self._parse_waypoints(waypoints_param)

                # Current state
                self._position = None
                self._velocity = None
                self._defender_position = None
                self._defender_velocity = None
                self._current_waypoint_idx = 0

                # For switchable mode, track active sub-mode
                self._active_submode = 'scripted'

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
                self._target_center = [41.5, 12.5]  # center of target region [38,45]x[10,15]

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
                            self.get_logger().error(
                                f'Optimal mode requires phi_h and phi_z. '
                                f'Loaded: {loader.loaded_names}'
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

                self.cmd_vel_pub.publish(cmd)

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
                """Game-theoretic optimal control using HJ value function gradients.

                Attacker minimizes the game value function (bang-bang control):
                  Horizontal: d_x = -U_A_h * sign(dPhi_h/dx_A), d_y = -U_A_h * sign(dPhi_h/dy_A)
                  Vertical:   d_z = -U_A_z * sign(dPhi_z/dz_A)
                Falls back to simple goal-seeking if VFs unavailable or gradient is near-zero.
                """
                cmd = Twist()
                if self._position is None:
                    return cmd

                x_A, y_A, z_A = self._position

                # If VFs not loaded or defender state not available, fall back to goal-seeking
                if (self._vf_loader is None
                        or self._defender_position is None
                        or self._defender_velocity is None):
                    return self._goal_seeking_fallback(x_A, y_A, z_A)

                x_D, y_D, z_D = self._defender_position
                vx_D, vy_D, vz_D = self._defender_velocity

                try:
                    # --- Horizontal optimal control from phi_h ---
                    # 6D state: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
                    h_state = np.array([x_D, y_D, vx_D, vy_D, x_A, y_A])
                    h_grad = self._vf_loader.get_gradient('phi_h', h_state)
                    # Attacker minimizes: indices 4 (x_A) and 5 (y_A)
                    grad_xa = h_grad[4]
                    grad_ya = h_grad[5]

                    if abs(grad_xa) < 1e-10 and abs(grad_ya) < 1e-10:
                        # Near-zero gradient: goal-seeking fallback
                        dx = self._target_center[0] - x_A
                        dy = self._target_center[1] - y_A
                        dist_h = math.sqrt(dx * dx + dy * dy)
                        if dist_h > 0.1:
                            cmd.linear.x = (dx / dist_h) * self._U_A_h
                            cmd.linear.y = (dy / dist_h) * self._U_A_h
                    else:
                        cmd.linear.x = -self._U_A_h if grad_xa >= 0 else self._U_A_h
                        cmd.linear.y = -self._U_A_h if grad_ya >= 0 else self._U_A_h

                    # --- Vertical optimal control from phi_z ---
                    # 3D state: [z_D, v_D_z, z_A]
                    v_state = np.array([z_D, vz_D, z_A])
                    v_grad = self._vf_loader.get_gradient('phi_z', v_state)
                    # Attacker minimizes: index 2 (z_A)
                    grad_za = v_grad[2]

                    if abs(grad_za) < 1e-10:
                        # Near-zero gradient: hold target altitude
                        target_alt = self.get_parameter('target_altitude').value
                        dz = target_alt - z_A
                        cmd.linear.z = max(-self._U_A_z, min(self._U_A_z, 2.0 * dz))
                    else:
                        cmd.linear.z = -self._U_A_z if grad_za >= 0 else self._U_A_z

                except Exception as e:
                    self.get_logger().warn(
                        f'Optimal control error: {e}, using goal-seeking',
                        throttle_duration_sec=5.0,
                    )
                    return self._goal_seeking_fallback(x_A, y_A, z_A)

                return cmd

            def _goal_seeking_fallback(self, x_A, y_A, z_A):
                """Simple goal-seeking: fly toward target center at max speed."""
                cmd = Twist()
                dx = self._target_center[0] - x_A
                dy = self._target_center[1] - y_A
                dist_h = math.sqrt(dx * dx + dy * dy)
                if dist_h > 0.1:
                    cmd.linear.x = (dx / dist_h) * self._U_A_h
                    cmd.linear.y = (dy / dist_h) * self._U_A_h
                target_alt = self.get_parameter('target_altitude').value
                dz = target_alt - z_A
                cmd.linear.z = max(-self._U_A_z, min(self._U_A_z, 2.0 * dz))
                return cmd

        rclpy.init(args=args)
        node = AttackerControllerNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('attacker_controller: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
