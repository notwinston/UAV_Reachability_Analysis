"""Attacker controller ROS2 node.

Supports four control modes:
  - scripted:   Follow waypoints toward target region at constant speed.
  - keyboard:   Republish teleop_twist_keyboard input.
  - optimal:    STUB - will use value functions from SW-2.
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

        class AttackerControllerNode(Node):
            """Attacker controller with multiple control modes."""

            def __init__(self):
                super().__init__('attacker_controller')

                # Parameters
                self.declare_parameter('mode', 'scripted')
                self.declare_parameter('max_speed', 0.5)
                self.declare_parameter('speed_fraction', 0.8)
                self.declare_parameter('target_x', 7.0)
                self.declare_parameter('target_y', 4.0)
                self.declare_parameter('target_z', 2.0)
                self.declare_parameter('waypoints', [0.0])

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
                """Optimal control using value functions (STUB).

                This will be completed in SW-4 when value functions from SW-2
                are available.
                """
                self.get_logger().warn(
                    'Optimal mode requires value functions from SW-2. '
                    'Using zero velocity.',
                    throttle_duration_sec=5.0,
                )
                return Twist()

        rclpy.init(args=args)
        node = AttackerControllerNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('attacker_controller: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
