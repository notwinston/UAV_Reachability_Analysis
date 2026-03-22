"""PX4 adapter ROS2 node.

Bridges high-level velocity commands (geometry_msgs/Twist) to PX4 offboard
control messages. Handles ENU <-> NED coordinate conversion, arming, and
offboard mode engagement.
"""

import math

def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from geometry_msgs.msg import Twist
        from px4_msgs.msg import (
            OffboardControlMode,
            TrajectorySetpoint,
            VehicleCommand,
            VehicleOdometry,
            VehicleStatus,
        )

        class PX4AdapterNode(Node):
            """Adapts ROS2 Twist velocity commands to PX4 offboard control."""

            def __init__(self):
                super().__init__('px4_adapter')

                # Parameters
                self.declare_parameter('vehicle_id', 1)
                self.declare_parameter('cmd_vel_topic', '/defender/cmd_vel')
                self.declare_parameter('fmu_topic_prefix', '')
                # fmu_topic_prefix: PX4 UXRCE namespace, e.g. 'defender' -> /defender/fmu/in/...

                self.vehicle_id = self.get_parameter('vehicle_id').value
                cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
                prefix = self.get_parameter('fmu_topic_prefix').value
                fmu_prefix = f'/{prefix}/' if prefix else '/'

                # QoS for PX4 topics (best-effort, transient local)
                qos_px4 = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )

                # Subscribers
                self.cmd_vel_sub = self.create_subscription(
                    Twist, cmd_vel_topic, self._cmd_vel_callback, 10
                )
                qos_sub = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                # Try both vehicle_status and vehicle_status_v3 (PX4 version varies)
                for suffix in ['vehicle_status', 'vehicle_status_v3']:
                    self.create_subscription(
                        VehicleStatus,
                        f'{fmu_prefix}fmu/out/{suffix}',
                        self._vehicle_status_callback,
                        qos_sub,
                    )
                # Odometry for altitude tracking during takeoff
                self.create_subscription(
                    VehicleOdometry,
                    f'{fmu_prefix}fmu/out/vehicle_odometry',
                    self._odometry_callback,
                    qos_sub,
                )

                # Publishers to PX4 (volatile QoS to match PX4's DDS config)
                qos_pub = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self.offboard_mode_pub = self.create_publisher(
                    OffboardControlMode, f'{fmu_prefix}fmu/in/offboard_control_mode', qos_pub
                )
                self.trajectory_pub = self.create_publisher(
                    TrajectorySetpoint, f'{fmu_prefix}fmu/in/trajectory_setpoint', qos_pub
                )
                # VehicleCommand needs RELIABLE QoS for PX4 to receive it
                qos_cmd = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self.vehicle_cmd_pub = self.create_publisher(
                    VehicleCommand, f'{fmu_prefix}fmu/in/vehicle_command', qos_cmd
                )

                # State
                self._cmd_vel = Twist()
                self._offboard_setpoint_count = 0
                self._armed = False
                self._offboard_engaged = False
                self._nav_state = 0
                self._px4_connected = False
                self._last_arm_time = 0.0
                self._arm_attempts = 0
                self._startup_count = 0
                self._takeoff_count = 0
                self._takeoff_done = False
                self._takeoff_altitude = 3.0  # meters above ground
                self._current_z = 0.0  # NED z from odometry (negative = above ground)

                # 20Hz heartbeat timer (offboard mode requires continuous commands)
                self._timer = self.create_timer(0.05, self._timer_callback)

                self.get_logger().info(
                    f'PX4 adapter started for vehicle {self.vehicle_id}, '
                    f'listening on {cmd_vel_topic}'
                )

            def _cmd_vel_callback(self, msg: Twist):
                """Store latest velocity command."""
                self._cmd_vel = msg

            def _vehicle_status_callback(self, msg: VehicleStatus):
                """Track PX4 connection, arming state, and nav state."""
                self._px4_connected = True
                was_armed = self._armed
                self._armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
                self._nav_state = msg.nav_state
                if self._armed and not was_armed:
                    self.get_logger().info('PX4 confirmed ARMED')
                # Check if offboard mode is actually engaged (nav_state 14 = offboard)
                if msg.nav_state == 14 and not self._offboard_engaged:
                    self._offboard_engaged = True
                    self.get_logger().info('PX4 confirmed OFFBOARD mode')

            def _odometry_callback(self, msg: VehicleOdometry):
                """Track current altitude from PX4 odometry (NED frame)."""
                self._current_z = msg.position[2]  # NED: negative = above ground

            def _timer_callback(self):
                """20Hz heartbeat: publish offboard mode and velocity setpoints."""
                # Always publish offboard control mode to maintain heartbeat
                self._publish_offboard_control_mode()

                # Publish velocity setpoint (ENU -> NED conversion)
                self._publish_velocity_setpoint()

                # Wait for PX4 to connect (or assume connected after 15s = 300 ticks)
                # GPS needs time to converge before arming
                self._startup_count += 1
                if not self._px4_connected and self._startup_count < 300:
                    return
                if not self._px4_connected and self._startup_count == 300:
                    self.get_logger().warn('No vehicle_status received, proceeding anyway')
                    self._px4_connected = True

                # Phase 1: Send setpoints for 2s before requesting offboard mode
                if self._offboard_setpoint_count < 40:
                    self._offboard_setpoint_count += 1
                    return

                # Phase 2: Engage offboard mode (retry every 2s until confirmed)
                if not self._offboard_engaged:
                    now = self.get_clock().now().nanoseconds / 1e9
                    if self._arm_attempts == 0 or now - self._last_arm_time >= 2.0:
                        self._engage_offboard_mode()
                        self._last_arm_time = now
                        self._arm_attempts += 1
                        if self._arm_attempts >= 5:
                            self.get_logger().warn(
                                'Offboard mode not confirmed after 5 attempts, proceeding'
                            )
                            self._offboard_engaged = True
                            self._arm_attempts = 0
                    return

                # Phase 3: Arm the vehicle (retry every 2s until confirmed)
                if not self._armed:
                    now = self.get_clock().now().nanoseconds / 1e9
                    if now - self._last_arm_time >= 2.0:
                        self._arm()
                        self._last_arm_time = now
                        self._arm_attempts += 1
                        if self._arm_attempts >= 5:
                            self.get_logger().warn(
                                'Arm not confirmed after 5 attempts, assuming armed'
                            )
                            self._armed = True
                    return

                # Phase 4: Takeoff to target altitude before forwarding cmd_vel
                if not self._takeoff_done:
                    self._takeoff_count += 1
                    # Check if we've reached target altitude (NED: z is negative when above ground)
                    altitude = -self._current_z
                    if altitude >= self._takeoff_altitude * 0.9:
                        self._takeoff_done = True
                        self.get_logger().info(
                            f'Takeoff complete at {altitude:.1f}m, forwarding cmd_vel'
                        )
                    elif self._takeoff_count > 200:  # 10s timeout
                        self._takeoff_done = True
                        self.get_logger().warn(
                            f'Takeoff timeout at {altitude:.1f}m, forwarding cmd_vel'
                        )

            def _publish_offboard_control_mode(self):
                """Publish offboard control mode requesting velocity control."""
                msg = OffboardControlMode()
                msg.position = False
                msg.velocity = True
                msg.acceleration = False
                msg.attitude = False
                msg.body_rate = False
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.offboard_mode_pub.publish(msg)

            def _publish_velocity_setpoint(self):
                """Convert ENU velocity command to NED and publish.

                Game/Gazebo ENU: x=east, y=north, z=up
                PX4 NED:  x=north, y=east, z=down
                Velocities are clamped to prevent motor saturation in simulation.
                """
                msg = TrajectorySetpoint()

                if self._armed and self._takeoff_done:
                    # Normal operation: forward cmd_vel
                    # Match game parameters: U_D_h=6.0 m/s, U_D_z=4.0 m/s
                    max_h = 6.0
                    max_v = 4.0
                    vx = max(-max_h, min(max_h, self._cmd_vel.linear.x))
                    vy = max(-max_h, min(max_h, self._cmd_vel.linear.y))
                    vz = max(-max_v, min(max_v, self._cmd_vel.linear.z))
                    # ENU -> NED coordinate conversion
                    msg.velocity[0] = vy    # NED x (north) = ENU y (north)
                    msg.velocity[1] = vx    # NED y (east) = ENU x (east)
                    msg.velocity[2] = -vz   # NED z (down) = -ENU z (up)
                elif self._armed and not self._takeoff_done:
                    # Takeoff: climb at 1 m/s (NED z = -1.0 = upward)
                    msg.velocity[0] = 0.0
                    msg.velocity[1] = 0.0
                    msg.velocity[2] = -1.0
                else:
                    # Pre-arm: hold zero velocity (required heartbeat for offboard)
                    msg.velocity[0] = 0.0
                    msg.velocity[1] = 0.0
                    msg.velocity[2] = 0.0

                # Set NaN for position/acceleration to indicate velocity-only control
                msg.position[0] = float('nan')
                msg.position[1] = float('nan')
                msg.position[2] = float('nan')
                msg.acceleration[0] = float('nan')
                msg.acceleration[1] = float('nan')
                msg.acceleration[2] = float('nan')
                msg.yaw = 0.0  # Hold heading at 0 (north)
                msg.yawspeed = 0.0

                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(msg)

            def _publish_vehicle_command(self, command: int, param1=0.0, param2=0.0):
                """Publish a VehicleCommand."""
                msg = VehicleCommand()
                msg.param1 = param1
                msg.param2 = param2
                msg.command = command
                msg.target_system = self.vehicle_id + 1  # PX4 instance i has MAV_SYS_ID=i+1
                msg.target_component = 1
                msg.source_system = 1
                msg.source_component = 1
                msg.from_external = True
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.vehicle_cmd_pub.publish(msg)

            def _arm(self):
                """Send force-arm command (bypasses preflight health checks for SITL)."""
                self._publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0,
                    param2=21196.0,  # Force arm magic number
                )
                self.get_logger().info('Force arm command sent')

            def _engage_offboard_mode(self):
                """Send offboard mode command."""
                self._publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    param1=1.0,
                    param2=6.0,  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
                )
                self.get_logger().info('Offboard mode command sent')

            @staticmethod
            def enu_to_ned(x_enu, y_enu, z_enu):
                """Convert ENU coordinates to NED.

                ENU: x=east, y=north, z=up
                NED: x=north, y=east, z=down
                """
                return y_enu, x_enu, -z_enu

            @staticmethod
            def ned_to_enu(x_ned, y_ned, z_ned):
                """Convert NED coordinates to ENU.

                NED: x=north, y=east, z=down
                ENU: x=east, y=north, z=up
                """
                return y_ned, x_ned, -z_ned

        rclpy.init(args=args)
        node = PX4AdapterNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('px4_adapter: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
