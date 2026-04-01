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
            VehicleLocalPosition,
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

                qos_sub = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )

                # Subscribers
                self.cmd_vel_sub = self.create_subscription(
                    Twist, cmd_vel_topic, self._cmd_vel_callback, 10
                )
                # Use VehicleLocalPosition to detect DDS connection —
                # avoids px4_msgs/VehicleStatus payload-size mismatch across PX4 versions.
                for suffix in ['vehicle_local_position', 'vehicle_local_position_v1']:
                    self.create_subscription(
                        VehicleLocalPosition,
                        f'{fmu_prefix}fmu/out/{suffix}',
                        self._local_pos_callback,
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
                # In SITL, offboard mode and arm commands work reliably once DDS is live.
                # We assume success after a small number of sent commands rather than
                # waiting for VehicleStatus (which has a payload-size mismatch in this setup).
                self._offboard_sent = 0   # count of DO_SET_MODE commands sent
                self._arm_sent = 0        # count of arm commands sent
                self._armed = False       # set True after arm commands are sent
                self._offboard_engaged = False  # set True after offboard commands sent
                self._px4_connected = False
                self._last_cmd_time = 0.0
                self._startup_count = 0
                # Takeoff is skipped: drones spawn at their operating altitude in Gazebo.
                self._takeoff_done = True

                # 20Hz heartbeat timer (offboard mode requires continuous commands)
                self._timer = self.create_timer(0.05, self._timer_callback)

                self.get_logger().info(
                    f'PX4 adapter started for vehicle {self.vehicle_id}, '
                    f'listening on {cmd_vel_topic}'
                )

            def _cmd_vel_callback(self, msg: Twist):
                """Store latest velocity command."""
                self._cmd_vel = msg

            def _local_pos_callback(self, msg: VehicleLocalPosition):
                """Detect PX4 DDS connection from local position heartbeat."""
                if not self._px4_connected:
                    self._px4_connected = True
                    self.get_logger().info('PX4 DDS connection established')

            def _timer_callback(self):
                """20Hz heartbeat: publish offboard mode and velocity setpoints."""
                # Always publish offboard control mode to maintain heartbeat
                self._publish_offboard_control_mode()

                # Publish velocity setpoint (ENU -> NED conversion)
                self._publish_velocity_setpoint()

                # Phase 1: Wait for PX4 DDS connection (local_position received)
                if not self._px4_connected:
                    self._startup_count += 1
                    if self._startup_count % 100 == 0:  # log every 5s
                        self.get_logger().info(
                            f'Waiting for PX4 DDS connection... ({self._startup_count // 20}s)'
                        )
                    return

                # Phase 2: Send offboard mode heartbeat for 2s before requesting mode switch
                if self._offboard_setpoint_count < 40:
                    self._offboard_setpoint_count += 1
                    return

                # Phase 3: Request offboard mode (3 attempts, 2s apart, then assume done)
                if not self._offboard_engaged:
                    now = self.get_clock().now().nanoseconds / 1e9
                    if self._offboard_sent == 0 or now - self._last_cmd_time >= 2.0:
                        self._engage_offboard_mode()
                        self._last_cmd_time = now
                        self._offboard_sent += 1
                        if self._offboard_sent >= 3:
                            self._offboard_engaged = True
                            self.get_logger().info('Offboard mode requested (3 commands sent)')
                    return

                # Phase 4: Force-arm (2 attempts, 2s apart, then assume armed)
                if not self._armed:
                    now = self.get_clock().now().nanoseconds / 1e9
                    if self._arm_sent == 0 or now - self._last_cmd_time >= 2.0:
                        self._arm()
                        self._last_cmd_time = now
                        self._arm_sent += 1
                        if self._arm_sent >= 2:
                            self._armed = True
                            self.get_logger().info('Armed (force-arm commands sent)')
                    return

                # Phase 5: Forwarding cmd_vel to PX4 offboard velocity control.

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
                """
                msg = TrajectorySetpoint()

                if self._armed:
                    # Forward cmd_vel from game controller (ENU -> NED conversion)
                    # Clamp to game speed limits: U_D_h=6.0 m/s, U_D_z=4.0 m/s
                    max_h = 6.0
                    max_v = 4.0
                    vx = max(-max_h, min(max_h, self._cmd_vel.linear.x))
                    vy = max(-max_h, min(max_h, self._cmd_vel.linear.y))
                    vz = max(-max_v, min(max_v, self._cmd_vel.linear.z))
                    # ENU -> NED: north=ENU_y, east=ENU_x, down=-ENU_z
                    msg.velocity[0] = vy
                    msg.velocity[1] = vx
                    msg.velocity[2] = -vz
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
                """Convert ENU coordinates to NED."""
                return y_enu, x_enu, -z_enu

            @staticmethod
            def ned_to_enu(x_ned, y_ned, z_ned):
                """Convert NED coordinates to ENU."""
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
