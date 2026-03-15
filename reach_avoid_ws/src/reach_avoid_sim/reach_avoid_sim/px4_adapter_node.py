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
        )

        class PX4AdapterNode(Node):
            """Adapts ROS2 Twist velocity commands to PX4 offboard control."""

            def __init__(self):
                super().__init__('px4_adapter')

                # Parameters
                self.declare_parameter('vehicle_id', 1)
                self.declare_parameter('cmd_vel_topic', '/defender/cmd_vel')

                self.vehicle_id = self.get_parameter('vehicle_id').value
                cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

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

                # Publishers to PX4
                self.offboard_mode_pub = self.create_publisher(
                    OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4
                )
                self.trajectory_pub = self.create_publisher(
                    TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4
                )
                self.vehicle_cmd_pub = self.create_publisher(
                    VehicleCommand, '/fmu/in/vehicle_command', qos_px4
                )

                # State
                self._cmd_vel = Twist()
                self._offboard_setpoint_count = 0
                self._armed = False
                self._offboard_engaged = False

                # 20Hz heartbeat timer (offboard mode requires continuous commands)
                self._timer = self.create_timer(0.05, self._timer_callback)

                self.get_logger().info(
                    f'PX4 adapter started for vehicle {self.vehicle_id}, '
                    f'listening on {cmd_vel_topic}'
                )

            def _cmd_vel_callback(self, msg: Twist):
                """Store latest velocity command."""
                self._cmd_vel = msg

            def _timer_callback(self):
                """20Hz heartbeat: publish offboard mode and velocity setpoints."""
                # Always publish offboard control mode to maintain heartbeat
                self._publish_offboard_control_mode()

                # Publish velocity setpoint (ENU -> NED conversion)
                self._publish_velocity_setpoint()

                # Arming sequence: send enough offboard setpoints, then engage
                if self._offboard_setpoint_count < 20:
                    self._offboard_setpoint_count += 1
                elif not self._offboard_engaged:
                    self._engage_offboard_mode()
                    self._offboard_engaged = True
                elif not self._armed:
                    self._arm()
                    self._armed = True

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

                ROS2 ENU: x=east, y=north, z=up
                PX4 NED:  x=north, y=east, z=down
                """
                msg = TrajectorySetpoint()

                # ENU -> NED coordinate conversion
                # ENU x (east) -> NED y (east)
                # ENU y (north) -> NED x (north)
                # ENU z (up) -> NED z (down, negated)
                msg.velocity[0] = self._cmd_vel.linear.y   # NED x = ENU north
                msg.velocity[1] = self._cmd_vel.linear.x   # NED y = ENU east
                msg.velocity[2] = -self._cmd_vel.linear.z   # NED z = -ENU up

                # Set NaN for position/acceleration to indicate velocity-only control
                msg.position[0] = float('nan')
                msg.position[1] = float('nan')
                msg.position[2] = float('nan')
                msg.acceleration[0] = float('nan')
                msg.acceleration[1] = float('nan')
                msg.acceleration[2] = float('nan')
                msg.yaw = float('nan')
                msg.yawspeed = self._cmd_vel.angular.z

                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(msg)

            def _publish_vehicle_command(self, command: int, param1=0.0, param2=0.0):
                """Publish a VehicleCommand."""
                msg = VehicleCommand()
                msg.param1 = param1
                msg.param2 = param2
                msg.command = command
                msg.target_system = self.vehicle_id
                msg.target_component = 1
                msg.source_system = 1
                msg.source_component = 1
                msg.from_external = True
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.vehicle_cmd_pub.publish(msg)

            def _arm(self):
                """Send arm command."""
                self._publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0,
                )
                self.get_logger().info('Arm command sent')

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
