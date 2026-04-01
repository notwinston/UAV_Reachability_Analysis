"""Crazyswarm2 adapter ROS2 node.

Bridges high-level velocity commands (geometry_msgs/Twist) to Crazyswarm2
cmdVelocityWorld commands for Crazyflie drones. Subscribes to motion capture
for state estimation and publishes drone states.
"""

import math


def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist, PoseStamped, TwistStamped

        # Try importing Crazyswarm2 / crazyflie_py
        try:
            from crazyflie_py import Crazyswarm
            HAS_CRAZYSWARM = True
        except ImportError:
            HAS_CRAZYSWARM = False

        class CrazyswarmAdapterNode(Node):
            """Adapts ROS2 Twist velocity commands to Crazyswarm2 Crazyflie API.

            Subscribes to /defender/cmd_vel and forwards velocity commands to
            Crazyflie drones via Crazyswarm2. Reads motion capture data and
            publishes drone states.
            """

            def __init__(self):
                super().__init__('crazyswarm_adapter')

                # Parameters
                self.declare_parameter('defender_uri', 'radio://0/80/2M/E7E7E7E701')
                self.declare_parameter('attacker_uri', 'radio://0/80/2M/E7E7E7E702')
                self.declare_parameter('mocap_topic', '/mocap/rigid_bodies')
                self.declare_parameter('takeoff_height', 1.0)
                self.declare_parameter('cmd_rate', 20.0)

                self._defender_uri = self.get_parameter('defender_uri').value
                self._attacker_uri = self.get_parameter('attacker_uri').value
                self._mocap_topic = self.get_parameter('mocap_topic').value
                self._takeoff_height = self.get_parameter('takeoff_height').value
                cmd_rate = self.get_parameter('cmd_rate').value

                # State
                self._cmd_vel = Twist()
                self._defender_cf = None
                self._attacker_cf = None
                self._is_flying = False

                # Previous positions for velocity estimation
                self._prev_defender_pos = None
                self._prev_attacker_pos = None
                self._prev_time = None

                # Initialize Crazyswarm2 if available
                if HAS_CRAZYSWARM:
                    try:
                        self._swarm = Crazyswarm()
                        allcfs = self._swarm.allcfs
                        # Find defender and attacker CFs by URI
                        for cf in allcfs.crazyflies:
                            uri = cf.uri if hasattr(cf, 'uri') else ''
                            if self._defender_uri in str(uri):
                                self._defender_cf = cf
                            elif self._attacker_uri in str(uri):
                                self._attacker_cf = cf
                        if self._defender_cf is None and len(allcfs.crazyflies) > 0:
                            self._defender_cf = allcfs.crazyflies[0]
                        if self._attacker_cf is None and len(allcfs.crazyflies) > 1:
                            self._attacker_cf = allcfs.crazyflies[1]
                        self.get_logger().info(
                            f'Crazyswarm2 initialized: defender={self._defender_cf is not None}, '
                            f'attacker={self._attacker_cf is not None}'
                        )
                    except Exception as e:
                        self.get_logger().error(f'Failed to init Crazyswarm2: {e}')
                        self._swarm = None
                else:
                    self.get_logger().warn(
                        'crazyflie_py not available, running without hardware'
                    )
                    self._swarm = None

                # Subscriber: velocity commands from defender controller
                self.create_subscription(
                    Twist, '/defender/cmd_vel', self._cmd_vel_callback, 10
                )

                # Subscriber: motion capture poses for defender and attacker
                self.create_subscription(
                    PoseStamped, '/mocap/defender', self._mocap_defender_callback, 10
                )
                self.create_subscription(
                    PoseStamped, '/mocap/attacker', self._mocap_attacker_callback, 10
                )

                # Publishers: drone state
                self._defender_state_pub = self.create_publisher(
                    PoseStamped, '/defender/state', 10
                )
                self._attacker_state_pub = self.create_publisher(
                    PoseStamped, '/attacker/state', 10
                )
                self._defender_vel_pub = self.create_publisher(
                    TwistStamped, '/defender/velocity', 10
                )
                self._attacker_vel_pub = self.create_publisher(
                    TwistStamped, '/attacker/velocity', 10
                )

                # Timer for command forwarding at cmd_rate Hz
                self._timer = self.create_timer(1.0 / cmd_rate, self._timer_callback)

                # Services for takeoff/landing
                from std_srvs.srv import Trigger
                self.create_service(Trigger, '/hw/takeoff', self._takeoff_callback)
                self.create_service(Trigger, '/hw/land', self._land_callback)

                self.get_logger().info(
                    f'CrazyswarmAdapterNode started (rate={cmd_rate}Hz, '
                    f'defender_uri={self._defender_uri})'
                )

            def _cmd_vel_callback(self, msg: Twist):
                """Store latest velocity command."""
                self._cmd_vel = msg

            def _mocap_defender_callback(self, msg: PoseStamped):
                """Process motion capture data for defender drone."""
                # Publish state directly
                self._defender_state_pub.publish(msg)

                # Estimate velocity from position differences
                now = self.get_clock().now()
                pos = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )
                if self._prev_defender_pos is not None and self._prev_time is not None:
                    dt = (now - self._prev_time).nanoseconds * 1e-9
                    if dt > 0.001:
                        vel_msg = TwistStamped()
                        vel_msg.header = msg.header
                        vel_msg.twist.linear.x = (pos[0] - self._prev_defender_pos[0]) / dt
                        vel_msg.twist.linear.y = (pos[1] - self._prev_defender_pos[1]) / dt
                        vel_msg.twist.linear.z = (pos[2] - self._prev_defender_pos[2]) / dt
                        self._defender_vel_pub.publish(vel_msg)

                self._prev_defender_pos = pos
                self._prev_time = now

            def _mocap_attacker_callback(self, msg: PoseStamped):
                """Process motion capture data for attacker drone."""
                # Publish state directly
                self._attacker_state_pub.publish(msg)

                # Estimate velocity from position differences
                now = self.get_clock().now()
                pos = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )
                if self._prev_attacker_pos is not None:
                    # Reuse prev_time from defender for simplicity; use msg stamps if needed
                    dt_ns = now.nanoseconds - (self._prev_time.nanoseconds if self._prev_time else now.nanoseconds)
                    dt = dt_ns * 1e-9 if dt_ns > 0 else 0.0
                    if dt > 0.001:
                        vel_msg = TwistStamped()
                        vel_msg.header = msg.header
                        vel_msg.twist.linear.x = (pos[0] - self._prev_attacker_pos[0]) / dt
                        vel_msg.twist.linear.y = (pos[1] - self._prev_attacker_pos[1]) / dt
                        vel_msg.twist.linear.z = (pos[2] - self._prev_attacker_pos[2]) / dt
                        self._attacker_vel_pub.publish(vel_msg)

                self._prev_attacker_pos = pos

            def _timer_callback(self):
                """Forward velocity commands to Crazyflie at cmd_rate Hz."""
                if not self._is_flying or self._defender_cf is None:
                    return

                try:
                    # Crazyswarm2 cmdVelocityWorld: send velocity in world frame
                    vx = self._cmd_vel.linear.x
                    vy = self._cmd_vel.linear.y
                    vz = self._cmd_vel.linear.z
                    yaw_rate = self._cmd_vel.angular.z
                    self._defender_cf.cmdVelocityWorld(
                        [vx, vy, vz], yaw_rate
                    )
                except Exception as e:
                    self.get_logger().error(
                        f'Failed to send velocity command: {e}',
                        throttle_duration_sec=1.0,
                    )

            def _takeoff_callback(self, request, response):
                """Service callback for takeoff."""
                from std_srvs.srv import Trigger
                if self._is_flying:
                    response.success = False
                    response.message = 'Already flying'
                    return response

                if self._defender_cf is not None:
                    try:
                        self._defender_cf.takeoff(
                            targetHeight=self._takeoff_height, duration=2.0
                        )
                        self._is_flying = True
                        response.success = True
                        response.message = f'Takeoff to {self._takeoff_height}m'
                        self.get_logger().info(f'Takeoff initiated: {self._takeoff_height}m')
                    except Exception as e:
                        response.success = False
                        response.message = f'Takeoff failed: {e}'
                        self.get_logger().error(f'Takeoff failed: {e}')
                else:
                    # No hardware, simulate takeoff
                    self._is_flying = True
                    response.success = True
                    response.message = 'Takeoff (no hardware)'
                    self.get_logger().warn('Takeoff called without hardware')

                return response

            def _land_callback(self, request, response):
                """Service callback for landing."""
                from std_srvs.srv import Trigger
                if self._defender_cf is not None:
                    try:
                        self._defender_cf.land(targetHeight=0.05, duration=2.0)
                        self.get_logger().info('Landing initiated')
                    except Exception as e:
                        self.get_logger().error(f'Landing failed: {e}')

                self._is_flying = False
                response.success = True
                response.message = 'Landing'
                return response

        rclpy.init(args=args)
        node = CrazyswarmAdapterNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('crazyswarm_adapter: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
