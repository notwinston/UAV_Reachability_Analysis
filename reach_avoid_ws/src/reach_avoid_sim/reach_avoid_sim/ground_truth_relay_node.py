"""Ground truth relay ROS2 node.

Subscribes to Gazebo ground truth poses (via ros_gz_bridge) and republishes
as ROS2 PoseStamped/TwistStamped for defender and attacker state topics.
Computes velocity via finite differences when Gazebo does not provide it.
"""

import math

def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
        from builtin_interfaces.msg import Time

        try:
            from ros_gz_interfaces.msg import EntityWrench
            _HAS_GZ_INTERFACES = True
        except ImportError:
            _HAS_GZ_INTERFACES = False

        class GroundTruthRelayNode(Node):
            """Relays Gazebo ground truth to game state topics.

            Subscribes to model pose topics from ros_gz_bridge and publishes
            normalized PoseStamped and TwistStamped messages for each drone.
            """

            def __init__(self):
                super().__init__('ground_truth_relay')

                # Parameters (PX4 spawns as x500_1, x500_2; instance 1=defender, 2=attacker)
                self.declare_parameter('defender_model_name', 'x500_1')
                self.declare_parameter('attacker_model_name', 'x500_2')
                self.declare_parameter('world_name', 'reach_avoid_arena')
                self.declare_parameter('publish_rate', 50.0)

                self._defender_model = self.get_parameter('defender_model_name').value
                self._attacker_model = self.get_parameter('attacker_model_name').value
                world_name = self.get_parameter('world_name').value
                publish_rate = self.get_parameter('publish_rate').value

                # Publishers
                self.defender_pose_pub = self.create_publisher(
                    PoseStamped, '/defender/state', 10
                )
                self.defender_vel_pub = self.create_publisher(
                    TwistStamped, '/defender/velocity', 10
                )
                self.attacker_pose_pub = self.create_publisher(
                    PoseStamped, '/attacker/state', 10
                )
                self.attacker_vel_pub = self.create_publisher(
                    TwistStamped, '/attacker/velocity', 10
                )

                # Subscribe to Gazebo model pose topics via ros_gz_bridge
                # The bridge typically exposes /model/<name>/pose as PoseStamped
                qos_gz = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )

                self.create_subscription(
                    PoseStamped,
                    f'/model/{self._defender_model}/pose',
                    self._defender_pose_callback,
                    qos_gz,
                )
                self.create_subscription(
                    PoseStamped,
                    f'/model/{self._attacker_model}/pose',
                    self._attacker_pose_callback,
                    qos_gz,
                )

                # State for finite-difference velocity estimation
                self._defender_prev_pose = None
                self._defender_prev_time = None
                self._attacker_prev_pose = None
                self._attacker_prev_time = None

                # Timer for periodic publishing (ensures consistent rate)
                self._latest_defender_pose = None
                self._latest_attacker_pose = None
                self._timer = self.create_timer(1.0 / publish_rate, self._publish_timer)

                self.get_logger().info(
                    f'Ground truth relay started: '
                    f'defender={self._defender_model}, '
                    f'attacker={self._attacker_model}'
                )

            def _defender_pose_callback(self, msg: PoseStamped):
                """Store latest defender pose from Gazebo."""
                self._latest_defender_pose = msg

            def _attacker_pose_callback(self, msg: PoseStamped):
                """Store latest attacker pose from Gazebo."""
                self._latest_attacker_pose = msg

            def _publish_timer(self):
                """Publish state and velocity at fixed rate."""
                now = self.get_clock().now()

                if self._latest_defender_pose is not None:
                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = now.to_msg()
                    pose_msg.header.frame_id = 'world'
                    pose_msg.pose = self._latest_defender_pose.pose
                    self.defender_pose_pub.publish(pose_msg)

                    # Compute and publish velocity via finite differences
                    vel_msg = self._compute_velocity(
                        self._latest_defender_pose,
                        self._defender_prev_pose,
                        self._defender_prev_time,
                        now,
                    )
                    if vel_msg is not None:
                        self.defender_vel_pub.publish(vel_msg)

                    self._defender_prev_pose = self._latest_defender_pose
                    self._defender_prev_time = now

                if self._latest_attacker_pose is not None:
                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = now.to_msg()
                    pose_msg.header.frame_id = 'world'
                    pose_msg.pose = self._latest_attacker_pose.pose
                    self.attacker_pose_pub.publish(pose_msg)

                    # Compute and publish velocity via finite differences
                    vel_msg = self._compute_velocity(
                        self._latest_attacker_pose,
                        self._attacker_prev_pose,
                        self._attacker_prev_time,
                        now,
                    )
                    if vel_msg is not None:
                        self.attacker_vel_pub.publish(vel_msg)

                    self._attacker_prev_pose = self._latest_attacker_pose
                    self._attacker_prev_time = now

            def _compute_velocity(self, current_pose, prev_pose, prev_time, current_time):
                """Compute velocity via finite differences on position.

                Returns TwistStamped or None if no previous data.
                """
                if prev_pose is None or prev_time is None:
                    return None

                dt_ns = (current_time.nanoseconds - prev_time.nanoseconds)
                if dt_ns <= 0:
                    return None
                dt = dt_ns / 1e9

                vel_msg = TwistStamped()
                vel_msg.header.stamp = current_time.to_msg()
                vel_msg.header.frame_id = 'world'

                # Linear velocity from position differences
                vel_msg.twist.linear.x = (
                    current_pose.pose.position.x - prev_pose.pose.position.x
                ) / dt
                vel_msg.twist.linear.y = (
                    current_pose.pose.position.y - prev_pose.pose.position.y
                ) / dt
                vel_msg.twist.linear.z = (
                    current_pose.pose.position.z - prev_pose.pose.position.z
                ) / dt

                return vel_msg

        rclpy.init(args=args)
        node = GroundTruthRelayNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('ground_truth_relay: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
