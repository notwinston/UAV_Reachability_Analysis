"""Ground truth relay ROS2 node.

Reads drone positions from Gazebo's model-pose stream when available.
Falls back to PX4 vehicle_local_position DDS topics for setups without the
Gazebo pose bridge.
"""

def main(args=None):
    try:
        import math
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from tf2_msgs.msg import TFMessage

        try:
            from px4_msgs.msg import VehicleLocalPosition
            _HAS_PX4_MSGS = True
        except ImportError:
            _HAS_PX4_MSGS = False

        class GroundTruthRelayNode(Node):
            def __init__(self):
                super().__init__('ground_truth_relay')

                self.declare_parameter('defender_model_name', 'x500_1')
                self.declare_parameter('attacker_model_name', 'x500_2')
                self.declare_parameter('world_name', 'reach_avoid_arena')
                self.declare_parameter('publish_rate', 50.0)
                # Spawn positions (must match simulation.launch.py PX4_GZ_MODEL_POSE)
                self.declare_parameter('defender_spawn_x', 5.0)
                self.declare_parameter('defender_spawn_y', 12.5)
                self.declare_parameter('defender_spawn_z', 3.0)
                self.declare_parameter('attacker_spawn_x', 5.0)
                self.declare_parameter('attacker_spawn_y', 20.0)
                self.declare_parameter('attacker_spawn_z', 3.0)

                self._defender_model = self.get_parameter('defender_model_name').value
                self._attacker_model = self.get_parameter('attacker_model_name').value
                publish_rate = self.get_parameter('publish_rate').value

                self._defender_spawn = [
                    self.get_parameter('defender_spawn_x').value,
                    self.get_parameter('defender_spawn_y').value,
                    self.get_parameter('defender_spawn_z').value,
                ]
                self._attacker_spawn = [
                    self.get_parameter('attacker_spawn_x').value,
                    self.get_parameter('attacker_spawn_y').value,
                    self.get_parameter('attacker_spawn_z').value,
                ]

                # Publishers
                self.defender_pose_pub = self.create_publisher(PoseStamped, '/defender/state', 10)
                self.defender_vel_pub = self.create_publisher(TwistStamped, '/defender/velocity', 10)
                self.attacker_pose_pub = self.create_publisher(PoseStamped, '/attacker/state', 10)
                self.attacker_vel_pub = self.create_publisher(TwistStamped, '/attacker/velocity', 10)
                self.create_subscription(TFMessage, '/gz/pose_info', self._gz_pose_cb, 10)

                qos_px4 = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )

                if _HAS_PX4_MSGS:
                    self.create_subscription(
                        VehicleLocalPosition,
                        '/defender/fmu/out/vehicle_local_position_v1',
                        self._defender_cb, qos_px4)
                    self.create_subscription(
                        VehicleLocalPosition,
                        '/attacker/fmu/out/vehicle_local_position_v1',
                        self._attacker_cb, qos_px4)

                    self.get_logger().info(
                        'Using Gazebo /gz/pose_info for ground truth with PX4 local-position fallback'
                    )
                else:
                    self.get_logger().error('px4_msgs not available — cannot relay ground truth')

                self._latest_defender_pose = None
                self._latest_attacker_pose = None
                self._latest_defender_vel = None
                self._latest_attacker_vel = None
                self._last_gz_samples = {}
                self._have_gz_defender = False
                self._have_gz_attacker = False
                self._timer = self.create_timer(1.0 / publish_rate, self._publish_timer)

                self.get_logger().info(
                    f'Ground truth relay started: '
                    f'defender={self._defender_model} spawn={self._defender_spawn}, '
                    f'attacker={self._attacker_model} spawn={self._attacker_spawn}')

            def _gz_pose_cb(self, msg: TFMessage):
                for transform in msg.transforms:
                    name = transform.child_frame_id
                    if self._defender_model in name:
                        self._latest_defender_pose, self._latest_defender_vel = \
                            self._convert_gz_transform(transform, 'defender')
                        self._have_gz_defender = True
                    elif self._attacker_model in name:
                        self._latest_attacker_pose, self._latest_attacker_vel = \
                            self._convert_gz_transform(transform, 'attacker')
                        self._have_gz_attacker = True

            def _convert_gz_transform(self, transform, key):
                pose = PoseStamped()
                pose.header.frame_id = 'world'
                pose.pose.position.x = float(transform.transform.translation.x)
                pose.pose.position.y = float(transform.transform.translation.y)
                pose.pose.position.z = float(transform.transform.translation.z)
                pose.pose.orientation = transform.transform.rotation

                now = self.get_clock().now().nanoseconds / 1e9
                vel = TwistStamped()
                vel.header.frame_id = 'world'
                previous = self._last_gz_samples.get(key)
                current = (now, pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
                if previous is not None:
                    dt = max(1e-3, current[0] - previous[0])
                    vel.twist.linear.x = (current[1] - previous[1]) / dt
                    vel.twist.linear.y = (current[2] - previous[2]) / dt
                    vel.twist.linear.z = (current[3] - previous[3]) / dt
                self._last_gz_samples[key] = current
                return pose, vel

            def _convert(self, msg, spawn):
                """Convert PX4 local position (NED, relative to spawn) to absolute Gazebo frame.

                PX4 NED: x=North, y=East, z=Down (relative to takeoff/home)
                Gazebo ENU: x=East, y=North, z=Up
                Spawn is in Gazebo ENU coordinates.
                """
                pose = PoseStamped()
                pose.header.frame_id = 'world'
                # NED -> ENU: swap x<->y, negate z
                pose.pose.position.x = spawn[0] + float(msg.y)   # Gazebo x (East) = spawn_east + NED_y (East)
                pose.pose.position.y = spawn[1] + float(msg.x)   # Gazebo y (North) = spawn_north + NED_x (North)
                pose.pose.position.z = spawn[2] + float(-msg.z)   # Gazebo z (Up) = spawn_up + (-NED_z) (Up)

                vel = TwistStamped()
                vel.header.frame_id = 'world'
                vel.twist.linear.x = float(msg.vy)   # ENU vx (East) = NED vy (East)
                vel.twist.linear.y = float(msg.vx)   # ENU vy (North) = NED vx (North)
                vel.twist.linear.z = float(-msg.vz)   # ENU vz (Up) = -NED vz (Down)
                return pose, vel

            def _defender_cb(self, msg):
                if self._have_gz_defender:
                    return
                if msg.z_valid or msg.timestamp > 0:
                    self._latest_defender_pose, self._latest_defender_vel = \
                        self._convert(msg, self._defender_spawn)

            def _attacker_cb(self, msg):
                if self._have_gz_attacker:
                    return
                if msg.z_valid or msg.timestamp > 0:
                    self._latest_attacker_pose, self._latest_attacker_vel = \
                        self._convert(msg, self._attacker_spawn)

            def _publish_timer(self):
                now = self.get_clock().now()
                if self._latest_defender_pose is not None:
                    self._latest_defender_pose.header.stamp = now.to_msg()
                    self.defender_pose_pub.publish(self._latest_defender_pose)
                    if self._latest_defender_vel:
                        self._latest_defender_vel.header.stamp = now.to_msg()
                        self.defender_vel_pub.publish(self._latest_defender_vel)
                if self._latest_attacker_pose is not None:
                    self._latest_attacker_pose.header.stamp = now.to_msg()
                    self.attacker_pose_pub.publish(self._latest_attacker_pose)
                    if self._latest_attacker_vel:
                        self._latest_attacker_vel.header.stamp = now.to_msg()
                        self.attacker_vel_pub.publish(self._latest_attacker_vel)

        rclpy.init(args=args)
        node = GroundTruthRelayNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('ground_truth_relay: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
