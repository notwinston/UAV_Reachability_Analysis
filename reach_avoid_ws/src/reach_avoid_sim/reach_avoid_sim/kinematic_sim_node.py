"""Kinematic drone simulator - bypasses PX4 entirely.

Defender: double integrator dynamics v_dot = k * (u - v), then pos += v * dt.
Attacker: single integrator dynamics pos += cmd_vel * dt.
Subscribes to cmd_vel topics, publishes state and velocity topics.
No PX4, no arming, no EKF - just demonstrates the reach-avoid algorithms.
"""

def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
        import time as _time

        class KinematicSimNode(Node):
            def __init__(self):
                super().__init__('kinematic_sim')

                # Spawn positions (match simulation.launch.py)
                self.declare_parameter('defender_x', 5.0)
                self.declare_parameter('defender_y', 12.5)
                self.declare_parameter('defender_z', 3.0)
                self.declare_parameter('attacker_x', 5.0)
                self.declare_parameter('attacker_y', 20.0)
                self.declare_parameter('attacker_z', 3.0)
                self.declare_parameter('sim_rate', 50.0)

                # Drone state: [x, y, z]
                self._defender_pos = [
                    self.get_parameter('defender_x').value,
                    self.get_parameter('defender_y').value,
                    self.get_parameter('defender_z').value,
                ]
                self._attacker_pos = [
                    self.get_parameter('attacker_x').value,
                    self.get_parameter('attacker_y').value,
                    self.get_parameter('attacker_z').value,
                ]

                self._defender_vel = [0.0, 0.0, 0.0]
                self._defender_cmd = [0.0, 0.0, 0.0]
                self._attacker_vel = [0.0, 0.0, 0.0]

                # Defender double-integrator gains (from game_params.yaml)
                self._k = [0.7, 0.7, 1.5]  # k_x, k_y, k_z

                sim_rate = self.get_parameter('sim_rate').value
                self._dt = 1.0 / sim_rate

                # Subscribers: velocity commands from controllers
                self.create_subscription(Twist, '/defender/cmd_vel', self._def_cmd_cb, 10)
                self.create_subscription(Twist, '/attacker/cmd_vel', self._att_cmd_cb, 10)

                # Publishers: state (PoseStamped) and velocity (TwistStamped)
                self._def_pose_pub = self.create_publisher(PoseStamped, '/defender/state', 10)
                self._def_vel_pub = self.create_publisher(TwistStamped, '/defender/velocity', 10)
                self._att_pose_pub = self.create_publisher(PoseStamped, '/attacker/state', 10)
                self._att_vel_pub = self.create_publisher(TwistStamped, '/attacker/velocity', 10)

                # Sim loop
                self._timer = self.create_timer(self._dt, self._sim_step)

                self.get_logger().info(
                    f'Kinematic sim started: '
                    f'defender={self._defender_pos}, attacker={self._attacker_pos}, '
                    f'rate={sim_rate}Hz')

            def _def_cmd_cb(self, msg):
                self._defender_cmd = [msg.linear.x, msg.linear.y, msg.linear.z]

            def _att_cmd_cb(self, msg):
                self._attacker_vel = [msg.linear.x, msg.linear.y, msg.linear.z]

            def _sim_step(self):
                now = self.get_clock().now()

                # Defender: double integrator dynamics v_dot = k * (u - v)
                for i in range(3):
                    self._defender_vel[i] += self._dt * self._k[i] * (self._defender_cmd[i] - self._defender_vel[i])
                self._defender_pos[0] += self._defender_vel[0] * self._dt
                self._defender_pos[1] += self._defender_vel[1] * self._dt
                self._defender_pos[2] += self._defender_vel[2] * self._dt

                # Attacker: single integrator (direct velocity)
                for i in range(3):
                    self._attacker_pos[i] += self._attacker_vel[i] * self._dt

                # Clamp to arena bounds [0,45] x [0,25] x [0,20]
                self._defender_pos[0] = max(0.0, min(45.0, self._defender_pos[0]))
                self._defender_pos[1] = max(0.0, min(25.0, self._defender_pos[1]))
                self._defender_pos[2] = max(0.0, min(20.0, self._defender_pos[2]))
                self._attacker_pos[0] = max(0.0, min(45.0, self._attacker_pos[0]))
                self._attacker_pos[1] = max(0.0, min(25.0, self._attacker_pos[1]))
                self._attacker_pos[2] = max(0.0, min(20.0, self._attacker_pos[2]))

                # Zero wall-normal velocity on wall contact (prevent wall-pushing)
                bounds_lo = [0.0, 0.0, 0.0]
                bounds_hi = [45.0, 25.0, 20.0]
                for i in range(3):
                    if self._defender_pos[i] <= bounds_lo[i] and self._defender_vel[i] < 0:
                        self._defender_vel[i] = 0.0
                    if self._defender_pos[i] >= bounds_hi[i] and self._defender_vel[i] > 0:
                        self._defender_vel[i] = 0.0

                # Publish defender state
                dp = PoseStamped()
                dp.header.stamp = now.to_msg()
                dp.header.frame_id = 'world'
                dp.pose.position.x = self._defender_pos[0]
                dp.pose.position.y = self._defender_pos[1]
                dp.pose.position.z = self._defender_pos[2]
                dp.pose.orientation.w = 1.0
                self._def_pose_pub.publish(dp)

                dv = TwistStamped()
                dv.header.stamp = now.to_msg()
                dv.header.frame_id = 'world'
                dv.twist.linear.x = self._defender_vel[0]
                dv.twist.linear.y = self._defender_vel[1]
                dv.twist.linear.z = self._defender_vel[2]
                self._def_vel_pub.publish(dv)

                # Publish attacker state
                ap = PoseStamped()
                ap.header.stamp = now.to_msg()
                ap.header.frame_id = 'world'
                ap.pose.position.x = self._attacker_pos[0]
                ap.pose.position.y = self._attacker_pos[1]
                ap.pose.position.z = self._attacker_pos[2]
                ap.pose.orientation.w = 1.0
                self._att_pose_pub.publish(ap)

                av = TwistStamped()
                av.header.stamp = now.to_msg()
                av.header.frame_id = 'world'
                av.twist.linear.x = self._attacker_vel[0]
                av.twist.linear.y = self._attacker_vel[1]
                av.twist.linear.z = self._attacker_vel[2]
                self._att_vel_pub.publish(av)

        rclpy.init(args=args)
        node = KinematicSimNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('kinematic_sim: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
