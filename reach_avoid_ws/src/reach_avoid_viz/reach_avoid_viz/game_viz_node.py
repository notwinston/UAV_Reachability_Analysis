"""Game visualization ROS2 node.

Publishes MarkerArray to /game/markers for RViz display of drones,
target region, obstacles, and capture zone.
"""

from __future__ import annotations

import yaml
from pathlib import Path


# Game geometry defaults (from game_params.yaml / arena SDF)
DEFAULT_TARGET = {"x_min": 6.0, "x_max": 8.0, "y_min": 3.0, "y_max": 5.0}
DEFAULT_OBSTACLES = [
    {"x_min": 3.0, "x_max": 4.0, "y_min": 2.0, "y_max": 6.0, "z_min": 0.0, "z_max": 4.0},
]
DEFAULT_ROOM = {"x_min": 0.0, "x_max": 45.0, "y_min": 0.0, "y_max": 25.0, "z_min": 0.0, "z_max": 20.0}
DEFAULT_D_H = 3.0
DEFAULT_D_Z = 1.0


def _load_game_params(path: str) -> dict:
    """Load game params YAML if available."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Point, PoseStamped
        from std_msgs.msg import String, ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
        from builtin_interfaces.msg import Duration

        class GameVizNode(Node):
            """Publishes visualization markers for the reach-avoid game."""

            def __init__(self):
                super().__init__("game_viz")

                self.declare_parameter(
                    "game_params_file", "/workspace/config/game_params.yaml"
                )
                self.declare_parameter("capture_distance_horizontal", -1.0)
                self.declare_parameter("capture_distance_vertical", -1.0)
                params_file = self.get_parameter("game_params_file").value
                gp = _load_game_params(params_file)

                # Extract geometry
                self._room = gp.get("room", DEFAULT_ROOM)
                target = gp.get("target_region", DEFAULT_TARGET)
                self._target = target
                self._obstacles = gp.get("obstacles", DEFAULT_OBSTACLES)
                capture = gp.get("capture", {})
                self._d_h = capture.get("d_h", DEFAULT_D_H)
                self._d_z = capture.get("d_z", DEFAULT_D_Z)
                capture_d_h = float(self.get_parameter("capture_distance_horizontal").value)
                capture_d_z = float(self.get_parameter("capture_distance_vertical").value)
                if capture_d_h > 0.0:
                    self._d_h = capture_d_h
                if capture_d_z > 0.0:
                    self._d_z = capture_d_z

                # State storage
                self._defender_pos = None
                self._attacker_pos = None
                self._game_status = ""

                # Subscribers
                self.create_subscription(
                    PoseStamped, "/defender/state", self._defender_cb, 10
                )
                self.create_subscription(
                    PoseStamped, "/attacker/state", self._attacker_cb, 10
                )
                self.create_subscription(
                    String, "/game/status", self._status_cb, 10
                )

                # Publisher
                self._marker_pub = self.create_publisher(
                    MarkerArray, "/game/markers", 10
                )

                # Timer at 10Hz
                self._timer = self.create_timer(0.1, self._publish_markers)
                self._marker_id = 0

                self.get_logger().info("Game visualization node started")

            def _defender_cb(self, msg: PoseStamped):
                self._defender_pos = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )

            def _attacker_cb(self, msg: PoseStamped):
                self._attacker_pos = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )

            def _status_cb(self, msg: String):
                self._game_status = msg.data

            def _publish_markers(self):
                """Build and publish all visualization markers."""
                self._marker_id = 0
                ma = MarkerArray()

                # Static markers: room, target, obstacles
                ma.markers.append(self._make_room_outline_marker())
                ma.markers.append(self._make_target_marker())
                for i, obs in enumerate(self._obstacles):
                    ma.markers.append(self._make_obstacle_marker(obs, i))

                # Drone markers
                if self._defender_pos is not None:
                    ma.markers.append(self._make_drone_marker(
                        self._defender_pos, "defender",
                        ColorRGBA(r=0.2, g=0.2, b=1.0, a=0.9),
                    ))

                if self._attacker_pos is not None:
                    ma.markers.append(self._make_drone_marker(
                        self._attacker_pos, "attacker",
                        ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9),
                    ))

                # Capture zone (centered on defender)
                if self._defender_pos is not None:
                    ma.markers.append(self._make_capture_zone_marker())
                    ma.markers.append(self._make_label_marker(
                        self._defender_pos, "defender_label", "Defender",
                        ColorRGBA(r=0.2, g=0.5, b=1.0, a=1.0),
                    ))
                if self._attacker_pos is not None:
                    ma.markers.append(self._make_label_marker(
                        self._attacker_pos, "attacker_label", "Attacker",
                        ColorRGBA(r=1.0, g=0.35, b=0.35, a=1.0),
                    ))
                ma.markers.append(self._make_status_marker())

                self._marker_pub.publish(ma)

            def _next_id(self) -> int:
                mid = self._marker_id
                self._marker_id += 1
                return mid

            def _make_drone_marker(self, pos, ns, color):
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = ns
                m.id = self._next_id()
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = pos[0]
                m.pose.position.y = pos[1]
                m.pose.position.z = pos[2]
                m.pose.orientation.w = 1.0
                m.scale.x = 0.5
                m.scale.y = 0.5
                m.scale.z = 0.5
                m.color = color
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                return m

            def _make_room_outline_marker(self):
                room = self._room
                x_min = room.get("x_min", 0.0)
                x_max = room.get("x_max", 45.0)
                y_min = room.get("y_min", 0.0)
                y_max = room.get("y_max", 25.0)
                z_min = room.get("z_min", 0.0)
                z_max = room.get("z_max", 20.0)
                corners = {
                    "swb": (x_min, y_min, z_min),
                    "seb": (x_max, y_min, z_min),
                    "nwb": (x_min, y_max, z_min),
                    "neb": (x_max, y_max, z_min),
                    "swt": (x_min, y_min, z_max),
                    "set": (x_max, y_min, z_max),
                    "nwt": (x_min, y_max, z_max),
                    "net": (x_max, y_max, z_max),
                }
                edges = [
                    ("swb", "seb"), ("seb", "neb"), ("neb", "nwb"), ("nwb", "swb"),
                    ("swt", "set"), ("set", "net"), ("net", "nwt"), ("nwt", "swt"),
                    ("swb", "swt"), ("seb", "set"), ("neb", "net"), ("nwb", "nwt"),
                ]
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "room"
                m.id = self._next_id()
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = 0.08
                m.color = ColorRGBA(r=0.85, g=0.9, b=1.0, a=0.85)
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                for start_key, end_key in edges:
                    start = corners[start_key]
                    end = corners[end_key]
                    m.points.append(Point(x=start[0], y=start[1], z=start[2]))
                    m.points.append(Point(x=end[0], y=end[1], z=end[2]))
                return m

            def _make_target_marker(self):
                t = self._target
                x_min = t.get("x_min", 6.0)
                x_max = t.get("x_max", 8.0)
                y_min = t.get("y_min", 3.0)
                y_max = t.get("y_max", 5.0)
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                sx = x_max - x_min
                sy = y_max - y_min

                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "target"
                m.id = self._next_id()
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = cx
                m.pose.position.y = cy
                m.pose.position.z = 0.05
                m.pose.orientation.w = 1.0
                m.scale.x = sx
                m.scale.y = sy
                m.scale.z = 0.1
                m.color = ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.5)
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                return m

            def _make_obstacle_marker(self, obs, idx):
                x_min = obs.get("x_min", 0.0)
                x_max = obs.get("x_max", 1.0)
                y_min = obs.get("y_min", 0.0)
                y_max = obs.get("y_max", 1.0)
                z_min = obs.get("z_min", 0.0)
                z_max = obs.get("z_max", 4.0)

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                cz = (z_min + z_max) / 2.0
                sx = x_max - x_min
                sy = y_max - y_min
                sz = z_max - z_min

                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "obstacle"
                m.id = self._next_id()
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = cx
                m.pose.position.y = cy
                m.pose.position.z = cz
                m.pose.orientation.w = 1.0
                m.scale.x = sx
                m.scale.y = sy
                m.scale.z = sz
                m.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6)
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                return m

            def _make_capture_zone_marker(self):
                """Wireframe cylinder centered on defender showing capture zone."""
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "capture_zone"
                m.id = self._next_id()
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = self._defender_pos[0]
                m.pose.position.y = self._defender_pos[1]
                m.pose.position.z = self._defender_pos[2]
                m.pose.orientation.w = 1.0
                # Cylinder: diameter = 2 * d_h, height = 2 * d_z
                m.scale.x = 2.0 * self._d_h
                m.scale.y = 2.0 * self._d_h
                m.scale.z = 2.0 * self._d_z
                m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.2)
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                return m

            def _make_label_marker(self, pos, ns, text, color):
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = ns
                m.id = self._next_id()
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                m.pose.position.x = pos[0]
                m.pose.position.y = pos[1]
                m.pose.position.z = pos[2] + 0.9
                m.pose.orientation.w = 1.0
                m.scale.z = 0.7
                m.color = color
                m.text = text
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                return m

            def _make_status_marker(self):
                room = self._room
                status = self._game_status if self._game_status else "RUNNING"
                m = Marker()
                m.header.frame_id = "world"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "status"
                m.id = self._next_id()
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                m.pose.position.x = room.get("x_min", 0.0) + 2.0
                m.pose.position.y = room.get("y_min", 0.0) + 2.0
                m.pose.position.z = room.get("z_max", 20.0) - 1.5
                m.pose.orientation.w = 1.0
                m.scale.z = 0.8
                m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
                m.text = f"Status: {status}"
                m.lifetime = Duration(sec=0, nanosec=200_000_000)
                return m

        rclpy.init(args=args)
        node = GameVizNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print("game_viz: rclpy not available, running in stub mode")


if __name__ == "__main__":
    main()
