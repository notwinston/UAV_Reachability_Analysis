"""Safety monitor ROS2 node.

Monitors drone states for safety violations: geofence, speed limits,
altitude bounds, and inter-drone distance. Triggers emergency landing
on any violation. Designed to be fail-safe: defaults to emergency land
on any error.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SafetyConfig:
    """Configuration for safety checks."""
    # Room bounds
    room_x_min: float = 0.0
    room_x_max: float = 45.0
    room_y_min: float = 0.0
    room_y_max: float = 25.0
    room_z_min: float = 0.0
    room_z_max: float = 20.0
    # Safety margins
    geofence_margin: float = 0.5
    altitude_min: float = 0.3
    altitude_ceiling_margin: float = 0.5
    # Speed limits
    max_speed_defender: float = 6.0
    max_speed_attacker: float = 3.0
    speed_tolerance: float = 1.1  # 10% tolerance
    # Inter-drone distance
    min_inter_drone_distance: float = 0.5
    # Watchdog
    state_timeout: float = 1.0


@dataclass
class SafetyViolation:
    """Describes a single safety violation."""
    check_name: str
    message: str
    severity: str = "critical"  # "critical" or "warning"


def check_safety(
    defender_pos: Optional[Tuple[float, float, float]],
    attacker_pos: Optional[Tuple[float, float, float]],
    defender_vel: Optional[Tuple[float, float, float]],
    attacker_vel: Optional[Tuple[float, float, float]],
    config: SafetyConfig,
) -> List[SafetyViolation]:
    """Check all safety conditions. Returns list of violations (empty = safe).

    This is a pure function with no ROS2 dependencies for testability.

    Args:
        defender_pos: (x, y, z) or None if unknown
        attacker_pos: (x, y, z) or None if unknown
        defender_vel: (vx, vy, vz) or None if unknown
        attacker_vel: (vx, vy, vz) or None if unknown
        config: SafetyConfig with room bounds and limits

    Returns:
        List of SafetyViolation objects. Empty list means all checks pass.
    """
    violations = []

    # Geofence check for defender
    if defender_pos is not None:
        gf = _check_geofence(defender_pos, config, "defender")
        violations.extend(gf)

    # Geofence check for attacker
    if attacker_pos is not None:
        gf = _check_geofence(attacker_pos, config, "attacker")
        violations.extend(gf)

    # Altitude check for defender
    if defender_pos is not None:
        alt = _check_altitude(defender_pos[2], config, "defender")
        violations.extend(alt)

    # Altitude check for attacker
    if attacker_pos is not None:
        alt = _check_altitude(attacker_pos[2], config, "attacker")
        violations.extend(alt)

    # Speed check for defender
    if defender_vel is not None:
        sp = _check_speed(defender_vel, config.max_speed_defender, config.speed_tolerance, "defender")
        violations.extend(sp)

    # Speed check for attacker
    if attacker_vel is not None:
        sp = _check_speed(attacker_vel, config.max_speed_attacker, config.speed_tolerance, "attacker")
        violations.extend(sp)

    # Inter-drone distance check
    if defender_pos is not None and attacker_pos is not None:
        dist = _check_inter_drone_distance(defender_pos, attacker_pos, config)
        violations.extend(dist)

    return violations


def _check_geofence(
    pos: Tuple[float, float, float],
    config: SafetyConfig,
    drone_name: str,
) -> List[SafetyViolation]:
    """Check if position is within geofence (room bounds with margin)."""
    violations = []
    x, y, z = pos
    margin = config.geofence_margin

    if x < config.room_x_min + margin:
        violations.append(SafetyViolation(
            "geofence",
            f"{drone_name} x={x:.2f} below min {config.room_x_min + margin:.2f}",
        ))
    if x > config.room_x_max - margin:
        violations.append(SafetyViolation(
            "geofence",
            f"{drone_name} x={x:.2f} above max {config.room_x_max - margin:.2f}",
        ))
    if y < config.room_y_min + margin:
        violations.append(SafetyViolation(
            "geofence",
            f"{drone_name} y={y:.2f} below min {config.room_y_min + margin:.2f}",
        ))
    if y > config.room_y_max - margin:
        violations.append(SafetyViolation(
            "geofence",
            f"{drone_name} y={y:.2f} above max {config.room_y_max - margin:.2f}",
        ))

    return violations


def _check_altitude(
    z: float,
    config: SafetyConfig,
    drone_name: str,
) -> List[SafetyViolation]:
    """Check altitude is within safe range: altitude_min < z < room_z_max - ceiling_margin."""
    violations = []
    z_max = config.room_z_max - config.altitude_ceiling_margin

    if z < config.altitude_min:
        violations.append(SafetyViolation(
            "altitude",
            f"{drone_name} z={z:.2f} below min altitude {config.altitude_min:.2f}",
        ))
    if z > z_max:
        violations.append(SafetyViolation(
            "altitude",
            f"{drone_name} z={z:.2f} above max altitude {z_max:.2f}",
        ))

    return violations


def _check_speed(
    vel: Tuple[float, float, float],
    max_speed: float,
    tolerance: float,
    drone_name: str,
) -> List[SafetyViolation]:
    """Check velocity magnitude against speed limit with tolerance."""
    violations = []
    vx, vy, vz = vel
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    limit = max_speed * tolerance

    if speed > limit:
        violations.append(SafetyViolation(
            "speed",
            f"{drone_name} speed={speed:.2f} exceeds limit {limit:.2f}",
        ))

    return violations


def _check_inter_drone_distance(
    defender_pos: Tuple[float, float, float],
    attacker_pos: Tuple[float, float, float],
    config: SafetyConfig,
) -> List[SafetyViolation]:
    """Check inter-drone distance is above minimum."""
    violations = []
    dx = defender_pos[0] - attacker_pos[0]
    dy = defender_pos[1] - attacker_pos[1]
    dz = defender_pos[2] - attacker_pos[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < config.min_inter_drone_distance:
        violations.append(SafetyViolation(
            "inter_drone_distance",
            f"distance={dist:.2f} below min {config.min_inter_drone_distance:.2f}",
        ))

    return violations


def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from std_msgs.msg import Bool, String
        from std_srvs.srv import Trigger

        class SafetyMonitorNode(Node):
            """ROS2 safety monitor for hardware reach-avoid games.

            Monitors drone positions and velocities, checks geofence, speed,
            altitude, and inter-drone distance. Triggers emergency landing
            on any violation. Fail-safe: defaults to emergency on errors.
            """

            def __init__(self):
                super().__init__('safety_monitor')

                # Parameters
                self.declare_parameter('room_x_min', 0.0)
                self.declare_parameter('room_x_max', 45.0)
                self.declare_parameter('room_y_min', 0.0)
                self.declare_parameter('room_y_max', 25.0)
                self.declare_parameter('room_z_min', 0.0)
                self.declare_parameter('room_z_max', 20.0)
                self.declare_parameter('geofence_margin', 0.5)
                self.declare_parameter('altitude_min', 0.3)
                self.declare_parameter('altitude_ceiling_margin', 0.5)
                self.declare_parameter('max_speed_defender', 6.0)
                self.declare_parameter('max_speed_attacker', 3.0)
                self.declare_parameter('min_inter_drone_distance', 0.5)
                self.declare_parameter('state_timeout', 1.0)
                self.declare_parameter('safety_rate', 50.0)

                # Build config from parameters
                self._config = SafetyConfig(
                    room_x_min=self.get_parameter('room_x_min').value,
                    room_x_max=self.get_parameter('room_x_max').value,
                    room_y_min=self.get_parameter('room_y_min').value,
                    room_y_max=self.get_parameter('room_y_max').value,
                    room_z_min=self.get_parameter('room_z_min').value,
                    room_z_max=self.get_parameter('room_z_max').value,
                    geofence_margin=self.get_parameter('geofence_margin').value,
                    altitude_min=self.get_parameter('altitude_min').value,
                    altitude_ceiling_margin=self.get_parameter('altitude_ceiling_margin').value,
                    max_speed_defender=self.get_parameter('max_speed_defender').value,
                    max_speed_attacker=self.get_parameter('max_speed_attacker').value,
                    min_inter_drone_distance=self.get_parameter('min_inter_drone_distance').value,
                    state_timeout=self.get_parameter('state_timeout').value,
                )

                safety_rate = self.get_parameter('safety_rate').value

                # State storage
                self._defender_pos = None
                self._attacker_pos = None
                self._defender_vel = None
                self._attacker_vel = None
                self._last_defender_time = None
                self._last_attacker_time = None

                # Armed state: safety monitor must be armed to allow flight
                self._armed = False
                self._emergency = False

                # Subscribers
                self.create_subscription(
                    PoseStamped, '/defender/state', self._defender_state_cb, 10
                )
                self.create_subscription(
                    PoseStamped, '/attacker/state', self._attacker_state_cb, 10
                )
                self.create_subscription(
                    TwistStamped, '/defender/velocity', self._defender_vel_cb, 10
                )
                self.create_subscription(
                    TwistStamped, '/attacker/velocity', self._attacker_vel_cb, 10
                )

                # Publishers
                self._emergency_pub = self.create_publisher(Bool, '/safety/emergency', 10)
                self._status_pub = self.create_publisher(String, '/safety/status', 10)

                # Services
                self.create_service(Trigger, '/safety/arm', self._arm_callback)
                self.create_service(Trigger, '/safety/disarm', self._disarm_callback)

                # Timer for safety checks at safety_rate Hz
                self._timer = self.create_timer(1.0 / safety_rate, self._safety_check_loop)

                self.get_logger().info(
                    f'SafetyMonitorNode started at {safety_rate}Hz, '
                    f'room=[{self._config.room_x_min},{self._config.room_x_max}]x'
                    f'[{self._config.room_y_min},{self._config.room_y_max}]x'
                    f'[{self._config.room_z_min},{self._config.room_z_max}]'
                )

            def _defender_state_cb(self, msg: PoseStamped):
                self._defender_pos = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )
                self._last_defender_time = self.get_clock().now()

            def _attacker_state_cb(self, msg: PoseStamped):
                self._attacker_pos = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )
                self._last_attacker_time = self.get_clock().now()

            def _defender_vel_cb(self, msg: TwistStamped):
                self._defender_vel = (
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.linear.z,
                )

            def _attacker_vel_cb(self, msg: TwistStamped):
                self._attacker_vel = (
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.linear.z,
                )

            def _arm_callback(self, request, response):
                self._armed = True
                self._emergency = False
                response.success = True
                response.message = 'Safety monitor armed'
                self.get_logger().info('Safety monitor ARMED')
                return response

            def _disarm_callback(self, request, response):
                self._armed = False
                response.success = True
                response.message = 'Safety monitor disarmed'
                self.get_logger().info('Safety monitor DISARMED')
                return response

            def _safety_check_loop(self):
                """50Hz safety check loop. Fail-safe: emergency on any error."""
                try:
                    if not self._armed:
                        self._publish_status('DISARMED')
                        return

                    # Watchdog: check state freshness
                    now = self.get_clock().now()
                    timeout_ns = int(self._config.state_timeout * 1e9)

                    if self._last_defender_time is not None:
                        dt = (now - self._last_defender_time).nanoseconds
                        if dt > timeout_ns:
                            self._trigger_emergency(
                                f'Defender state timeout: {dt*1e-9:.2f}s'
                            )
                            return

                    if self._last_attacker_time is not None:
                        dt = (now - self._last_attacker_time).nanoseconds
                        if dt > timeout_ns:
                            self._trigger_emergency(
                                f'Attacker state timeout: {dt*1e-9:.2f}s'
                            )
                            return

                    # Run safety checks
                    violations = check_safety(
                        self._defender_pos,
                        self._attacker_pos,
                        self._defender_vel,
                        self._attacker_vel,
                        self._config,
                    )

                    if violations:
                        details = '; '.join(v.message for v in violations)
                        self._trigger_emergency(details)
                    else:
                        # All clear
                        if self._emergency:
                            # Stay in emergency until rearmed
                            self._publish_emergency(True)
                            self._publish_status('EMERGENCY (latch)')
                        else:
                            self._publish_emergency(False)
                            self._publish_status('OK')

                except Exception as e:
                    # Fail-safe: any error triggers emergency
                    self._trigger_emergency(f'Safety check error: {e}')

            def _trigger_emergency(self, reason: str):
                """Trigger emergency landing."""
                if not self._emergency:
                    self.get_logger().error(f'EMERGENCY: {reason}')
                self._emergency = True
                self._publish_emergency(True)
                self._publish_status(f'EMERGENCY: {reason}')

            def _publish_emergency(self, is_emergency: bool):
                msg = Bool()
                msg.data = is_emergency
                self._emergency_pub.publish(msg)

            def _publish_status(self, status: str):
                msg = String()
                msg.data = status
                self._status_pub.publish(msg)

        rclpy.init(args=args)
        node = SafetyMonitorNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        print('safety_monitor: rclpy not available, running in stub mode')


if __name__ == '__main__':
    main()
