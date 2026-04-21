"""PX4 adapter ROS2 node.

Bridges high-level velocity commands (geometry_msgs/Twist) to PX4 offboard
control messages. Handles ENU <-> NED coordinate conversion, arming, and
offboard mode engagement.
"""

import math


def _clamp(value, low, high):
    return max(low, min(high, value))


def _limit_axis_rate(target, previous, max_delta):
    return previous + _clamp(target - previous, -max_delta, max_delta)


def _clamp_horizontal_speed(vx, vy, limit):
    speed = math.hypot(vx, vy)
    if speed <= limit or speed < 1e-9:
        return vx, vy
    scale = limit / speed
    return vx * scale, vy * scale


def _select_control_position(preference, state_position, px4_position):
    """Choose which position estimate should drive safety conditioning.

    Parameters
    ----------
    preference:
        One of ``auto``, ``state_topic``, or ``px4_local_position``.
    state_position:
        Position from the relay / state topic in ENU world coordinates.
    px4_position:
        Position reconstructed from PX4 VehicleLocalPosition in ENU.
    """
    if preference == 'state_topic':
        return state_position if state_position is not None else px4_position
    if preference == 'px4_local_position':
        return px4_position if px4_position is not None else state_position
    if state_position is not None:
        return state_position
    return px4_position


def main(args=None):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from geometry_msgs.msg import PoseStamped, Twist
        from std_msgs.msg import String
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
                self.declare_parameter('state_topic', '')
                self.declare_parameter('fmu_topic_prefix', '')
                self.declare_parameter('spawn_x', 0.0)
                self.declare_parameter('spawn_y', 0.0)
                self.declare_parameter('spawn_z', 0.0)
                self.declare_parameter('target_altitude', 10.0)
                self.declare_parameter('room_x_min', 0.0)
                self.declare_parameter('room_x_max', 45.0)
                self.declare_parameter('room_y_min', 0.0)
                self.declare_parameter('room_y_max', 25.0)
                self.declare_parameter('room_z_min', 0.5)
                self.declare_parameter('room_z_max', 20.0)
                self.declare_parameter('obstacle_x_min', 15.0)
                self.declare_parameter('obstacle_x_max', 20.0)
                self.declare_parameter('obstacle_y_min', 5.0)
                self.declare_parameter('obstacle_y_max', 20.0)
                self.declare_parameter('safety_margin', 1.5)
                self.declare_parameter('obstacle_margin', 3.0)
                self.declare_parameter('safety_lookahead', 1.0)
                self.declare_parameter('command_filter_alpha', 0.35)
                self.declare_parameter('max_accel_horizontal', 1.0)
                self.declare_parameter('max_accel_vertical', 1.0)
                self.declare_parameter('max_speed_horizontal', 3.0)
                self.declare_parameter('max_speed_vertical', 0.6)
                self.declare_parameter('altitude_hold_gain', 0.8)
                self.declare_parameter('altitude_hold_enabled', True)
                self.declare_parameter('position_source_preference', 'px4_local_position')
                # fmu_topic_prefix: PX4 UXRCE namespace, e.g. 'defender' -> /defender/fmu/in/...

                self.vehicle_id = self.get_parameter('vehicle_id').value
                cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
                state_topic = self.get_parameter('state_topic').value
                prefix = self.get_parameter('fmu_topic_prefix').value
                fmu_prefix = f'/{prefix}/' if prefix else '/'
                self._spawn = [
                    float(self.get_parameter('spawn_x').value),
                    float(self.get_parameter('spawn_y').value),
                    float(self.get_parameter('spawn_z').value),
                ]
                self._target_altitude = float(self.get_parameter('target_altitude').value)
                self._room_min = [
                    float(self.get_parameter('room_x_min').value),
                    float(self.get_parameter('room_y_min').value),
                    float(self.get_parameter('room_z_min').value),
                ]
                self._room_max = [
                    float(self.get_parameter('room_x_max').value),
                    float(self.get_parameter('room_y_max').value),
                    float(self.get_parameter('room_z_max').value),
                ]
                self._obstacle = {
                    'x_min': float(self.get_parameter('obstacle_x_min').value),
                    'x_max': float(self.get_parameter('obstacle_x_max').value),
                    'y_min': float(self.get_parameter('obstacle_y_min').value),
                    'y_max': float(self.get_parameter('obstacle_y_max').value),
                }
                self._safety_margin = float(self.get_parameter('safety_margin').value)
                self._obstacle_margin = float(self.get_parameter('obstacle_margin').value)
                self._safety_lookahead = float(self.get_parameter('safety_lookahead').value)
                self._filter_alpha = float(self.get_parameter('command_filter_alpha').value)
                self._max_accel_h = float(self.get_parameter('max_accel_horizontal').value)
                self._max_accel_z = float(self.get_parameter('max_accel_vertical').value)
                self._max_speed_h = float(self.get_parameter('max_speed_horizontal').value)
                self._max_speed_z = float(self.get_parameter('max_speed_vertical').value)
                self._altitude_hold_gain = float(self.get_parameter('altitude_hold_gain').value)
                self._altitude_hold_enabled = bool(self.get_parameter('altitude_hold_enabled').value)
                self._position_source_preference = str(
                    self.get_parameter('position_source_preference').value
                ).strip() or 'px4_local_position'

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
                if state_topic:
                    self.create_subscription(PoseStamped, state_topic, self._state_callback, 10)
                self.create_subscription(String, '/game/status', self._game_status_callback, 10)
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
                self._position_enu = None
                self._px4_position_enu = None
                self._velocity_enu = [0.0, 0.0, 0.0]
                self._filtered_cmd = [0.0, 0.0, 0.0]
                self._last_filter_time = None
                self._terminal_stop = False
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

            def _game_status_callback(self, msg: String):
                status = msg.data.split('|', 1)[0].strip()
                if status in ('CAPTURED', 'ATTACKER_REACHED_TARGET'):
                    self._terminal_stop = True

            def _state_callback(self, msg: PoseStamped):
                self._position_enu = [
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                ]

            def _local_pos_callback(self, msg: VehicleLocalPosition):
                """Detect PX4 DDS connection from local position heartbeat."""
                self._px4_position_enu = [
                    self._spawn[0] + float(msg.y),
                    self._spawn[1] + float(msg.x),
                    self._spawn[2] - float(msg.z),
                ]
                if self._position_enu is None:
                    self._position_enu = list(self._px4_position_enu)
                self._velocity_enu = [
                    float(msg.vy),
                    float(msg.vx),
                    -float(msg.vz),
                ]
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
                    vx, vy, vz = self._safe_filtered_command()
                    # ENU -> NED: north=ENU_y, east=ENU_x, down=-ENU_z
                    msg.velocity[0] = vy
                    msg.velocity[1] = vx
                    msg.velocity[2] = -vz
                else:
                    # Pre-arm: hold zero velocity (required heartbeat for offboard)
                    msg.velocity[0] = 0.0
                    msg.velocity[1] = 0.0
                    msg.velocity[2] = 0.0

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

            def _control_position(self):
                """Choose the position source used for safety conditioning."""
                return _select_control_position(
                    self._position_source_preference,
                    self._position_enu,
                    self._px4_position_enu,
                )

            def _safe_filtered_command(self):
                """Clamp, smooth, and project game commands before PX4 receives them."""
                max_h = self._max_speed_h
                max_v = self._max_speed_z
                control_pos = self._control_position()
                if self._terminal_stop:
                    vx, vy, vz = self._smooth_command(0.0, 0.0, 0.0)
                    return vx, vy, vz
                vx = _clamp(float(self._cmd_vel.linear.x), -max_h, max_h)
                vy = _clamp(float(self._cmd_vel.linear.y), -max_h, max_h)
                vz = _clamp(float(self._cmd_vel.linear.z), -max_v, max_v)
                if not all(math.isfinite(v) for v in (vx, vy, vz)):
                    vx, vy, vz = 0.0, 0.0, 0.0
                vx, vy = _clamp_horizontal_speed(vx, vy, max_h)
                vx, vy, vz = self._apply_geofence_projection(vx, vy, vz, control_pos)
                vx, vy = self._apply_obstacle_projection(vx, vy, control_pos)
                vx, vy = _clamp_horizontal_speed(vx, vy, max_h)
                vz = _clamp(vz, -max_v, max_v)
                vx, vy, vz = self._smooth_command(vx, vy, vz)
                vz = self._altitude_hold_command(vz, control_pos)
                if (
                    self._altitude_hold_enabled
                    and control_pos is not None
                    and control_pos[2] < self._target_altitude - 0.75
                ):
                    vx, vy = 0.0, 0.0
                vx, vy, vz = self._apply_geofence_projection(vx, vy, vz, control_pos)
                vx, vy = self._apply_obstacle_projection(vx, vy, control_pos)
                vx, vy = _clamp_horizontal_speed(vx, vy, max_h)
                return vx, vy, _clamp(vz, -max_v, max_v)

            def _altitude_hold_command(self, requested_vz, position=None):
                """Keep SITL vehicles near their spawn altitude to avoid vertical runaway."""
                if not self._altitude_hold_enabled:
                    return requested_vz
                position = self._control_position() if position is None else position
                if position is None:
                    return requested_vz
                target_z = _clamp(
                    self._target_altitude,
                    self._room_min[2] + 1.0,
                    self._room_max[2] - 1.0,
                )
                error = target_z - position[2]
                hold_vz = _clamp(
                    self._altitude_hold_gain * error,
                    -self._max_speed_z,
                    self._max_speed_z,
                )
                if abs(error) > 0.25:
                    return hold_vz
                return _clamp(requested_vz, -0.25, 0.25)

            def _smooth_command(self, vx, vy, vz):
                now = self.get_clock().now().nanoseconds / 1e9
                if self._last_filter_time is None:
                    self._last_filter_time = now
                dt = max(0.001, min(0.2, now - self._last_filter_time))
                self._last_filter_time = now

                max_delta_h = self._max_accel_h * dt
                max_delta_z = self._max_accel_z * dt
                limited = [
                    _limit_axis_rate(vx, self._filtered_cmd[0], max_delta_h),
                    _limit_axis_rate(vy, self._filtered_cmd[1], max_delta_h),
                    _limit_axis_rate(vz, self._filtered_cmd[2], max_delta_z),
                ]
                alpha = _clamp(self._filter_alpha, 0.0, 1.0)
                self._filtered_cmd = [
                    (1.0 - alpha) * self._filtered_cmd[0] + alpha * limited[0],
                    (1.0 - alpha) * self._filtered_cmd[1] + alpha * limited[1],
                    (1.0 - alpha) * self._filtered_cmd[2] + alpha * limited[2],
                ]
                self._filtered_cmd[0], self._filtered_cmd[1] = _clamp_horizontal_speed(
                    self._filtered_cmd[0], self._filtered_cmd[1], self._max_speed_h
                )
                self._filtered_cmd[2] = _clamp(
                    self._filtered_cmd[2], -self._max_speed_z, self._max_speed_z
                )
                return tuple(self._filtered_cmd)

            def _apply_geofence_projection(self, vx, vy, vz, position=None):
                position = self._control_position() if position is None else position
                if position is None:
                    return vx, vy, vz
                cmd = [vx, vy, vz]
                for i in range(3):
                    p = position[i]
                    lo = self._room_min[i]
                    hi = self._room_max[i]
                    margin = self._safety_margin
                    inward_speed = self._max_speed_z if i == 2 else self._max_speed_h
                    if p < lo + margin:
                        desired = min(inward_speed, max(0.0, (lo + margin - p) / self._safety_lookahead))
                        cmd[i] = max(cmd[i], desired)
                    if p > hi - margin:
                        desired = -min(inward_speed, max(0.0, (p - (hi - margin)) / self._safety_lookahead))
                        cmd[i] = min(cmd[i], desired)
                    if p <= lo + margin and cmd[i] < 0.0:
                        cmd[i] *= max(0.0, (p - lo) / margin)
                    if p >= hi - margin and cmd[i] > 0.0:
                        cmd[i] *= max(0.0, (hi - p) / margin)
                    projected = p + cmd[i] * self._safety_lookahead
                    if projected < lo + 0.25:
                        cmd[i] = max(cmd[i], (lo + 0.25 - p) / self._safety_lookahead)
                    if projected > hi - 0.25:
                        cmd[i] = min(cmd[i], (hi - 0.25 - p) / self._safety_lookahead)
                    if p <= lo + 0.25:
                        cmd[i] = max(cmd[i], 0.5)
                    if p >= hi - 0.25:
                        cmd[i] = min(cmd[i], -0.5)
                return cmd[0], cmd[1], cmd[2]

            def _apply_obstacle_projection(self, vx, vy, position=None):
                position = self._control_position() if position is None else position
                if position is None:
                    return vx, vy
                x, y = position[0], position[1]
                obs = self._obstacle
                margin = self._obstacle_margin
                px = x + vx * self._safety_lookahead
                py = y + vy * self._safety_lookahead
                near_now = (
                    obs['x_min'] - margin <= x <= obs['x_max'] + margin
                    and obs['y_min'] - margin <= y <= obs['y_max'] + margin
                )
                near_next = (
                    obs['x_min'] - margin <= px <= obs['x_max'] + margin
                    and obs['y_min'] - margin <= py <= obs['y_max'] + margin
                )
                if not (near_now or near_next):
                    return vx, vy

                distances = {
                    'left': abs(x - obs['x_min']),
                    'right': abs(x - obs['x_max']),
                    'bottom': abs(y - obs['y_min']),
                    'top': abs(y - obs['y_max']),
                }
                side = min(distances, key=distances.get)
                inside = obs['x_min'] <= x <= obs['x_max'] and obs['y_min'] <= y <= obs['y_max']
                push = 1.0 if inside else 0.0

                if side == 'left':
                    if vx > 0.0:
                        vx = 0.0
                    vx = min(vx, -push)
                elif side == 'right':
                    if vx < 0.0:
                        vx = 0.0
                    vx = max(vx, push)
                elif side == 'bottom':
                    if vy > 0.0:
                        vy = 0.0
                    vy = min(vy, -push)
                elif side == 'top':
                    if vy < 0.0:
                        vy = 0.0
                    vy = max(vy, push)
                return vx, vy

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
