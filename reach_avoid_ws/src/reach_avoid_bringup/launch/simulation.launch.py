"""Launch file for full reach-avoid simulation.

Uses new Gazebo (gz sim / Harmonic) with reach_avoid_arena world.
PX4 SITL with gz_x500 spawns x500_defender and x500_attacker drones.
"""

import os
import socket
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_path(candidates, fallback):
    for p in candidates:
        if os.path.exists(p):
            return p
    return fallback


def _get_display():
    """Return a working DISPLAY value, or None for headless.

    Priority:
    1. Xvfb / native socket if /tmp/.X11-unix/X<n> exists for current DISPLAY
    2. host.docker.internal:0 (Docker Desktop / Windows XLaunch via TCP)
    3. None (headless)
    """
    display = os.environ.get('DISPLAY', '')
    if display:
        # Check if the Unix socket exists for this display number
        num = display.split(':', 1)[-1].split('.')[0]
        if os.path.exists(f'/tmp/.X11-unix/X{num}'):
            return display

    # Try Docker Desktop Windows host (VcXsrv / XLaunch)
    try:
        host = socket.gethostbyname('host.docker.internal')
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((host, 6000))
        s.close()
        return 'host.docker.internal:0'
    except Exception:
        pass

    # Fallback: Xvfb on :1 if it's running
    if os.path.exists('/tmp/.X11-unix/X1'):
        return ':1'

    return None


def get_px4_path():
    """Find PX4-Autopilot path from env or workspace."""
    if os.environ.get("PX4_AUTOPILOT") and os.path.isdir(os.environ["PX4_AUTOPILOT"]):
        return os.environ["PX4_AUTOPILOT"]
    for candidate in [
        "/opt/PX4-Autopilot",
        "/workspaces/ros2_ws/src/PX4-Autopilot",
        os.path.join(os.path.expanduser("~"), "PX4-Autopilot"),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.expanduser("~"), "PX4-Autopilot")


def generate_launch_description():
    sim_pkg = get_package_share_directory('reach_avoid_sim')
    bringup_pkg = get_package_share_directory('reach_avoid_bringup')

    attacker_mode_arg = DeclareLaunchArgument(
        'attacker_mode',
        default_value='optimal',
        description='Attacker control mode: scripted, keyboard, optimal, switchable',
    )

    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(sim_pkg, 'worlds', 'reach_avoid_arena.sdf'),
        description='Path to Gazebo SDF world file',
    )

    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value=get_px4_path(),
        description='Path to PX4-Autopilot directory',
    )

    # New Gazebo (gz sim / Harmonic)
    # PX4 models/plugins/server_config needed when PX4 spawns x500 via create service
    px4_dir = get_px4_path()
    px4_models = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'models')
    px4_worlds = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'worlds')
    px4_plugins = os.path.join(px4_dir, 'build', 'px4_sitl_default', 'src', 'modules', 'simulation', 'gz_plugins')
    px4_server_config = os.path.join(px4_dir, 'src', 'modules', 'simulation', 'gz_bridge', 'server.config')
    gazebo_env = os.environ.copy()
    gazebo_env['GZ_SIM_RESOURCE_PATH'] = ':'.join(filter(None, [
        gazebo_env.get('GZ_SIM_RESOURCE_PATH', ''),
        px4_models,
        px4_worlds,
    ]))
    gazebo_env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = ':'.join(filter(None, [
        gazebo_env.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
        px4_plugins,
    ]))
    gazebo_env['GZ_SIM_SERVER_CONFIG_PATH'] = px4_server_config
    gazebo_env['LIBGL_ALWAYS_SOFTWARE'] = '1'
    # Detect working X11 display (handles Docker/WSL2/XLaunch setups)
    active_display = _get_display()
    gz_cmd = ['gz', 'sim', '-r']
    if active_display:
        # Set DISPLAY in environment so all child processes (Gazebo, RViz, etc.) inherit it
        os.environ['DISPLAY'] = active_display
        gazebo_env['DISPLAY'] = active_display
        gazebo_env['QT_X11_NO_MITSHM'] = '1'  # avoids shared-memory X11 issues in containers
    else:
        gz_cmd.append('-s')  # headless server mode (no GUI)
    gz_cmd.append(LaunchConfiguration('world_file'))

    gazebo = ExecuteProcess(
        cmd=gz_cmd,
        output='screen',
        additional_env=gazebo_env,
    )

    # ros_gz_bridge: clock + model poses (YAML config maps GZ /world/... to ROS /clock, /model/.../pose)
    # Note: ros_gz_bridge has no launch file in Humble; run parameter_bridge directly with config
    gz_bridge_config = os.path.join(bringup_pkg, 'config', 'gz_bridge.yaml')
    gz_bridge = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '--ros-args', '-p', f'config_file:={gz_bridge_config}',
        ],
        output='screen',
    )

    xrce_dds_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    def add_px4_processes(context):
        px4_dir = context.perform_substitution(LaunchConfiguration('px4_dir'))
        px4_bin = os.path.join(px4_dir, 'build', 'px4_sitl_default', 'bin', 'px4')
        if not os.path.isfile(px4_bin):
            raise FileNotFoundError(
                f'PX4 binary not found at {px4_bin}. '
                'Build with: cd PX4-Autopilot && make distclean && make px4_sitl gz_x500'
            )
        px4_models = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'models')
        px4_worlds = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'worlds')
        px4_plugins = os.path.join(px4_dir, 'build', 'px4_sitl_default', 'src', 'modules', 'simulation', 'gz_plugins')
        gz_resource_path = ':'.join(filter(None, [
            os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
            px4_models,
            px4_worlds,
        ]))
        env = os.environ.copy()
        env.update({
            'PX4_GZ_STANDALONE': '1',
            'PX4_SIM_MODEL': 'gz_x500',
            'PX4_SYS_AUTOSTART': '4001',
            'PX4_GZ_WORLD': 'reach_avoid_arena',
            'PX4_GZ_NO_FOLLOW': '1',
            'PX4_GZ_MODELS': px4_models,
            'PX4_GZ_WORLDS': px4_worlds,
            'GZ_SIM_RESOURCE_PATH': gz_resource_path,
            'GZ_SIM_SYSTEM_PLUGIN_PATH': ':'.join(filter(None, [
                os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
                px4_plugins,
            ])),
            # --- GPS and EKF2 ---
            'PX4_PARAM_SYS_HAS_GPS': '1',
            'PX4_PARAM_COM_ARM_WO_GPS': '1',
            'PX4_PARAM_EKF2_GPS_CTRL': '7',
            'PX4_PARAM_EKF2_HGT_REF': '1',
            'PX4_PARAM_EKF2_BARO_CTRL': '1',
            # --- Magnetometer (keep enabled for heading, relax checks) ---
            'PX4_PARAM_EKF2_MAG_TYPE': '1',     # Automatic mag fusion
            'PX4_PARAM_EKF2_MAG_CHECK': '0',
            'PX4_PARAM_COM_ARM_MAG_STR': '0',
            # --- Disable ALL failsafes for SITL ---
            'PX4_PARAM_CBRK_SUPPLY_CHK': '894281',
            'PX4_PARAM_CBRK_FLIGHTTERM': '121212',
            'PX4_PARAM_COM_RCL_EXCEPT': '4',
            'PX4_PARAM_NAV_DLL_ACT': '0',
            'PX4_PARAM_NAV_RCL_ACT': '0',
            'PX4_PARAM_COM_OBL_RC_ACT': '0',
            'PX4_PARAM_COM_DISARM_PRFLT': '-1',
            'PX4_PARAM_COM_FLT_TIME_MAX': '0',
            'PX4_PARAM_FD_FAIL_P': '0',
            'PX4_PARAM_FD_FAIL_R': '0',
            'PX4_PARAM_FD_ACT': '0',
            'PX4_PARAM_COM_IMB_PROP_ACT': '0',
        })
        # Do NOT set PX4_GZ_MODEL_NAME - PX4 will spawn models as x500_1, x500_2
        # Arena: 45x25m, walls at x=0,45 and y=0,25. Obstacle x=[15,20], y=[5,20]. Target x=[38,45], y=[10,15]
        defender_env = {
            **env,
            'PX4_GZ_MODEL_POSE': '5.0,12.5,3.0',
            'PX4_UXRCE_DDS_NS': 'defender',
        }
        attacker_env = {
            **env,
            'PX4_GZ_MODEL_POSE': '5.0,20.0,3.0',
            'PX4_UXRCE_DDS_NS': 'attacker',
        }
        return [
            TimerAction(
                period=6.0,
                actions=[
                    ExecuteProcess(
                        cmd=[px4_bin, '-i', '1'],
                        cwd=px4_dir,
                        additional_env=defender_env,
                        output='screen',
                    ),
                    ExecuteProcess(
                        cmd=[px4_bin, '-i', '2'],
                        cwd=px4_dir,
                        additional_env=attacker_env,
                        output='screen',
                    ),
                ],
            ),
        ]

    px4_processes = OpaqueFunction(function=add_px4_processes)

    px4_adapter_defender = Node(
        package='reach_avoid_sim',
        executable='px4_adapter',
        name='px4_adapter_defender',
        parameters=[{
            'vehicle_id': 1,
            'cmd_vel_topic': '/defender/cmd_vel',
            'fmu_topic_prefix': 'defender',
        }],
        output='screen',
    )

    px4_adapter_attacker = Node(
        package='reach_avoid_sim',
        executable='px4_adapter',
        name='px4_adapter_attacker',
        parameters=[{
            'vehicle_id': 2,
            'cmd_vel_topic': '/attacker/cmd_vel',
            'fmu_topic_prefix': 'attacker',
        }],
        output='screen',
    )

    ground_truth_relay = Node(
        package='reach_avoid_sim',
        executable='ground_truth_relay',
        name='ground_truth_relay',
        parameters=[{
            'defender_model_name': 'x500_1',
            'attacker_model_name': 'x500_2',
            'world_name': 'reach_avoid_arena',
            'publish_rate': 50.0,
        }],
        output='screen',
    )

    # Attacker waypoints: flat list [x1,y1,z1, x2,y2,z2, ...] navigating around
    # the obstacle (x=[15,20], y=[5,20]) by going below y=5.
    attacker_waypoints = [
        5.0, 12.5, 10.0,    # Starting area
        12.0, 12.5, 10.0,   # Approach obstacle
        12.0, 3.0, 10.0,    # Go around obstacle (below y=5)
        25.0, 3.0, 10.0,    # Past obstacle
        25.0, 12.5, 10.0,   # Re-center
        41.5, 12.5, 10.0,   # Target center
    ]

    vf_default = _find_path([
        os.path.join(os.path.expanduser('~'), 'ws', 'data', 'value_functions'),
        '/workspace/data/value_functions',
    ], '/workspace/data/value_functions')

    attacker_controller = Node(
        package='attacker_controller',
        executable='attacker_node',
        name='attacker_controller',
        parameters=[{
            'mode': LaunchConfiguration('attacker_mode'),
            'max_speed': 2.0,
            'speed_fraction': 0.8,
            'target_x': 41.5,
            'target_y': 12.5,
            'target_z': 10.0,
            'waypoints': attacker_waypoints,
            'value_function_dir': vf_default,
            'target_altitude': 10.0,
        }],
        output='screen',
    )

    # Delay px4_adapter, ground_truth_relay, attacker_controller until after PX4 has
    # spawned (6s) and EKF2 has received sensor data and converged (~12s more).
    # Arm before EKF2 ready causes "ekf2 missing data" / "heading estimate invalid".
    delayed_controllers = TimerAction(
        period=18.0,
        actions=[
            px4_adapter_defender,
            px4_adapter_attacker,
            ground_truth_relay,
            attacker_controller,
        ],
    )

    # Kill any stale processes from previous runs before starting fresh
    px4_rootfs = os.path.join(
        get_px4_path(), 'build', 'px4_sitl_default', 'rootfs'
    )
    cleanup = ExecuteProcess(
        cmd=['bash', '-c',
             # Use -x (exact name) so these pkill commands never kill this bash script itself
             'pkill -9 -x MicroXRCEAgent 2>/dev/null; '
             'pkill -9 -x px4 2>/dev/null; '
             'pkill -9 -x kinematic_sim 2>/dev/null; '
             'fuser -k 8888/udp 2>/dev/null; '
             f'rm -f /tmp/px4_lock-1 /tmp/px4_lock-2; '
             f'rm -rf {px4_rootfs}/1 {px4_rootfs}/2; '
             'sleep 1.5'],
        output='screen',
    )

    return LaunchDescription([
        attacker_mode_arg,
        world_file_arg,
        px4_dir_arg,
        cleanup,
        gazebo,
        gz_bridge,
        TimerAction(period=1.0, actions=[xrce_dds_agent]),  # brief delay after cleanup
        px4_processes,
        delayed_controllers,
    ])
