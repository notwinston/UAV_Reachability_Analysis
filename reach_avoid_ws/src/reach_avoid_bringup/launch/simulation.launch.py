"""Launch file for full reach-avoid simulation.

Uses new Gazebo (gz sim / Harmonic) with reach_avoid_arena world.
PX4 SITL with gz_x500 spawns x500_defender and x500_attacker drones.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
        default_value='scripted',
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
    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r',
            LaunchConfiguration('world_file'),
        ],
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
            # Relax SITL preflight: no GCS, no mag interference, relax heading/mag checks
            'PX4_PARAM_NAV_DLL_ACT': '0',
            'PX4_PARAM_COM_ARM_MAG_STR': '0',
            'PX4_PARAM_EKF2_MAG_CHECK': '0',
            'PX4_PARAM_EKF2_MAG_GATE': '10',  # Lenient mag innovation gate (default 3) for sim
            # Indoor SITL: no GPS, use barometer for height, allow arming without GPS
            'PX4_PARAM_COM_ARM_WO_GPS': '1',
            'PX4_PARAM_SYS_HAS_GPS': '0',
            'PX4_PARAM_EKF2_GPS_CTRL': '0',
            'PX4_PARAM_EKF2_HGT_REF': '2',   # Barometer height reference
            'PX4_PARAM_EKF2_BARO_CTRL': '1',
            'PX4_PARAM_EKF2_MAG_TYPE': '1',   # Automatic mag fusion type
        })
        # Do NOT set PX4_GZ_MODEL_NAME - PX4 will spawn models as x500_1, x500_2
        # Arena: 8x8m, walls at x=0,8 and y=0,8. Obstacle x=[3,4], y=[2,6]. Target x=[6,8], y=[3,5]
        defender_env = {
            **env,
            'PX4_GZ_MODEL_POSE': '1.5,1.5,0.5',
            'PX4_UXRCE_DDS_NS': 'defender',
        }
        attacker_env = {
            **env,
            'PX4_GZ_MODEL_POSE': '1.5,6.5,0.5',
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

    attacker_controller = Node(
        package='attacker_controller',
        executable='attacker_node',
        name='attacker_controller',
        parameters=[{
            'mode': LaunchConfiguration('attacker_mode'),
            'max_speed': 0.5,
            'speed_fraction': 0.8,
            'target_x': 7.0,
            'target_y': 4.0,
            'target_z': 2.0,
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

    return LaunchDescription([
        attacker_mode_arg,
        world_file_arg,
        px4_dir_arg,
        gazebo,
        gz_bridge,
        xrce_dds_agent,
        px4_processes,
        delayed_controllers,
    ])
