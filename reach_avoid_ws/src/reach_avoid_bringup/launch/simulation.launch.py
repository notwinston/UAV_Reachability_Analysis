"""Launch file for full reach-avoid simulation.

Launches Gazebo with arena world, micro-XRCE-DDS-Agent, PX4 adapters
for defender and attacker, ground truth relay, and attacker controller.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    sim_pkg = get_package_share_directory('reach_avoid_sim')
    bringup_pkg = get_package_share_directory('reach_avoid_bringup')

    # Launch arguments
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

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(bringup_pkg, 'config', 'simulation_params.yaml'),
        description='Path to simulation parameters YAML',
    )

    # Gazebo Harmonic simulation
    gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r',
            LaunchConfiguration('world_file'),
        ],
        output='screen',
    )

    # ros_gz_bridge for clock
    gz_bridge = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
    )

    # micro-XRCE-DDS-Agent (PX4 <-> ROS2 bridge)
    xrce_dds_agent = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'udp4', '-p', '8888',
        ],
        output='screen',
    )

    # PX4 adapter for defender (vehicle_id=1)
    px4_adapter_defender = Node(
        package='reach_avoid_sim',
        executable='px4_adapter',
        name='px4_adapter_defender',
        parameters=[{
            'vehicle_id': 1,
            'cmd_vel_topic': '/defender/cmd_vel',
        }],
        output='screen',
    )

    # PX4 adapter for attacker (vehicle_id=2)
    px4_adapter_attacker = Node(
        package='reach_avoid_sim',
        executable='px4_adapter',
        name='px4_adapter_attacker',
        parameters=[{
            'vehicle_id': 2,
            'cmd_vel_topic': '/attacker/cmd_vel',
        }],
        output='screen',
    )

    # Ground truth relay
    ground_truth_relay = Node(
        package='reach_avoid_sim',
        executable='ground_truth_relay',
        name='ground_truth_relay',
        parameters=[{
            'defender_model_name': 'x500_defender',
            'attacker_model_name': 'x500_attacker',
            'world_name': 'reach_avoid_arena',
            'publish_rate': 50.0,
        }],
        output='screen',
    )

    # Attacker controller
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

    return LaunchDescription([
        attacker_mode_arg,
        world_file_arg,
        config_file_arg,
        gazebo,
        gz_bridge,
        xrce_dds_agent,
        px4_adapter_defender,
        px4_adapter_attacker,
        ground_truth_relay,
        attacker_controller,
    ])
