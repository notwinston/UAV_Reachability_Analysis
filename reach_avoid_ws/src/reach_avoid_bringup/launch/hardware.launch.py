"""Launch file for hardware reach-avoid game with Crazyflie drones.

Launches Crazyswarm2 adapter (replaces PX4 adapter for real hardware),
defender controller, attacker controller, safety monitor, and game
visualization with optional RViz2.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    bringup_pkg = get_package_share_directory('reach_avoid_bringup')
    viz_pkg = get_package_share_directory('reach_avoid_viz')

    # Hardware params file
    hw_params_file = os.path.join(bringup_pkg, 'config', 'hardware_params.yaml')

    # Launch arguments
    attacker_mode_arg = DeclareLaunchArgument(
        'attacker_mode',
        default_value='optimal',
        description='Attacker control mode: scripted, keyboard, optimal, switchable',
    )

    value_function_dir_arg = DeclareLaunchArgument(
        'value_function_dir',
        default_value='/workspace/data/value_functions/',
        description='Path to directory containing value function .npz files',
    )

    defender_uri_arg = DeclareLaunchArgument(
        'defender_uri',
        default_value='radio://0/80/2M/E7E7E7E701',
        description='Crazyflie URI for defender drone',
    )

    attacker_uri_arg = DeclareLaunchArgument(
        'attacker_uri',
        default_value='radio://0/80/2M/E7E7E7E702',
        description='Crazyflie URI for attacker drone',
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 visualization',
    )

    # Crazyswarm2 adapter (replaces PX4 adapter for real hardware)
    crazyswarm_adapter = Node(
        package='reach_avoid_hw',
        executable='crazyswarm_adapter',
        name='crazyswarm_adapter',
        parameters=[
            hw_params_file,
            {
                'defender_uri': LaunchConfiguration('defender_uri'),
                'attacker_uri': LaunchConfiguration('attacker_uri'),
            },
        ],
        output='screen',
    )

    # Defender controller (reach-track control with value functions)
    defender_controller = Node(
        package='reach_avoid_controller',
        executable='defender_node',
        name='defender_controller',
        parameters=[
            hw_params_file,
            {
                'value_function_dir': LaunchConfiguration('value_function_dir'),
            },
        ],
        output='screen',
    )

    # Attacker controller
    attacker_controller = Node(
        package='attacker_controller',
        executable='attacker_node',
        name='attacker_controller',
        parameters=[
            hw_params_file,
            {
                'mode': LaunchConfiguration('attacker_mode'),
            },
        ],
        output='screen',
    )

    # Safety monitor
    safety_monitor = Node(
        package='reach_avoid_hw',
        executable='safety_monitor',
        name='safety_monitor',
        parameters=[hw_params_file],
        output='screen',
    )

    # Game visualization
    game_viz = Node(
        package='reach_avoid_viz',
        executable='game_viz',
        name='game_viz',
        parameters=[{
            'game_params_file': '/workspace/config/game_params.yaml',
        }],
        output='screen',
    )

    # RViz2 (conditional)
    rviz_config = os.path.join(viz_pkg, 'config', 'game_viz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        attacker_mode_arg,
        value_function_dir_arg,
        defender_uri_arg,
        attacker_uri_arg,
        use_rviz_arg,
        crazyswarm_adapter,
        defender_controller,
        attacker_controller,
        safety_monitor,
        game_viz,
        rviz,
    ])
