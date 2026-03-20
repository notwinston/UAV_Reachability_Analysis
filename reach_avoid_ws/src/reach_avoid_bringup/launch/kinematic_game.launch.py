"""Lightweight reach-avoid game launch using kinematic simulation (no PX4).

Runs the full game without Gazebo or PX4 — just the controllers and a simple
velocity-integration simulator. Demonstrates the reach-avoid algorithms.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_path(candidates, fallback):
    for p in candidates:
        if os.path.exists(p):
            return p
    return fallback


def generate_launch_description():
    bringup_pkg = get_package_share_directory('reach_avoid_bringup')

    vf_default = _find_path([
        os.path.join(os.path.expanduser('~'), 'ws', 'data', 'value_functions'),
        '/workspace/data/value_functions',
    ], '/workspace/data/value_functions')

    gp_default = _find_path([
        os.path.join(os.path.expanduser('~'), 'ws', 'config', 'game_params.yaml'),
        '/workspace/config/game_params.yaml',
    ], '/workspace/config/game_params.yaml')

    # Launch arguments
    attacker_mode_arg = DeclareLaunchArgument(
        'attacker_mode', default_value='scripted')
    vf_dir_arg = DeclareLaunchArgument(
        'value_function_dir', default_value=vf_default)
    gp_arg = DeclareLaunchArgument(
        'game_params_file', default_value=gp_default)

    # Kinematic simulator (replaces Gazebo + PX4 + adapters + ground_truth_relay)
    kinematic_sim = Node(
        package='reach_avoid_sim',
        executable='kinematic_sim',
        name='kinematic_sim',
        parameters=[{
            'defender_x': 5.0,
            'defender_y': 12.5,
            'defender_z': 3.0,
            'attacker_x': 5.0,
            'attacker_y': 20.0,
            'attacker_z': 3.0,
            'sim_rate': 50.0,
        }],
        output='screen',
    )

    # Attacker waypoints
    attacker_waypoints = [
        5.0, 12.5, 10.0,
        12.0, 12.5, 10.0,
        12.0, 3.0, 10.0,
        25.0, 3.0, 10.0,
        25.0, 12.5, 10.0,
        41.5, 12.5, 10.0,
    ]

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
        }],
        output='screen',
    )

    # Defender controller (delayed to let sim publish initial state)
    defender_controller = Node(
        package='reach_avoid_controller',
        executable='defender_node',
        name='defender_controller',
        parameters=[{
            'value_function_dir': LaunchConfiguration('value_function_dir'),
            'control_rate': 50.0,
            'pid_gain_z': 2.0,
            'pid_gain_h': 2.0,
            'margin_z_factor': 0.3,
            'margin_h_factor': 0.3,
        }],
        output='screen',
    )

    # Static TF
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        name='world_to_map_tf',
    )

    # Game visualization
    game_viz = Node(
        package='reach_avoid_viz',
        executable='game_viz',
        name='game_viz',
        parameters=[{
            'game_params_file': LaunchConfiguration('game_params_file'),
        }],
        output='screen',
    )

    # Delay controllers by 2s to let kinematic_sim start publishing
    delayed = TimerAction(period=2.0, actions=[
        attacker_controller,
        defender_controller,
    ])

    return LaunchDescription([
        attacker_mode_arg,
        vf_dir_arg,
        gp_arg,
        kinematic_sim,
        static_tf,
        game_viz,
        delayed,
    ])
