"""Full game launch: simulation + defender controller + visualization + RViz.

Includes the simulation.launch.py from SW-3 and adds:
- Defender controller node (reach-track control with value functions)
- Game visualization node (MarkerArray publisher)
- RViz2 with game_viz.rviz config
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    bringup_pkg = get_package_share_directory('reach_avoid_bringup')
    viz_pkg = get_package_share_directory('reach_avoid_viz')

    # Launch arguments
    attacker_mode_arg = DeclareLaunchArgument(
        'attacker_mode',
        default_value='scripted',
        description='Attacker control mode: scripted, keyboard, optimal, switchable',
    )

    value_function_dir_arg = DeclareLaunchArgument(
        'value_function_dir',
        default_value='/workspace/data/value_functions/',
        description='Path to directory containing value function .npz files',
    )

    # Include simulation.launch.py from SW-3
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'simulation.launch.py')
        ),
        launch_arguments={
            'attacker_mode': LaunchConfiguration('attacker_mode'),
        }.items(),
    )

    # Defender controller node
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

    # Game visualization node
    game_viz = Node(
        package='reach_avoid_viz',
        executable='game_viz',
        name='game_viz',
        parameters=[{
            'game_params_file': '/workspace/config/game_params.yaml',
        }],
        output='screen',
    )

    # RViz2 with game visualization config
    rviz_config = os.path.join(viz_pkg, 'config', 'game_viz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=None,
    )

    return LaunchDescription([
        attacker_mode_arg,
        value_function_dir_arg,
        simulation_launch,
        defender_controller,
        game_viz,
        rviz,
    ])
