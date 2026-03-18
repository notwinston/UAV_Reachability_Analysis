"""Full game launch: simulation + defender controller + visualization + RViz.

Includes the simulation.launch.py from SW-3 and adds:
- Defender controller node (reach-track control with value functions)
- Game visualization node (MarkerArray publisher)
- RViz2 with game_viz.rviz config
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
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
        default_value='/workspaces/ros2_ws/src/UAV_Reachability_Analysis/value_functions/',
        description='Path to directory containing value function .npz files',
    )

    game_params_arg = DeclareLaunchArgument(
        'game_params_file',
        default_value='/workspaces/ros2_ws/src/UAV_Reachability_Analysis/config/game_params.yaml',
        description='Path to game_params.yaml',
    )

    # Include simulation.launch.py
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'simulation.launch.py')
        ),
        launch_arguments={
            'attacker_mode': LaunchConfiguration('attacker_mode'),
        }.items(),
    )

    # Static TF: world frame for RViz (markers use frame_id="world")
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        name='world_to_map_tf',
    )

    # Defender controller node (use venv's scipy/numpy to avoid binary incompatibility)
    venv_site = os.path.join(
        '/workspaces/ros2_ws/src/UAV_Reachability_Analysis/.venv',
        'lib', 'python3.10', 'site-packages'
    )
    defender_env = {}
    if os.path.isdir(venv_site):
        defender_env['PYTHONPATH'] = venv_site + (':' + os.environ.get('PYTHONPATH', '') if os.environ.get('PYTHONPATH') else '')

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
        additional_env=defender_env,
        output='screen',
    )

    # Delay defender until after PX4 spawns and ground_truth_relay is publishing
    delayed_defender = TimerAction(period=12.0, actions=[defender_controller])

    # Game visualization node
    game_viz = Node(
        package='reach_avoid_viz',
        executable='game_viz',
        name='game_viz',
        parameters=[{
            'game_params_file': LaunchConfiguration('game_params_file'),
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
        game_params_arg,
        simulation_launch,
        static_tf,
        delayed_defender,
        game_viz,
        rviz,
    ])
