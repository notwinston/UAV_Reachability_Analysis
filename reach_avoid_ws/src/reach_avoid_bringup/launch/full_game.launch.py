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
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_path(candidates, fallback):
    """Return the first existing path from candidates, or fallback."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return fallback


def generate_launch_description():
    # Package directories
    bringup_pkg = get_package_share_directory('reach_avoid_bringup')
    viz_pkg = get_package_share_directory('reach_avoid_viz')

    # Discover value_function_dir and game_params_file across common locations
    vf_default = _find_path([
        '/workspaces/UAV_Reachability_Analysis/data/value_functions',
        os.path.join(os.path.expanduser('~'), 'ws', 'data', 'value_functions'),
        '/workspace/data/value_functions',
        '/workspaces/ros2_ws/src/UAV_Reachability_Analysis/value_functions',
    ], '/workspace/data/value_functions')

    gp_default = _find_path([
        '/workspaces/UAV_Reachability_Analysis/config/generated_calibrated_game_params.yaml',
        '/workspaces/UAV_Reachability_Analysis/config/game_params.yaml',
        os.path.join(os.path.expanduser('~'), 'ws', 'config', 'game_params.yaml'),
        '/workspace/config/game_params.yaml',
        '/workspaces/ros2_ws/src/UAV_Reachability_Analysis/config/game_params.yaml',
    ], '/workspace/config/game_params.yaml')

    # Launch arguments
    attacker_mode_arg = DeclareLaunchArgument(
        'attacker_mode',
        default_value='optimal',
        description='Attacker control mode: scripted, keyboard, optimal, switchable',
    )

    value_function_dir_arg = DeclareLaunchArgument(
        'value_function_dir',
        default_value=vf_default,
        description='Path to directory containing value function .npz files',
    )

    game_params_arg = DeclareLaunchArgument(
        'game_params_file',
        default_value=gp_default,
        description='Path to game_params.yaml',
    )

    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value='/opt/PX4-Autopilot',
        description='Path to PX4-Autopilot directory',
    )

    defender_pose_arg = DeclareLaunchArgument(
        'defender_pose',
        default_value='5.0,12.5,3.0',
        description='Initial Gazebo pose for the defender as x,y,z',
    )

    attacker_pose_arg = DeclareLaunchArgument(
        'attacker_pose',
        default_value='5.0,20.0,3.0',
        description='Initial Gazebo pose for the attacker as x,y,z',
    )

    capture_distance_horizontal_arg = DeclareLaunchArgument(
        'capture_distance_horizontal',
        default_value='-1.0',
        description='Override horizontal capture distance; negative keeps VF/config default',
    )

    capture_distance_vertical_arg = DeclareLaunchArgument(
        'capture_distance_vertical',
        default_value='-1.0',
        description='Override vertical capture distance; negative keeps VF/config default',
    )

    # Include simulation.launch.py
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'simulation.launch.py')
        ),
        launch_arguments={
            'attacker_mode': LaunchConfiguration('attacker_mode'),
            'px4_dir': LaunchConfiguration('px4_dir'),
            'defender_pose': LaunchConfiguration('defender_pose'),
            'attacker_pose': LaunchConfiguration('attacker_pose'),
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
    venv_candidates = [
        os.path.join(os.path.expanduser('~'), 'ws', '.venv', 'lib', 'python3.10', 'site-packages'),
        '/workspace/.venv/lib/python3.10/site-packages',
        '/workspaces/ros2_ws/src/UAV_Reachability_Analysis/.venv/lib/python3.10/site-packages',
    ]
    defender_env = {}
    for venv_site in venv_candidates:
        if os.path.isdir(venv_site):
            defender_env['PYTHONPATH'] = venv_site + (':' + os.environ.get('PYTHONPATH', '') if os.environ.get('PYTHONPATH') else '')
            break

    defender_controller = Node(
        package='reach_avoid_controller',
        executable='defender_node',
        name='defender_controller',
        parameters=[{
            'value_function_dir': LaunchConfiguration('value_function_dir'),
            'control_rate': 50.0,
            'pid_gain_z': 8.0,
            'pid_gain_h': 2.0,
            'margin_z_factor': 0.3,
            'margin_h_factor': 0.3,
            'min_hj_closure_fraction': 0.85,
            'command_filter_alpha': 0.35,
            'max_accel_horizontal': 12.0,
            'max_accel_vertical': 8.0,
            'capture_distance_horizontal': LaunchConfiguration('capture_distance_horizontal'),
            'capture_distance_vertical': LaunchConfiguration('capture_distance_vertical'),
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
            'capture_distance_horizontal': LaunchConfiguration('capture_distance_horizontal'),
            'capture_distance_vertical': LaunchConfiguration('capture_distance_vertical'),
        }],
        output='screen',
    )

    trajectory_recorder = Node(
        package='reach_avoid_viz',
        executable='trajectory_recorder',
        name='trajectory_recorder',
        parameters=[{
            'game_params_file': LaunchConfiguration('game_params_file'),
            'output_dir': '/workspaces/UAV_Reachability_Analysis/data/plots/gazebo_runs',
            'sample_stride': 2,
            'autosave_period_sec': 10.0,
            'capture_distance_horizontal': LaunchConfiguration('capture_distance_horizontal'),
            'capture_distance_vertical': LaunchConfiguration('capture_distance_vertical'),
        }],
        output='screen',
    )

    # RViz2 with game visualization config (only when DISPLAY is available)
    rviz_config = os.path.join(viz_pkg, 'config', 'game_viz.rviz')
    has_display = bool(os.environ.get('DISPLAY'))
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(str(has_display).lower()),
    )

    return LaunchDescription([
        attacker_mode_arg,
        value_function_dir_arg,
        game_params_arg,
        px4_dir_arg,
        defender_pose_arg,
        attacker_pose_arg,
        capture_distance_horizontal_arg,
        capture_distance_vertical_arg,
        simulation_launch,
        static_tf,
        delayed_defender,
        game_viz,
        trajectory_recorder,
        rviz,
    ])
