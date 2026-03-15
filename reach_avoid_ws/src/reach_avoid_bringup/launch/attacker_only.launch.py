"""Launch file for attacker controller testing only.

Launches just the attacker controller node for independent testing
without Gazebo, PX4, or the defender.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    attacker_mode_arg = DeclareLaunchArgument(
        'attacker_mode',
        default_value='scripted',
        description='Attacker control mode: scripted, keyboard, optimal, switchable',
    )

    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='0.5',
        description='Attacker maximum speed (m/s)',
    )

    # Attacker controller
    attacker_controller = Node(
        package='attacker_controller',
        executable='attacker_node',
        name='attacker_controller',
        parameters=[{
            'mode': LaunchConfiguration('attacker_mode'),
            'max_speed': LaunchConfiguration('max_speed'),
            'speed_fraction': 0.8,
            'target_x': 7.0,
            'target_y': 4.0,
            'target_z': 2.0,
        }],
        output='screen',
    )

    return LaunchDescription([
        attacker_mode_arg,
        max_speed_arg,
        attacker_controller,
    ])
