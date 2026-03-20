import os
from glob import glob
from setuptools import setup

package_name = 'reach_avoid_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='user@example.com',
    description='Simulation adapters and ground truth relay for reach-avoid games',
    license='MIT',
    entry_points={
        'console_scripts': [
            'px4_adapter = reach_avoid_sim.px4_adapter_node:main',
            'ground_truth_relay = reach_avoid_sim.ground_truth_relay_node:main',
            'kinematic_sim = reach_avoid_sim.kinematic_sim_node:main',
        ],
    },
)
