from setuptools import setup

package_name = 'reach_avoid_hw'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='user@example.com',
    description='Hardware interface nodes for reach-avoid games',
    license='MIT',
    entry_points={
        'console_scripts': [
            'crazyswarm_adapter = reach_avoid_hw.crazyswarm_adapter_node:main',
            'safety_monitor = reach_avoid_hw.safety_monitor_node:main',
        ],
    },
)
