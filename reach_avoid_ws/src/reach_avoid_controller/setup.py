from setuptools import setup

package_name = 'reach_avoid_controller'

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
    description='Defender controller for reach-avoid games',
    license='MIT',
    entry_points={
        'console_scripts': [
            'defender_node = reach_avoid_controller.defender_node:main',
        ],
    },
)
