import os
from glob import glob
from setuptools import setup

package_name = 'reach_avoid_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='user@example.com',
    description='Bringup package for reach-avoid games',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
