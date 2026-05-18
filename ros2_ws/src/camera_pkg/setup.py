from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jshim',
    maintainer_email='rjflekrwl@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'img_pub = camera_pkg.image_publisher:main',
            'img_proc = camera_pkg.image_processor:main',
            'img_pub_mission = camera_pkg.image_publisher_missions:main',
            'img_proc_turtle = camera_pkg.image_processor_turtle:main',
            'img_yolo = camera_pkg.image_publisher_yolo:main',
            'img_canny = camera_pkg.image_publisher_canny:main',
            'yolo_pub = camera_pkg.yolo_publisher:main',
        ],
    },
)
