import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'py_launch_example'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 런치 파일 설치를 위한 경로 설정
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
        # 설정(config) 파일 설치를 위한 경로 설정
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml')),
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
        ],
    },
)
