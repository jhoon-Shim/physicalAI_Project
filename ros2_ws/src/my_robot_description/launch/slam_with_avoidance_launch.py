import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value='room_world.world',
        description='Gazebo world file name'
    )

    pkg_dir = get_package_share_directory('my_robot_description')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'turtlebot.xacro')
    world_file = PathJoinSubstitution([pkg_dir, 'worlds', LaunchConfiguration('world')])
    
    rviz_file = os.path.join(pkg_dir, 'rviz', 'slam.rviz')

    robot_description = Command(['xacro ', xacro_file])

    rviz_args = ['-d', rviz_file] if os.path.exists(rviz_file) else []

# 1. SLAM Toolbox 실행 환경 객체 정의
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        ]),
        launch_arguments={'use_sim_time': 'True'}.items()
    )

    return LaunchDescription([
        declare_world_arg,
        
        # 1. 가제보 시뮬레이터 실행
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file,
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        
        # 2. 로봇 상태 퍼블리셔 (URDF 정보 발행)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }]
        ),
        
        # 3. 가제보 내부에 로봇 스폰
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description',
                       '-entity', 'turtlebot', '-z', '0.3'],
            output='screen'
        ),
        
        # 4. RViz2 시각화 툴 실행
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=rviz_args,
        ),
        
        # 5. [추가] my_robot_tools 패키지의 장애물 회피 자율주행 노드 실행
        Node(
            package='my_robot_tools',
            executable='obstacle_avoidance',
            name='obstacle_avoidance_py',
            output='screen',
            parameters=[{'use_sim_time': True}] # 가제보 시간과 동기화
        ),

        slam_toolbox_launch,
    ])