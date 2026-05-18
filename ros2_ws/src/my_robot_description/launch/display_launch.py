import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 패키지의 share 디렉토리 경로를 가져옵니다. 
    pkg_share = get_package_share_directory('my_robot_description')
    
    # 2. 사용할 xacro 파일과 rviz 설정 파일의 전체 경로를 생성합니다. 
    xacro_file = os.path.join(pkg_share, 'urdf', 'turtlebot.xacro')
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'turtlebot.rviz')
    
    # 3. 런타임(실행 시점)에 XACRO 파일을 순수 URDF 형식으로 동적 변환하는 명령어입니다. 
    robot_description_content = Command(['xacro ', xacro_file])
    
    # 4. 런치 파일을 통해 실행할 항목들을 리스트 형태로 반환합니다. 
    return LaunchDescription([
        # 시뮬레이션 시간 사용 여부를 결정하는 런치 인자(아규먼트)를 정의합니다. 
        DeclareLaunchArgument(
            'use_sim_time', 
            default_value='false',
            description='Use simulation clock if true' 
        ),
        
        # 로봇의 상태(TF 좌표계 트리)를 계산하고 발행하는 대장 노드를 켭니다. 
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description_content, # 위에서 변환한 URDF 내용 주입 
                'use_sim_time': LaunchConfiguration('use_sim_time') 
            }]
        ),
        
        # 로봇 관절 조작용 GUI 창을 띄우는 노드를 켭니다.
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui'
        ),
        
        # 3D 시각화 툴인 RViz2를 미리 준비된 설정 파일과 함께 켭니다. 
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),
    ])
