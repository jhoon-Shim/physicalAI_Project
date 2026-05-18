from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # RViz2 실행 여부를 결정하는 런치 인자 선언
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', 
        default_value='false',
        description='RViz2 시각화 실행 여부 (true/false)'
    )

    return LaunchDescription([
        use_rviz_arg,

        # 1. 이미지 발행 노드 (카메라 소스)
        Node(
            package='camera_pkg', 
            executable='img_pub',
            name='img_pub'
        ),

        # 2. YOLO 객체 탐지 노드
        Node(
            package='camera_pkg', 
            executable='yolo_pub',
            name='yolo_pub'
        ),

        # 3. Static TF 발행: map → odom (고정된 원점 기준)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_odom',
            arguments=['2.0', '0.0', '0.0', '0', '0', '0', 'map', 'odom']
        ),

        # 4. Dynamic TF 발행: odom → base_link (로봇의 주행 시뮬레이션)
        Node(
            package='tf_tutorial_pkg', 
            executable='odom_sim',
            name='odom_sim'
        ),

        # 5. Static TF 발행: base_link → camera_link (로봇 본체 대비 카메라 위치)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_base_to_camera',
            arguments=['0.1', '0.0', '0.2', '0', '0', '0', 'base_link', 'camera_link']
        ),

        # 6. Dynamic TF 발행: camera_link → object (YOLO 탐지 결과 반영)
        Node(
            package='tf_tutorial_pkg', 
            executable='tf_yolo',
            name='tf_yolo'
        ),

        # 7. TF Listener: 전체 TF 관계를 모니터링하고 로그 출력
        Node(
            package='tf_tutorial_pkg', 
            executable='tf_listener',
            name='tf_listener'
        ),

        # 8. RViz2 시각화 도구 (use_rviz 인자가 true일 때만 실행)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            # 사용자의 환경에 맞게 .rviz 설정 파일 경로 확인 필요
            arguments=['-d', '/home/jshim/2026 0513 1538 tf_tutorial.rviz'],
            condition=IfCondition(LaunchConfiguration('use_rviz'))
        ),
    ])