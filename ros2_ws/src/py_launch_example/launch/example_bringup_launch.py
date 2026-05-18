import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # 런치 인자 선언: rqt_image_view 실행 여부 결정
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to run rqt_image_view'
    )

    # 카메라 이미지 발행 노드
    image_publisher = Node(
        package='camera_pkg',
        executable='img_pub',
        name='image_publisher'
    )

    # YOLO 객체 검출 노드
    image_yolo = Node(
        package='camera_pkg',
        executable='img_yolo',
        name='image_yolo'
    )

    # Canny Edge 검출 노드
    img_canny = Node(
        package='camera_pkg',
        executable='img_canny',
        name='img_canny'
    )

    # rqt_image_view 노드 (use_rviz 인자가 true일 때만 실행)
    viewer_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_viewer',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    return LaunchDescription([
        use_rviz_arg,
        image_publisher,
        image_yolo,
        img_canny,
        viewer_node,
    ])