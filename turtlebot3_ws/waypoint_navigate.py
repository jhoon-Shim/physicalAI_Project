from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy
import time

rclpy.init()
nav = BasicNavigator()
nav.waitUntilNav2Active()

# RViz와의 마커 통신 호환성(QoS)을 맞추기 위해 TRANSIENT_LOCAL 설정 적용
qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
marker_pub = nav.create_publisher(MarkerArray, '/waypoints', qos)

# 경유할 중간 목적지 목록 설계 및 마커 생성
goals = []
marker_array = MarkerArray()

for i, (x, y) in enumerate([(1.0, 0.0), (1.5, 1.0), (0.0, 0.0)]):
    # 주행용 목적지 좌표 설정
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0
    goals.append(pose)

    # 시각화용 커스텀 마커 설정 (빨간색 구형)
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.ns = 'waypoints'
    marker.id = i
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0.1 # 바닥에서 약간 띄움
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0 # 불투명도
    marker.color.r = 1.0 # Red
    marker_array.markers.append(marker)

# RViz에 마커 표시 (/waypoints 토픽 구독 필요)
marker_pub.publish(marker_array)

# 경유지 순차 주행 미션 실행
nav.followWaypoints(goals)
while not nav.isTaskComplete():
    time.sleep(0.1) # CPU 과부하 방지
print("모든 목적지 경로 탐색 주행을 완료했습니다!")

rclpy.shutdown()