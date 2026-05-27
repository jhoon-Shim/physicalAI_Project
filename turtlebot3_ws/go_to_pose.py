from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from geometry_msgs.msg import PoseStamped
import time

rclpy.init()
nav = BasicNavigator()
nav.waitUntilNav2Active() # Nav2 노드 상태 활성화 대기

# 목표 목적지 좌표계 및 위치 지정
goal_pose = PoseStamped()
goal_pose.header.frame_id = 'map'
goal_pose.pose.position.x = 1.5
goal_pose.pose.position.y = 0.5
goal_pose.pose.orientation.w = 1.0

nav.goToPose(goal_pose)
while not nav.isTaskComplete():
    feedback = nav.getFeedback() # 실시간 잔여 도달 거리 및 피드백 획득
    time.sleep(0.1) # CPU 과부하 방지
print("목적지에 안전하게 도달했습니다!")
rclpy.shutdown()