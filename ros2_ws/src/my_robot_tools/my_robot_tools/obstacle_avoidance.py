#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ObstacleAvoidance(Node):
    def __init__(self):
        # 노드 이름을 'obstacle_avoidance_py'로 초기화합니다.
        super().__init__('obstacle_avoidance_py')
        
        # 통신 안정성을 위해 최신 데이터 10개만 저장하는 QoS를 설정합니다.
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        
        # 1. 로봇의 바퀴를 제어할 속도 명령 퍼블리셔를 생성합니다. (토픽 이름: cmd_vel)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        
        # 2. 라이다 센서 데이터를 받아올 서브스크라이버를 생성합니다. (토픽 이름: scan)
        # 데이터가 들어올 때마다 클래스 내부의 scan_callback 함수가 실행됩니다.
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile)

    def scan_callback(self, msg):
        # 로봇에게 보낼 전진 및 회전 속도를 담을 Twist 메시지 객체를 생성합니다.
        twist_msg = Twist()
        
        # 기본 주행 속도 세팅: 전방에 아무것도 없으면 초속 15cm(0.15) (--> 0.5 로 변경)로 직진합니다.
        linear_vel = 0.5
        angular_vel = 0.0
        
        # 라이다가 보내온 전체 거리 데이터 배열의 개수를 파악합니다. (보통 360개)
        num_samples = len(msg.ranges)
        if num_samples == 0:
            return

        # 배열의 정중앙 인덱스가 로봇의 실제 정면(0도)을 의미합니다.
        center_index = num_samples // 2
        
        # 정면 기준 좌측 45도, 우측 45도를 감시할 인덱스 범위(360칸 중 45칸)를 계산합니다.
        scope = num_samples // 8

        # 감지된 최단 거리를 비교하기 위한 초기값으로 센서 최대 거리인 3.5m를 잡습니다.
        min_left_dist = 3.5
        min_right_dist = 3.5
        
        # 위험을 감지할 안전 반경 기준선입니다. (60cm)
        safety_threshold = 0.6

        # [정면 기준 왼쪽 45도 감시] 중앙에서 왼쪽 범위 안의 데이터를 조사합니다.
        for i in range(center_index, center_index + scope):
            # 센서 오류값(NaN 또는 무한대 수치)을 필터링하여 정상적인 거리값만 추출합니다.
            if not math.isnan(msg.ranges[i]) and not math.isinf(msg.ranges[i]):
                min_left_dist = min(min_left_dist, msg.ranges[i])

        # [정면 기준 오른쪽 45도 감시] 중앙에서 오른쪽 범위 안의 데이터를 조사합니다.
        for i in range(center_index - scope, center_index):
            if not math.isnan(msg.ranges[i]) and not math.isinf(msg.ranges[i]):
                min_right_dist = min(min_right_dist, msg.ranges[i])

        # [부드러운 곡선 회피 연산] 좌우 감시 구역 중 한 곳이라도 60cm 이내로 벽이 다가오면 실행합니다.
        if min_left_dist < safety_threshold or min_right_dist < safety_threshold:
            
            # 발견된 장애물 중 가장 짝 달라붙어 있는 최단 거리를 구합니다.
            closest_dist = min(min_left_dist, min_right_dist)
            
            # [부드러운 감속] 장애물에 가까워질수록 전진 속도를 부드럽게 감소시킵니다.
            # 완전히 멈추지 않고 선회 반경을 그릴 수 있게 최소 5cm ~ 최대 15cm 속도 사이로 비례 가변 제어합니다.
            linear_vel = 0.05 + 0.1 * (closest_dist / safety_threshold)

            # [부드러운 조향 회전] 장애물이 있는 방향의 반대쪽으로 핸들을 꺾습니다.
            # 벽과 거리가 바짝 가까워질수록 회전 각속도를 더 크고 빠르게 높여서 비껴가도록 계산합니다.
            if min_left_dist < min_right_dist:
                # 왼쪽에 장애물이 더 가까우므로 우측 방향(음수 값)으로 선회합니다.
                angular_vel = -0.7 * (1.0 - (min_left_dist / safety_threshold))
            else:
                # 오른쪽에 장애물이 더 가까우므로 좌측 방향(양수 값)으로 선회합니다.
                angular_vel = 0.7 * (1.0 - (min_right_dist / safety_threshold))

        # 계산이 끝난 최종 선속도(X축)와 각속도(Z축)를 메시지에 주입합니다.
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        
        # cmd_vel 토픽을 통해 가제보 가상 시뮬레이터 속 로봇에게 명령을 내보냅니다.
        self.cmd_vel_pub.publish(twist_msg)

def main(args=None):
    # ROS2 파이썬 통신 시스템을 초기화합니다.
    rclpy.init(args=args)
    
    # 노드를 생성하고 활성화하여 센서 콜백 함수가 상시 대기 상태를 유지하도록 만듭니다.
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 프로그램 종료 시 노드를 깔끔하게 파괴하고 시스템을 다운시킵니다.
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()