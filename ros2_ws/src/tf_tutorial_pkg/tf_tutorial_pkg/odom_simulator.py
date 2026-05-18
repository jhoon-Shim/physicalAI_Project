import rclpy
from rclpy.node import Node
import tf2_ros
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped
import math

class OdomSimulator(Node):
    def __init__(self):
        # 노드 이름을 'odom_simulator'로 초기화
        super().__init__('odom_simulator')
        
        # TF 브로드캐스터 생성: 좌표계 변환 정보를 시스템에 전송
        self.br = tf2_ros.TransformBroadcaster(self)
        
        # 0.05초마다 주기적으로 timer_callback을 호출하는 타이머 설정
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # 원 운동의 반지름(m) 및 각속도(rad/s) 설정
        self.radius = 1.0
        self.omega = 0.5
        
        # 기준 시점 기록
        self.start_time = self.get_clock().now()

    def timer_callback(self):
        # 현재 시간 계산 및 경과 시간(초) 변환
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        # 원 운동 궤적 계산: odom 좌표계 중심을 기준으로 x, y 좌표 산출
        x = self.radius * math.cos(self.omega * t)
        y = self.radius * math.sin(self.omega * t)

        # 로봇의 방향(Yaw) 설정: 진행 방향인 접선 방향을 바라보도록 설정
        roll = 0.0
        pitch = 0.0
        yaw = self.omega * t + math.pi / 2

        # 오일러 각을 쿼터니언으로 변환
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

        # TF 메시지 생성 및 데이터 채우기
        trans = TransformStamped()
        trans.header.stamp = now.to_msg()
        trans.header.frame_id = 'odom'
        trans.child_frame_id = 'base_link'

        # 위치 정보 입력
        trans.transform.translation.x = x
        trans.transform.translation.y = y
        trans.transform.translation.z = 0.0

        # 회전 정보(쿼터니언) 입력
        trans.transform.rotation.x = qx
        trans.transform.rotation.y = qy
        trans.transform.rotation.z = qz
        trans.transform.rotation.w = qw

        # 계산된 변환 정보를 브로드캐스트
        self.br.sendTransform(trans)

def main(args=None):
    # ROS 2 초기화 및 노드 실행
    rclpy.init(args=args)
    node = OdomSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 사용자의 종료 신호(Ctrl+C) 처리
        pass
    finally:
        # 리소스 정리 및 종료
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()