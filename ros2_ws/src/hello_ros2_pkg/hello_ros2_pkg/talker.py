import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        # 'talker'라는 이름으로 노드 초기화
        super().__init__('talker')
        
        # 'hello_topic'이라는 토픽으로 String 메시지를 발행하는 퍼블리셔 생성 (큐 사이즈 10)
        self.publisher_ = self.create_publisher(String, 'hello_topic', 10)
        
        # 0.5초마다 timer_callback 함수를 실행하는 타이머 설정
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello ROS2! [{self.i}]'
        
        # 메시지 발행 및 터미널 로그 출력
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    # ROS2 통신 초기화
    rclpy.init(args=args)
    
    # 노드 생성 및 실행 유지
    node = Talker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 리소스 정리
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()