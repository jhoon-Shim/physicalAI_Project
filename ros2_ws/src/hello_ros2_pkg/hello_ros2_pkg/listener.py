import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        # 'listener'라는 이름으로 노드 초기화
        super().__init__('listener')
        
        # 'hello_topic'이라는 토픽을 구독함
        # 메시지 타입은 String이며, 메시지를 받을 때마다 listener_callback 함수를 실행
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.listener_callback,
            10)
        self.subscription  # 변수 사용 경고 방지용

    def listener_callback(self, msg):
        # 메시지를 받았을 때 터미널에 로그를 출력
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    # ROS 2 통신 초기화
    rclpy.init(args=args)
    
    # 노드 생성
    node = Listener()
    
    try:
        # 메시지를 기다리며 노드 실행 유지
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 리소스 정리
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()