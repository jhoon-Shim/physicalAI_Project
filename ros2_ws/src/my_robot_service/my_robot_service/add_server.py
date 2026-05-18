import rclpy
from rclpy.node import Node
from my_robot_interfaces.srv import AddTwoInts

class AddServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        # 서비스 생성: (서비스 타입, 서비스 이름, 콜백 함수)
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )
        self.get_logger().info('➕ 덧셈 서비스 서버가 준비되었습니다.')

    def add_two_ints_callback(self, request, response):
        # 요청받은 a와 b를 더함
        response.sum = request.a + request.b
        self.get_logger().info(f'📥 요청 수신: a={request.a}, b={request.b}')
        self.get_logger().info(f'📤 응답 전송: sum={response.sum}')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = AddServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()