import sys
import rclpy
from rclpy.node import Node
from my_robot_interfaces.srv import AddTwoInts

class AddClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        
        # 서버가 활성화될 때까지 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ 서버 연결 대기 중...')
        
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        # 비동기 호출
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    
    # 터미널 인자로 숫자를 받거나 기본값 사용
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    client = AddClient()
    response = client.send_request(a, b)
    
    if response is not None:
        client.get_logger().info(f'✅ 결과: {a} + {b} = {response.sum}')
    else:
        client.get_logger().error('❌ 서비스 호출 실패')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()