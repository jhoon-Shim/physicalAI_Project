import rclpy
from rclpy.node import Node
from my_robot_interfaces.srv import LedControl

class LedServiceClient(Node):
    def __init__(self):
        super().__init__('led_service_client')
        # 클라이언트 생성: (서비스 타입, 서비스 이름)
        self.cli = self.create_client(LedControl, 'set_led')
        
        # 서버가 활성화될 때까지 1초마다 확인하며 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ 서버가 아직 준비되지 않았습니다. 대기 중...')
        
        self.req = LedControl.Request()

    def send_request(self, state):
        """
        서버에 LED 상태 변경 요청을 보냄
        """
        self.req.state = state
        # 비동기 요청 호출 (결과를 기다리지 않고 Future 객체 즉시 반환)
        self.future = self.cli.call_async(self.req)
        
        # Future가 완료될 때까지(응답이 올 때까지) 노드를 실행하며 대기
        rclpy.spin_until_future_complete(self, self.future)
        
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = LedServiceClient()
    
    # LED 켜기(True) 요청 전송
    response = client.send_request(True)
    
    # 결과 출력
    client.get_logger().info(
        f'✅ 결과: {response.success}, 메시지: {response.message}'
    )

    # 작업 완료 후 자원 해제
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()