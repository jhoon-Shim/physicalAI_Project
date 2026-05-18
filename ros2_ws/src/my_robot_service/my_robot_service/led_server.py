import rclpy
from rclpy.node import Node
from my_robot_interfaces.srv import LedControl

class LedServiceServer(Node):
    def __init__(self):
        super().__init__('led_service_server')
        # 서비스 생성: (서비스 타입, 서비스 이름, 콜백 함수)
        self.srv = self.create_service(
            LedControl, 
            'set_led', 
            self.set_led_callback
        )
        self.get_logger().info('🚀 LED 서비스 서버가 시작되었습니다.')

    def set_led_callback(self, request, response):
        """
        서비스 요청을 처리하는 콜백 함수
        request.state: True(켜기) 또는 False(끄기)
        """
        if request.state:
            self.get_logger().info('💡 LED 켜기 요청 수신')
            response.success = True
            response.message = "LED를 성공적으로 켰습니다."
        else:
            self.get_logger().info('🌑 LED 끄기 요청 수신')
            response.success = True
            response.message = "LED를 성공적으로 껐습니다."
        
        # 처리 결과를 담은 response 객체 반환
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LedServiceServer()
    
    try:
        # 노드가 종료될 때까지 대기
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('서버가 종료됩니다.')
    finally:
        # 종료 전 자원 해제
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()