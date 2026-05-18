import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image     # 이미지 메시지 타입
from std_srvs.srv import Trigger      # 간단한 실행 요청 서비스 타입 (인자 없음)
import cv2                            # OpenCV
from cv_bridge import CvBridge        # ROS2 <-> OpenCV 변환 도구

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.current_frame = None

        # 1. 서브스크라이버 생성: 'image_raw' 토픽을 구독하여 영상을 받아옴
        self.subscription = self.create_subscription(
            Image, 
            'image_raw', 
            self.image_callback, 
            10
        )

        # 2. 서비스 서버 생성: 'capture_snapshot' 요청이 오면 사진 저장 실행
        self.srv = self.create_service(
            Trigger, 
            'capture_snapshot', 
            self.capture_callback
        )
        
        self.get_logger().info('🖼️ 이미지 프로세서 노드가 준비되었습니다.')

    def image_callback(self, msg):
        """
        영상을 수신할 때마다 실행되는 콜백 함수
        """
        # ROS2 이미지 메시지를 OpenCV 이미지(Numpy)로 변환
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # 실시간 화면 출력
        cv2.imshow("Camera View", self.current_frame)
        cv2.waitKey(1)  # OpenCV 윈도우 갱신을 위해 필수 (안 하면 화면 멈춤)

    def capture_callback(self, request, response):
        """
        'capture_snapshot' 서비스 요청이 왔을 때 실행되는 함수
        """
        if self.current_frame is not None:
            # 현재 프레임을 파일로 저장
            cv2.imwrite('snapshot.jpg', self.current_frame)
            response.success = True
            response.message = "📸 스냅샷이 'snapshot.jpg'로 저장되었습니다!"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "⚠️ 이미지가 아직 수신되지 않아 저장할 수 없습니다."
            self.get_logger().warn(response.message)
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()  # 프로그램 종료 시 윈도우 창 닫기
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()