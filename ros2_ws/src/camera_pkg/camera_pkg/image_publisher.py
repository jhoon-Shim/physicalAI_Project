import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # ROS2 표준 이미지 메시지 타입
import cv2                         # OpenCV 라이브러리
from cv_bridge import CvBridge     # OpenCV 이미지와 ROS2 메시지를 상호 변환해주는 도구
from rcl_interfaces.msg import SetParametersResult

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        # 1. 파라미터 선언 (이름, 기본값)
        # 속도는 15.0Hz, 크기는 요청하신 320x240으로 설정했습니다.
        self.declare_parameter('publish_rate', 15.0)
        self.declare_parameter('topic_name', 'image_raw')
        self.declare_parameter('image_size', [320, 240])

        # 2. 선언된 파라미터 값 가져와서 변수에 할당
        self.rate = self.get_parameter('publish_rate').value
        self.topic = self.get_parameter('topic_name').value
        self.size = self.get_parameter('image_size').value

        self.get_logger().info(
            f'🚀 노드 시작: {self.topic} ({self.size[0]}x{self.size[1]}) @ {self.rate}Hz'
        )

        # 3. 파라미터 변경 시 실행될 콜백 함수 등록
        self.add_on_set_parameters_callback(self.parameter_callback)

        # 4. 퍼블리셔 및 타이머 생성
        self.publisher_ = self.create_publisher(Image, self.topic, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

        # 5. 카메라 자원 및 CvBridge 초기화
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

    def parameter_callback(self, params):
        """
        런타임 중 파라미터 변경 요청이 들어오면 실행되는 콜백 함수
        """
        for param in params:
            # 발행 속도(publish_rate) 파라미터 변경 확인
            if param.name == 'publish_rate':
                self.rate = param.value
                self.get_logger().info(f'업데이트된 발행 속도: {self.rate}Hz')
                
                # 실제 타이머 주기를 변경하려면 아래 로직이 추가로 필요함. 이게 없으면 publish rate 값만 바뀌고 실제 hz는 변경 없음 
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

            elif param.name =='image_size':
                self.size = param.value
                self.get_logger().info(f"resolution: {self.size}")

        # 파라미터 적용 결과 성공 반환
        return SetParametersResult(successful=True)

    # def timer_callback(self):
    #     # 웹캠에서 프레임 읽기 (ret은 성공 여부, frame은 이미지 데이터)
    #     ret, frame = self.cap.read()
        
    #     if ret:
    #         # OpenCV 이미지(numpy array)를 ROS2 이미지 메시지로 변환
    #         # encoding="bgr8"은 색상 채널 순서를 의미해
    #         img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
    #         # 변환된 메시지를 토픽으로 발행
    #         self.publisher_.publish(img_msg)
    #         # self.get_logger().info('📸 이미지를 발행 중입니다...')
    #     else:
    #         self.get_logger().warn('⚠️ 카메라로부터 영상을 읽어올 수 없습니다.')


    # 속도가 너무 느려서 리사이즈 
    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            # 1. 해상도 리사이즈 (가장 확실한 해결책)
            frame = cv2.resize(frame, (self.size[0], self.size[1]))
            
            # 2. 이미지 메시지 변환
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_link"
            
            # 3. 발행
            self.publisher_.publish(img_msg)
            # self.get_logger().info('✅ 이미지 발행중.')
        else:
            self.get_logger().warn('⚠️ 카메라로부터 영상을 읽어올 수 없습니다.')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 노드를 종료합니다.')
    finally:
        # 종료 시 카메라 자원 해제 및 노드 파괴
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()