import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class TurtleVisionNode(Node):
    def __init__(self):
        super().__init__('turtle_vision_node')
        
        # 1. YOLO 모델 로드 (노트북에서 수행)
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # 2. 서브스크라이버: 터틀봇의 원본 영상을 가져옴
        # 터틀봇의 토픽명이 /image_raw 인지 /camera/image_raw 인지 확인 후 수정 가능
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # 3. 퍼블리셔: 처리된 결과를 새로운 이름으로 내보냄
        # 미션 1: YOLO 결과 토픽
        self.yolo_pub = self.create_publisher(Image, '/turtle_yolo_result', 10)
        # 미션 2: 엣지 결과 토픽
        self.edge_pub = self.create_publisher(Image, '/turtle_edge_result', 10)

        self.get_logger().info('🐢 터틀봇 비전 분석 노드가 가동되었습니다!')

    def image_callback(self, msg):
        # ROS2 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # --- 미션 1: YOLO 추론 및 박스 그리기 ---
        results = self.model(cv_image, stream=True, verbose=False)
        annotated_frame = cv_image.copy()
        for r in results:
            annotated_frame = r.plot()

        # YOLO 결과 퍼블리시
        yolo_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.yolo_pub.publish(yolo_msg)

        # --- 미션 2: Canny 엣지 처리 ---
        edge_frame = cv2.Canny(cv_image, 100, 200)
        
        # 엣지 결과 퍼블리시 (흑백이므로 mono8)
        edge_msg = self.bridge.cv2_to_imgmsg(edge_frame, encoding='mono8')
        self.edge_pub.publish(edge_msg)

        self.get_logger().info('✨ 분석 데이터 송신 중...')

def main(args=None):
    rclpy.init(args=args)
    node = TurtleVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()