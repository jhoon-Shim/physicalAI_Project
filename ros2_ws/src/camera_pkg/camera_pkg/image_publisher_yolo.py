import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloSubscriber(Node):
    def __init__(self):
        super().__init__('image_yolo')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        # 구독자: img_pub이 발행하는 'image_raw' 토픽을 받아옴
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        
        # 결과 발행을 위한 퍼블리셔
        self.publisher = self.create_publisher(Image, 'image_yolo_result', 10)
        self.get_logger().info('✅ YOLO 구독 노드가 시작되었습니다.')

    def image_callback(self, msg):
        # 받아온 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # YOLO 추론
        results = self.model(frame, stream=True, verbose=False)
        annotated_frame = frame.copy()
        for r in results:
            annotated_frame = r.plot()

        # 결과 발행
        result_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.publisher.publish(result_msg)
        # self.get_logger().info('✅ 욜로 발행중.')

def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()