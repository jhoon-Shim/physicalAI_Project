import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CannySubscriber(Node):
    def __init__(self):
        super().__init__('img_canny')
        self.bridge = CvBridge()

        # 구독자: 'image_raw' 토픽 구독
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        
        # 결과 발행
        self.publisher = self.create_publisher(Image, 'image_canny_result', 10)
        # self.get_logger().info('✅ Canny 구독 노드가 시작되었습니다.')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Canny 처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_frame = cv2.Canny(gray, 100, 200)

        # 결과 발행
        result_msg = self.bridge.cv2_to_imgmsg(edge_frame, encoding="mono8")
        self.publisher.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CannySubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()