import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO  # YOLOv8 라이브러리

class YoloVisionProcessor(Node):
    def __init__(self):
        super().__init__('yolo_vision_processor')
        
        # 1. 모델 로드 (yolov8n.pt는 없으면 자동으로 다운로드됨)
        # 만약 본인의 .pt 파일이 있다면 경로를 적어주면 돼: YOLO('/home/user/best.pt')
        self.model = YOLO('yolov8n.pt') 
        
        # 2. 퍼블리셔 설정
        self.raw_publisher = self.create_publisher(Image, 'image_raw', 10)
        self.edge_publisher = self.create_publisher(Image, 'image_edge', 10)
        
        # 3. 자원 초기화
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.05, self.timer_callback) # 약 20fps
        
        self.get_logger().info('🔥 YOLOv8 모델이 로드되었습니다. 검출을 시작합니다!')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # --- 미션 1: 실제 .pt 모델로 추론(Inference) ---
        # stream=True로 설정하면 메모리 효율이 좋아짐
        results = self.model(frame, stream=True, verbose=False)

        for r in results:
            # 검출된 결과가 그려진 프레임을 가져옴
            annotated_frame = r.plot() 

        # YOLO 결과가 포함된 영상 발행
        raw_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.raw_publisher.publish(raw_msg)

        # --- 미션 2: Canny 엣지 처리 ---
        # 엣지 검출은 원본(frame) 혹은 결과(annotated_frame) 중 원하는 걸로 수행
        edge_frame = cv2.Canny(frame, 100, 200)
        edge_msg = self.bridge.cv2_to_imgmsg(edge_frame, encoding="mono8")
        self.edge_publisher.publish(edge_msg)

        # self.get_logger().info('🚀 YOLO & Edge 토픽 동시 송출 중...')

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()