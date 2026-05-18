import rclpy
from rclpy.node import Node
from my_robot_interfaces.msg import ObjectDetection, ObjectDetectionArray
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge

class YoloPublisher(Node):
    def __init__(self):
        # 노드 이름을 'yolo_publisher'로 초기화
        super().__init__('yolo_publisher')
        
        # ROS 이미지 메시지를 OpenCV 형식으로 변환하기 위한 Bridge 생성
        self.bridge = CvBridge()
        
        # YOLOv8 모델 로드 (가장 가벼운 Nano 버전인 yolov8n.pt 사용)
        self.model = YOLO('yolov8n.pt')
        
        # 'image_raw' 토픽을 통해 원본 영상을 구독(Subscription) 설정
        self.create_subscription(Image, 'image_raw', self.callback, 10)
        
        # 'image_yolo' 토픽으로 검출된 객체 정보를 발행(Publisher) 설정
        self.pub = self.create_publisher(ObjectDetectionArray, 'image_yolo', 10)

    def callback(self, msg):
        # ROS Image 메시지를 OpenCV(BGR8) 이미지 형식으로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # YOLO 모델로 객체 검출 실행 (불필요한 로그 출력은 제외)
        results = self.model(frame, verbose=False)[0]
        
        # 발행할 커스텀 메시지 객체 생성 및 헤더 동기화
        array_msg = ObjectDetectionArray()
        array_msg.header = msg.header
        
        # 검출된 각 객체 박스 정보를 반복하여 메시지에 저장
        for box in results.boxes:
            detection = ObjectDetection()
            # 클래스 인덱스를 실제 클래스 이름으로 변환하여 저장
            detection.class_name = results.names[int(box.cls)]
            # 검출 신뢰도(Confidence Score) 저장
            detection.confidence = float(box.conf)
            # 바운딩 박스 좌표(Center X, Center Y, Width, Height) 저장
            detection.bbox = [int(v) for v in box.xywh[0].tolist()]
            # 검출된 객체 정보를 배열 메시지에 추가
            array_msg.detections.append(detection)
            
        # 최종 검출 데이터 배열을 'image_yolo' 토픽으로 발행
        self.pub.publish(array_msg)

def main(args=None):
    # rclpy 통신 라이브러리 초기화
    rclpy.init(args=args)
    # YoloPublisher 노드 인스턴스 생성
    node = YoloPublisher()
    # 노드가 종료될 때까지 반복 실행(이벤트 루프)
    rclpy.spin(node)
    # 노드 자원 해제 및 종료
    node.destroy_node()
    rclpy.shutdown()