import rclpy
from rclpy.node import Node
from my_robot_interfaces.msg import ObjectDetectionArray
from geometry_msgs.msg import TransformStamped
import tf2_ros

class TfYoloBroadcaster(Node):
    def __init__(self):
        # 노드 이름을 'tf_yolo_broadcaster'로 초기화
        super().__init__('tf_yolo_broadcaster')
        
        # TF 브로드캐스터 생성: 탐지된 객체의 위치를 TF로 전송
        self.br = tf2_ros.TransformBroadcaster(self)
        
        # '/image_yolo' 토픽을 구독하여 객체 탐지 데이터를 수신
        self.create_subscription(
            ObjectDetectionArray, 
            '/image_yolo', 
            self.callback, 
            10
        )

    def callback(self, msg):
        # 현재 시간을 메시지 타임스탬프로 사용
        now = self.get_clock().now().to_msg()
        
        for i, det in enumerate(msg.detections):
            # 탐지된 객체의 클래스가 'person'이 아니면 건너뜀
            if det.class_name != 'person':
                continue
            
            # Bounding Box 중심점 계산 (단순화된 3D 투영 가정)
            cx, cy, w, h = det.bbox
            
            # 이미지 좌표(320x240 기준)를 정규화하여 공간 좌표로 변환
            # x: 이미지 가로 중심에서의 거리, y: 이미지 세로 중심에서의 거리
            x = (cx - 160) / 320.0
            y = (cy - 120) / 240.0
            z = 1.0  # 깊이(거리)를 1m로 고정 가정
            
            # TF 메시지 생성
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'camera_link' # 기준 좌표계
            t.child_frame_id = f'object_{det.class_name}_{0}' # 대상 좌표계 이름
            
            # 카메라 좌표계(Z-앞, X-오른쪽, Y-아래)를 TF 좌표계로 매핑
            t.transform.translation.x = z
            t.transform.translation.y = -x
            t.transform.translation.z = -y
            
            # 회전 정보는 정지 상태(쿼터니언 기본값)로 설정
            t.transform.rotation.w = 1.0
            
            # 계산된 객체 위치 TF 발행
            self.br.sendTransform(t)
            
            self.get_logger().info(
                f'TF 발행: {t.child_frame_id} at ({z:.2f}, {-x:.2f}, {-y:.2f})'
            )
            
            # 첫 번째 'person' 객체만 처리하고 종료
            break

def main(args=None):
    # ROS 2 초기화 및 노드 실행
    rclpy.init(args=args)
    node = TfYoloBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()