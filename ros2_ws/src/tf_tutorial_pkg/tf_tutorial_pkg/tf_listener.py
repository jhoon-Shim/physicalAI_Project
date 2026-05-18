import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped

class TfListener(Node):
    def __init__(self):
        # 노드 이름을 'tf_listener'로 초기화
        super().__init__('tf_listener')
        
        # TF 데이터를 저장할 버퍼 생성
        self.tf_buffer = tf2_ros.Buffer()
        
        # TF 데이터를 수신할 리스너 생성 (버퍼에 데이터를 채움)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 1.0초마다 timer_callback을 호출하여 좌표 변환 조회
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        try:
            # lookup_transform: 특정 시점의 두 좌표계 사이의 변환 정보를 조회
            # target_frame: 기준이 되는 좌표계 ('base_link' --> map)
            # source_frame: 변환하고 싶은 대상 좌표계 ('camera_link')
            # rclpy.time.Time(): 가장 최신의 가용한 TF 정보를 가져옴
            trans = self.tf_buffer.lookup_transform(
                'map',
                'camera_link',
                rclpy.time.Time()
            )

            # 변환 정보에서 위치(translation) 데이터 추출
            t = trans.transform.translation
            
            # 조회된 위치 정보를 소수점 3자리까지 출력
            self.get_logger().info(
                f'camera_link in base_link: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}'
            )

        except tf2_ros.LookupException as e:
            # 좌표계 이름이 틀렸거나 연결되지 않았을 때 (조회 실패)
            self.get_logger().warn(f'TF 조회 실패: {e}')
            
        except tf2_ros.ConnectivityException as e:
            # 두 좌표계 사이의 연결 경로(Tree)가 없을 때
            self.get_logger().warn(f'좌표계 연결 실패: {e}')
            
        except tf2_ros.ExtrapolationException as e:
            # 요청한 시간대의 데이터가 아직 없거나 너무 과거일 때 (시간 불일치)
            self.get_logger().warn(f'시간 외삽 오류: {e}')

def main(args=None):
    # ROS 2 초기화
    rclpy.init(args=args)
    
    # 노드 인스턴스 생성
    node = TfListener()
    
    try:
        # 노드 실행 (콜백 함수들이 동작하도록 함)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C 종료 처리
        pass
    finally:
        # 종료 전 정리 작업
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()