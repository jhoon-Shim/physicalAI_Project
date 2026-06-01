import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters


class CompareViewer(Node):
    def __init__(self):
        super().__init__('compare_undistort_viewer')
        self.bridge = CvBridge()
        self.raw = None
        self.rect = None

        # message_filters를 사용하여 두 이미지의 Header(타임스탬프) 동기화
        self.raw_sub = message_filters.Subscriber(self, Image, '/image_raw')
        self.rect_sub = message_filters.Subscriber(self, Image, '/image_rect')
        
        # TimeSynchronizer로 타임스탬프가 완전히 일치하는 메시지끼리 묶음
        self.ts = message_filters.TimeSynchronizer(
            [self.raw_sub, self.rect_sub], queue_size=10)
        self.ts.registerCallback(self.sync_cb)

        cv2.namedWindow('Calibration Compare', cv2.WINDOW_NORMAL)
        self.get_logger().info('/image_raw 및 /image_rect 토픽 대기 중...')

    def sync_cb(self, raw_msg, rect_msg):
        self.raw = self.bridge.imgmsg_to_cv2(raw_msg, 'bgr8')
        self.rect = self.bridge.imgmsg_to_cv2(rect_msg, 'bgr8')
        self.show()

    def show(self):
        if self.raw is None or self.rect is None:
            return

        # 원본과 보정본의 해상도가 다르면 비교할 수 없으므로 경고 출력
        if self.raw.shape != self.rect.shape:
            self.get_logger().warning(f"이미지 해상도가 다릅니다! raw: {self.raw.shape}, rect: {self.rect.shape}")
            return

        h, w = self.raw.shape[:2]

        # 상단: 좌우 배치 (원본 | 보정됨)
        top = np.hstack([self.raw, self.rect])

        # 하단: 차이점 히트맵
        diff = cv2.absdiff(self.raw, self.rect)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # [추가됨] 픽셀 밝기 차이가 작아 파란색 계열만 나오는 것을 방지하기 위해 차이값을 5배 증폭
        diff_gray = cv2.convertScaleAbs(diff_gray, alpha=5.0)
        
        diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        diff_color = cv2.resize(diff_color, (w * 2, h))

        # 수직으로 병합
        display = np.vstack([top, diff_color])

        # 라벨 추가 (cv2 폰트는 한글이 깨질 수 있어 영어로 유지)
        cv2.putText(display, 'ORIGINAL', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, 'RECTIFIED', (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, 'DIFFERENCE (red = high distortion)',
                    (10, h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Calibration Compare', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("'q' 입력됨. 종료합니다...")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CompareViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()