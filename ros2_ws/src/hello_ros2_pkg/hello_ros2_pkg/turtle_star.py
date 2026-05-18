import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class TurtleStar(Node):
    def __init__(self):
        super().__init__('turtle_star')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # 1.5초마다 동작을 전환 (별을 더 크게 그리려면 시간을 늘리세요)
        self.timer_period = 1.5
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.is_moving = True  # 전진 중인지 회전 중인지 상태 체크
        
        # 별을 그리기 위한 각도: 144도 (외각 기준)
        # 라디안 변환: 144 * (pi / 180) = 약 2.513 rad
        self.star_angle = (144 * math.pi) / 180

    def timer_callback(self):
        msg = Twist()

        if self.is_moving:
            # 시원하게 쭉 전진 (속도 3.0)
            msg.linear.x = 3.0
            msg.angular.z = 0.0
            self.get_logger().info('쭉쭉 전진!')
        else:
            # 간지나게 144도 회전
            # 회전 속도 = (목표 각도 / 시간)
            msg.linear.x = 0.0
            msg.angular.z = self.star_angle / self.timer_period
            self.get_logger().info('멋지게 회전!')

        self.publisher_.publish(msg)
        
        # 다음 단계로 상태 전환 (무한 반복)
        self.is_moving = not self.is_moving

def main(args=None):
    rclpy.init(args=args)
    node = TurtleStar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 종료 시 거북이를 멈추게 함
        stop_msg = Twist()
        node.publisher_.publish(stop_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()