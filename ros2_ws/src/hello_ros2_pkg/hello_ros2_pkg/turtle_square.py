import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class TurtleSquare(Node):
    def __init__(self):
        super().__init__('turtle_square')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # 1초마다 상태를 전환하기 위한 타이머
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        self.is_moving = True  # True면 전진, False면 회전
        self.count = 0         # 총 8단계 (전진4 + 회전4)

    def timer_callback(self):
        msg = Twist()
        
        if self.count >= 8:
            self.get_logger().info('정사각형 완성!')
            self.timer.cancel() # 타이머 정지
            return

        if self.is_moving:
            # 1초간 전진 (속도 2.0)
            msg.linear.x = 2.0
            msg.angular.z = 0.0
            self.get_logger().info('전진 중...')
        else:
            # 1초간 90도 회전 (pi/2 rad/s)
            msg.linear.x = 0.0
            msg.angular.z = math.pi / 2
            self.get_logger().info('회전 중...')

        self.publisher_.publish(msg)
        
        # 상태 반전 및 카운트 증가
        self.is_moving = not self.is_moving
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = TurtleSquare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()