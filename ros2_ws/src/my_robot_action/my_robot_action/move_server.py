import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from my_robot_interfaces.action import MoveRobot

class RobotMoveServer(Node):
    def __init__(self):
        super().__init__('robot_move_server')
        # 액션 서버 생성: (노드, 액션 타입, 액션 이름, 콜백 함수)
        self._action_server = ActionServer(
            self, 
            MoveRobot, 
            'move_robot', 
            self.execute_callback
        )
        self.get_logger().info('🚀 로봇 이동 액션 서버가 시작되었습니다.')

    def execute_callback(self, goal_handle):
        """
        목표를 받아 실제로 로봇을 이동시키는 콜백 함수
        """
        self.get_logger().info(f'📍 목표 거리 {goal_handle.request.target_distance}m 이동 시작...')

        feedback_msg = MoveRobot.Feedback()
        target = goal_handle.request.target_distance

        # 목표 거리만큼 1초마다 이동하며 피드백 전송
        for i in range(1, int(target) + 1):
            # 중간에 취소 요청이 들어왔는지 확인
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('🛑 이동 취소됨')
                return MoveRobot.Result(reached=False)

            time.sleep(1.0)  # 이동 중인 상황 시뮬레이션
            
            # 피드백 업데이트 및 발행
            feedback_msg.current_distance = float(i)
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'📊 진행 중: {i}/{target}m')

        # 모든 과정 완료 후 성공 상태 알림
        goal_handle.succeed()
        
        result = MoveRobot.Result()
        result.reached = True
        return result

def main(args=None):
    rclpy.init(args=args)
    node = RobotMoveServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()