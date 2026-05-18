import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_interfaces.action import MoveRobot

class RobotMoveClient(Node):
    def __init__(self):
        super().__init__('robot_move_client')
        # 액션 클라이언트 생성: (노드, 액션 타입, 액션 이름)
        self._action_client = ActionClient(self, MoveRobot, 'move_robot')

    def send_goal(self, distance):
        """
        서버에 목표 거리를 전송하는 함수
        """
        goal_msg = MoveRobot.Goal()
        goal_msg.target_distance = distance

        # 서버가 켜질 때까지 대기
        self.get_logger().info('⏳ 액션 서버 대기 중...')
        self._action_client.wait_for_server()

        # 목표 비동기 전송 및 피드백 콜백 연결
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        # 목표 수락 여부 확인을 위한 콜백 추가
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        서버가 목표를 수락했는지 거절했는지 확인하는 콜백
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ 서버가 목표를 거절했습니다.')
            return

        self.get_logger().info('✅ 목표가 수락되었습니다. 이동을 시작합니다.')

        # 수락되었다면 이제 최종 결과를 기다리는 비동기 호출 실행
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """
        서버에서 실시간으로 보내주는 피드백을 처리
        """
        distance = feedback_msg.feedback.current_distance
        self.get_logger().info(f'📊 실시간 피드백: 현재 {distance}m 이동 중...')

    def get_result_callback(self, future):
        """
        모든 이동이 끝난 후 최종 결과를 처리
        """
        result = future.result().result
        self.get_logger().info(f'🏁 최종 결과: 도달 완료 = {result.reached}')
        
        # 작업 완료 후 노드 종료
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RobotMoveClient()
    
    # 5.0m 이동 목표 설정
    node.send_goal(5.0)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()