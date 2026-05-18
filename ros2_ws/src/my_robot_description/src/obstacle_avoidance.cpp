#include <chrono>
#include <memory>
#include <algorithm>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/twist.hpp"

// rclcpp::Node를 상속받아 자율주행 및 충돌 회피를 담당하는 클래스를 정의합니다.
class ObstacleAvoidance : public rclcpp::Node
{
public:
  ObstacleAvoidance() : Node("obstacle_avoidance")
  {
    // 통신 안정성을 위한 QoS(Quality of Service) 설정으로, 최신 데이터 10개만 보관하도록 세팅합니다.
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    
    // 가제보 속 로봇에게 이동 속도 명령을 전송할 퍼블리셔(Publisher)를 생성합니다.
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", qos);
    
    // 가제보 라이다 센서가 보내오는 거리 데이터를 받아올 서브스크라이버(Subscriber)를 생성합니다.
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", qos, std::bind(&ObstacleAvoidance::scan_callback, this, std::placeholders::_1));
  }

private:
  // 라이다 센서 데이터가 수신될 때마다 실행되는 핵심 제어 함수입니다.
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    // 로봇에게 최종적으로 전송할 속도 데이터(선속도, 각속도)를 담을 변수입니다.
    auto twist_msg = geometry_msgs::msg::Twist();
    
    // 기본 주행 속도를 정의합니다. 장애물이 없을 때는 초속 15cm(0.15) 속도로 곧장 직진합니다.
    double linear_vel = 0.5;
    double angular_vel = 0.0;
    
    // 라이다 센서가 측정해온 거리 배열의 총 개수(360개)를 파악합니다.
    int num_samples = msg->ranges.size();
    if (num_samples == 0) return;

    // [중요] 라이다 데이터 배열의 정중앙(예: 360개 중 180번)이 로봇의 진짜 정면(0도)입니다.
    int center_index = num_samples / 2; 
    
    // 정면을 기준으로 좌측 45도, 우측 45도를 감시하기 위한 인덱스 범위 범위(360개 중 45개)를 계산합니다.
    int scope = num_samples / 8;        

    // 감지된 최소 거리를 비교하기 위한 변수입니다. 초기값은 센서 측정 한계치인 3.5m로 설정합니다.
    double min_left_dist = 3.5;
    double min_right_dist = 3.5;
    
    // 위험을 감지할 안전 반경 기준선입니다. 장애물이 60cm(0.6m) 안으로 들어오면 회피 동작을 시작합니다.
    double safety_threshold = 0.6; 

    // [정면 기준 왼쪽 45도 감시] 중앙 인덱스부터 왼쪽으로 45칸까지 조사합니다.
    for (int i = center_index; i < center_index + scope; ++i) {
      // 센서 측정값 중 에러값(NaN 또는 감지 실패로 인한 무한대 값)이 아닌 정상적인 수치만 필터링합니다.
      if (!std::isnan(msg->ranges[i]) && !std::isinf(msg->ranges[i])) {
        min_left_dist = std::min(min_left_dist, static_cast<double>(msg->ranges[i]));
      }
    }

    // [정면 기준 오른쪽 45도 감시] 중앙 인덱스부터 오른쪽으로 45칸 뒤까지 조사합니다.
    for (int i = center_index - scope; i < center_index; ++i) {
      if (!std::isnan(msg->ranges[i]) && !std::isinf(msg->ranges[i])) {
        min_right_dist = std::min(min_right_dist, static_cast<double>(msg->ranges[i]));
      }
    }

    // [곡선 회피 연산] 왼쪽이나 오른쪽 감시 영역 중 하나라도 안전 기준선(60cm)보다 가까운 장애물이 감지되면 실행합니다.
    if (min_left_dist < safety_threshold || min_right_dist < safety_threshold) {
      
      // 양쪽 영역을 통틀어 로봇과 가장 가까이 붙어있는 장애물과의 실제 거리를 찾습니다.
      double closest_dist = std::min(min_left_dist, min_right_dist);
      
      // [부드러운 감속] 벽에 가까워질수록 전진 속도를 부드럽게 줄여나갑니다. 
      // 완전히 멈추지 않고 선회력을 유지할 수 있도록 최소 초속 5cm에서 최대 15cm 사이로 속도를 가변 제어합니다.
      linear_vel = 0.05 + 0.1 * (closest_dist / safety_threshold);

      // [부드러운 조향 회전] 장애물이 존재하는 방향의 반대쪽으로 바퀴를 틀어 턴을 준비합니다.
      // 벽과 거리가 바짝 가까워질수록 회전 각속도를 더 크고 격렬하게 높여서 확실하게 탈출하도록 설계합니다.
      if (min_left_dist < min_right_dist) {
        // 왼쪽에 장애물이 더 가까우므로 우측 방향(음수 값)으로 핸들을 꺾습니다.
        angular_vel = -0.7 * (1.0 - (min_left_dist / safety_threshold));
      } else {
        // 오른쪽에 장애물이 더 가까우므로 좌측 방향(양수 값)으로 핸들을 꺾습니다.
        angular_vel = 0.7 * (1.0 - (min_right_dist / safety_threshold));
      }
    }

    // 연산이 완료된 전진 속도(X축)와 회전 속도(Z축 각속도) 명령을 공용 토픽 메시지에 최종 입력합니다.
    twist_msg.linear.x = linear_vel;
    twist_msg.angular.z = angular_vel;
    
    // cmd_vel 통로를 통해 가제보 가상 세계에 있는 로봇에게 움직임 명령을 최종 전송합니다.
    cmd_vel_pub_->publish(twist_msg);
  }

  // ROS2 통신 처리를 위한 멤버 변수 포인터를 선언합니다.
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
};

int main(int argc, char * argv[])
{
  // ROS2 통신 시스템의 초기화를 수행합니다.
  rclcpp::init(argc, argv);
  
  // 우리가 만든 ObstacleAvoidance 노드를 실행시키고, 센서 데이터가 들어올 때까지 종료되지 않고 무한 루프를 돌며 대기하도록 지시합니다.
  rclcpp::spin(std::make_shared<ObstacleAvoidance>());
  
  // 프로그램이 강제 종료되거나 정지될 때 시스템을 안전하게 다운시킵니다.
  rclcpp::shutdown();
  return 0;
}