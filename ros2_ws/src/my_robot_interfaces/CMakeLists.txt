cmake_minimum_required(VERSION 3.8)
project(my_robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 필수 의존성 패키지 추가
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED) # 인터페이스 생성을 위해 반드시 필요
find_package(std_msgs REQUIRED) # 커스텀 메세지

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 2. 서비스 파일 정의 및 의존성 연결
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/LedControl.srv"
  # 만약 srv 내부에 다른 패키지의 메시지(ex: std_msgs)를 사용한다면 아래에 추가해야 함
  # DEPENDENCIES std_msgs 
  "action/MoveRobot.action"
  "srv/AddTwoInts.srv"

  "msg/ObjectDetection.msg"
  "msg/ObjectDetectionArray.msg"
  DEPENDENCIES std_msgs
)

# 3. 내보내기 설정 (다른 패키지에서 이 인터페이스를 사용할 수 있게 함)
ament_export_dependencies(rosidl_default_runtime)

ament_package()