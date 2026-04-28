import cv2       # OpenCV 라이브러리 임포트
import numpy as np  # 행렬 연산을 위한 NumPy

def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 결과 표시
        # cv2.imshow('Original Video', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # BGR → HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 파란색 범위 정의
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        # 마스크 생성 및 적용

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        masked = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

        # 마스킹 된 영역에만 에지 검출 적용
        low = cv2.getTrackbarPos('Low Threshold', 'Controls')
        high = cv2.getTrackbarPos('High Threshold', 'Controls')
        edges = cv2.Canny(masked, low, high) 
        cv2.imshow('Only Blue Viceo',edges)

        # 'q' 키를 누르면 종료 (25ms 대기)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
cv2.namedWindow('Controls')
cv2.createTrackbar('Low Threshold', 'Controls', 100, 255, lambda x: None)
cv2.createTrackbar('High Threshold', 'Controls', 200, 255, lambda x: None)
edge_detection_pipeline()

