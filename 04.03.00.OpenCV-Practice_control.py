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
        blurred_avg_frame = cv2.blur(gray_frame, (5, 5))
        blur_median   = cv2.medianBlur(gray_frame, 5)
        
        sobelx     = cv2.Sobel(blurred_avg_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely     = cv2.Sobel(blurred_avg_frame, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        
        # edges = cv2.Canny(blurred_avg_frame, 50, 150, L2gradient=True)  # NumPy 구현 결과와 맞추기 위해 L2gradient=True 적용
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # result = magnitude.copy()
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 900:          # 너무 작은 노이즈 제거
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # test = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        low = cv2.getTrackbarPos('Low Threshold', 'Controls')
        high = cv2.getTrackbarPos('High Threshold', 'Controls')
        edges = cv2.Canny(blurred_avg_frame, low, high)
        cv2.imshow('edge Viceo',edges)

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

