"""
YOLOv8 + 칼만 필터 실시간 웹캠 추적 (터미널에서 직접 실행)

칼만 필터 효과 확인:
    - 빨간 점: YOLO 원본 중심점 (떨림 있음)
    - 파란 점: 칼만 필터 보정 중심점 (부드럽게 안정화)
    - 탐지된 클래스별로 트래커 1개씩 운용 (같은 클래스 여러 객체 중 첫 번째만 추적)

실행 방법:
    python 05.02.02.Webcam-Kalman.py

종료 방법:
    'q' 키 입력

NIS 데이터:
    종료 시 nis_log.csv 저장 → 05.02.03.Kalman-NIS-Eval.py 로 분석
"""

import csv
import cv2
import numpy as np
from ultralytics import YOLO


class BBoxKalmanTracker:
    """
    YOLO 바운딩 박스 중심점(cx, cy)을 칼만 필터로 추적하는 트래커
    상태벡터: [cx, vx, cy, vy] (x/y 위치 + 속도)
    """
    def __init__(self, dt=1.0):  # dt: 프레임 단위 시간 간격 (1.0 = 1프레임). 실제 시간 반영 시 dt = 1/fps 사용
        self.dt = dt

        # 상태 전이 행렬 (등속 운동 모델)
        self.A = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ], dtype=float)

        # 센서 행렬 (cx, cy만 측정)
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=float)

        # Q: 프로세스 노이즈 — "내 운동 모델(등속 운동)이 얼마나 부정확한가"
        #   빨간 점: YOLO가 매 프레임 뱉는 바운딩박스 중심 (떨림 있음, 모델 무관)
        #   파란 점: 칼만 필터가 Q/R을 바탕으로 계산한 보정 위치 (부드럽게 안정화)
        #   Q 크게 → 운동 모델 불신 → 센서(YOLO) 더 믿음 → 파란 점이 빨간 점처럼 빠르게 따라감 (떨림 증가), 값을 10으로 늘려봐
        #   Q 작게 → 운동 모델 신뢰 → 파란 점이 더 부드럽지만 방향 전환에 느리게 반응
        self.Q = np.eye(4) * 0.1

        # R: 센서 노이즈 — "YOLO 바운딩박스 중심점이 얼마나 부정확한가"
        #   R 크게 → YOLO 검출 불신 → 운동 모델 더 믿음 → 파란 점이 YOLO 떨림을 무시하고 부드럽게 이동
        #   R 작게 → YOLO 검출 완전 신뢰 → 파란 점이 빨간 점과 거의 동일해짐 (칼만 필터 효과 없음), 값을 5로 줄여봐 
        self.R = np.eye(2) * 50.0
        self.x = np.zeros((4, 1))        # 상태 벡터 [cx, vx, cy, vy]^T — 첫 탐지 전이므로 0 초기화, init()에서 실제 값 설정
        self.P = np.eye(4) * 500         # 초기 공분산 — "위치/속도를 전혀 모른다"는 큰 불확실성으로 시작, 몇 프레임 후 자동 수렴
        self.initialized = False

    def init(self, cx, cy):
        self.x = np.array([[cx], [0.0], [cy], [0.0]], dtype=float)
        self.initialized = True

    def predict(self):
        # TODO: Predict 단계를 구현하세요
        # 힌트1: self.x: 상태 예측
        # 힌트2: self.P: 공분산 예측
        # 힌트3: 예측된 cx, cy 반환
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[0,0], self.x[2,0]

    def update(self, cx, cy):
        # TODO: Update 단계를 구현하세요
        # 힌트1: z: 측정값 벡터
        # 힌트2: innovation: 측정 잔차
        # 힌트2: S: Innovation 공분산
        # 힌트3: K: 칼만 게인
        # 힌트4: self.x: 상태 보정
        # 힌트5: self.P: 공분산 보정
        # 힌트6: 보정된 cx, cy 반환
        z = np.array([[cx], [cy]], dtype = float)
        innovation = z - self.C @ self.x
        
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.C) @ self.P

        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return self.x[0, 0], self.x[2, 0], nis


TRACK_CLS = 0       # 추적 대상 클래스 (0 = person)
USE_KALMAN = True   # True: 칼만 필터 활성 / False: YOLO 원본만 표시

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. source 번호를 확인하세요.")
    exit()

WINDOW_NAME = "YOLO + Kalman Filter  [q: 종료]"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

tracker = BBoxKalmanTracker()
nis_log = []    # (frame, nis) 누적
frame_idx = 0

print("YOLOv8 + 칼만 필터 추적 시작 — 'q' 키로 종료")
print(f"추적 대상: person 1명  |  칼만 필터: {'ON' if USE_KALMAN else 'OFF'}")
print("빨간 점: YOLO 원본  |  파란 점: 칼만 보정")

while True:
    success, frame = cap.read()
    if not success:
        print("프레임을 읽을 수 없습니다.")
        break

    frame = cv2.resize(frame, (1280, 720))

    frame_idx += 1
    results = model.predict(frame, classes=[TRACK_CLS], conf=0.5, verbose=False)

    for r in results:
        annotated = r.plot()

        if len(r.boxes) == 0:
            cv2.imshow(WINDOW_NAME, annotated)
            continue

        # 신뢰도가 가장 높은 person 1명만 추적
        best_box = max(r.boxes, key=lambda b: float(b.conf[0]))
        cx, cy = best_box.xywh[0][:2].tolist()

        # YOLO 원본 중심점: 빨간색 (항상 표시)
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        if USE_KALMAN:
            if not tracker.initialized:
                tracker.init(cx, cy)

            tracker.predict()
            filtered_cx, filtered_cy, nis = tracker.update(cx, cy)
            nis_log.append((frame_idx, nis))

            # 칼만 보정 중심점: 파란색
            cv2.circle(annotated, (int(filtered_cx), int(filtered_cy)), 8, (255, 0, 0), -1)

            # 두 점 연결선으로 차이 강조
            cv2.line(annotated,
                     (int(cx), int(cy)),
                     (int(filtered_cx), int(filtered_cy)),
                     (0, 255, 255), 1)

            cv2.putText(annotated, f"person: K({filtered_cx:.0f},{filtered_cy:.0f})",
                        (int(filtered_cx) + 10, int(filtered_cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            cv2.putText(annotated, f"person: ({cx:.0f},{cy:.0f})",
                        (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(WINDOW_NAME, annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if nis_log:
    csv_path = "nis_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "nis"])
        writer.writerows(nis_log)
    print(f"NIS 로그 저장 완료: {csv_path} ({len(nis_log)}개 샘플, 평균 NIS={np.mean([n for _, n in nis_log]):.2f})")

print("종료")
