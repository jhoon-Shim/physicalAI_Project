import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17 keypoints 연결 구조 (skeleton)
SKELETON = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # shoulders → hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

KEYPOINT_COLOR = (0, 255, 0)    # 관절점: 초록
SKELETON_COLOR = (255, 128, 0)  # 뼈대선: 주황
CONF_THRESHOLD = 0.5            # 신뢰도 낮은 keypoint 무시


def draw_pose(frame, keypoints):
    """keypoints: (17, 3) — x, y, confidence"""
    # 뼈대 선 그리기
    for a, b in SKELETON:
        x1, y1, c1 = keypoints[a]
        x2, y2, c2 = keypoints[b]
        if c1 > CONF_THRESHOLD and c2 > CONF_THRESHOLD:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), SKELETON_COLOR, 2)

    # 관절점 그리기
    for x, y, conf in keypoints:
        if conf > CONF_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 4, KEYPOINT_COLOR, -1)


def pose_pipeline():
    # yolov8n-pose.pt : nano pose 모델 (없으면 자동 다운로드)
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("can't open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=True)
        for r in results:
            annotated_frame = frame.copy()

            if r.keypoints is not None:
                kpts_all = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for kpts in kpts_all:
                    draw_pose(annotated_frame, kpts)

            # 바운딩박스 + 신뢰도 레이블
            annotated_frame = r.plot(img=annotated_frame, kpt_line=False)

            cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


pose_pipeline()