import cv2
import numpy as np
from ultralytics import YOLO

def segmentation_pipeline():
    # yolov8n-seg.pt : nano segmentation 모델 (없으면 자동 다운로드)
    model = YOLO('yolov8n-seg.pt')
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

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()   # (N, H, W)
                classes = r.boxes.cls.cpu().numpy()  # (N,)

                for mask, cls_id in zip(masks, classes):
                    # 마스크를 원본 프레임 크기로 리사이즈
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    binary = (mask_resized > 0.5).astype(np.uint8)

                    # 클래스별 색상 (hue 기반)
                    hue = int(cls_id * 180 / 80) % 180
                    color_hsv = np.uint8([[[hue, 200, 200]]])
                    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

                    # 반투명 오버레이
                    colored = np.zeros_like(frame, dtype=np.uint8)
                    colored[binary == 1] = color_bgr
                    annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored, 0.5, 0)

                    # 윤곽선
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_frame, contours, -1, color_bgr, 2)

            # 바운딩박스 + 레이블 표시
            annotated_frame = r.plot(img=annotated_frame, masks=False)

            cv2.imshow("YOLOv8 Segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

segmentation_pipeline()