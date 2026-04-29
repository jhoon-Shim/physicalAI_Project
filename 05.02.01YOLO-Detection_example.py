import cv2
import numpy as np
from ultralytics import YOLO

import cv2
from ultralytics import YOLO

def edge_detection_pipeline():
    model = YOLO('yolov8n.pt')
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
            annotated_frame = r.plot() # OpenCV
            cv2.imshow("YOLOv8 Real-time Inference", annotated_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

edge_detection_pipeline()