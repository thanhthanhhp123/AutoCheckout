import cv2
import numpy as np
from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Khởi tạo detector với YOLOv8
        
        Args:
            model_path: Đường dẫn đến model YOLOv8
            confidence_threshold: Ngưỡng confidence cho detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Các class cần detect (có thể customize)
        self.target_classes = None  # None = detect all classes
        
        print(f"✅ Detector initialized with model: {model_path}")
        print(f"📊 Confidence threshold: {confidence_threshold}")
        
    def detect(self, frame):
        """
        Detect objects trong frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            detections: List of [x1, y1, x2, y2, confidence, class_id]
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence threshold
                    if confidence >= self.confidence_threshold:
                        # Filter by target classes if specified
                        if self.target_classes is None or class_id in self.target_classes:
                            detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return detections
    
    def get_class_name(self, class_id):
        """
        Lấy tên class từ class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            class_name: Tên class
        """
        return self.model.names[class_id]
    
    def set_target_classes(self, class_names):
        """
        Đặt các class cần detect
        
        Args:
            class_names: List tên các class cần detect
        """
        if class_names is None:
            self.target_classes = None
            return
        
        self.target_classes = []
        for name in class_names:
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == name.lower():
                    self.target_classes.append(class_id)
                    break
        
        print(f"🎯 Target classes set: {class_names}")
        print(f"📋 Class IDs: {self.target_classes}")
    
    def annotate_frame(self, frame, detections):
        """
        Vẽ bounding box lên frame
        
        Args:
            frame: Input frame
            detections: List detections từ detect()
            
        Returns:
            annotated_frame: Frame đã vẽ bounding box
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Vẽ bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ label
            class_name = self.get_class_name(class_id)
            label = f"{class_name}: {confidence:.2f}"
            
            # Tính toán kích thước text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Vẽ background cho text
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - baseline),
                         (x1 + text_width, y1), 
                         (0, 255, 0), -1)
            
            # Vẽ text
            cv2.putText(annotated_frame, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def detect_and_annotate(self, frame):
        """
        Detect và vẽ bounding box trong một lệnh
        
        Args:
            frame: Input frame
            
        Returns:
            annotated_frame: Frame đã detect và vẽ bounding box
            detections: List detections
        """
        detections = self.detect(frame)
        annotated_frame = self.annotate_frame(frame, detections)
        return annotated_frame, detections


# Test function
def test_detector():
    """Test detector với camera hoặc video"""
    detector = Detector(model_path = 'content 2/runs/detect/custom_yolov82/weights/best.pt', confidence_threshold=0.7)
    
    # Test với video
    video_path = "test.mp4"  # Đường dẫn đến video test
    cap = cv2.VideoCapture(video_path)
    
    print("🎥 Testing detector with camera. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect và annotate
        annotated_frame, detections = detector.detect_and_annotate(frame)
        
        # Hiển thị thông tin
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Detector Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector()