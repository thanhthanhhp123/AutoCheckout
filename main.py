import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from detect.detector import Detector
from utils.camera_utils import CameraManager
from utils.crop_utils import CropManager

class AutoCheckoutStage1:
    def __init__(self, 
                 model_path="yolov8n.pt", 
                 confidence_threshold=0.5,
                 crops_dir="crops",
                 session_timeout=30):
        """
        Khởi tạo Stage 1 của hệ thống AutoCheckout
        
        Args:
            model_path: Đường dẫn đến model YOLOv8
            confidence_threshold: Ngưỡng confidence cho detection
            crops_dir: Thư mục lưu ảnh crop
            session_timeout: Thời gian timeout cho session (giây)
        """
        self.detector = Detector(model_path, confidence_threshold)
        self.tracker = DeepSort(max_age=50, n_init=3)
        self.camera_manager = CameraManager()
        self.crop_manager = CropManager(crops_dir)
        
        # Tracking state
        self.tracked_objects = {}  # {track_id: object_info}
        self.session_active = False
        self.session_start_time = None
        self.session_timeout = session_timeout
        
        # Ensure crops directory exists
        os.makedirs(crops_dir, exist_ok=True)
        
    def start_session(self):
        """Bắt đầu session mới"""
        self.session_active = True
        self.session_start_time = time.time()
        self.tracked_objects.clear()
        
        # Tạo thư mục cho session hiện tại
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.crop_manager.crops_dir, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        self.current_session_dir = session_dir
        
        print(f"🛒 Started new checkout session: {session_id}")
        return session_id
    
    def end_session(self):
        """Kết thúc session hiện tại"""
        if self.session_active:
            self.session_active = False
            session_summary = self.get_session_summary()
            print(f"📝 Session ended. Objects tracked: {len(self.tracked_objects)}")
            return session_summary
        return None
    
    def get_session_summary(self):
        """Lấy thông tin tổng kết session"""
        summary = {
            'session_id': os.path.basename(self.current_session_dir),
            'start_time': self.session_start_time,
            'end_time': time.time(),
            'objects_count': len(self.tracked_objects),
            'objects': []
        }
        
        for track_id, obj_info in self.tracked_objects.items():
            summary['objects'].append({
                'track_id': track_id,
                'first_seen': obj_info['first_seen'],
                'last_seen': obj_info['last_seen'],
                'crop_count': len(obj_info['crops']),
                'best_crop': obj_info['best_crop']
            })
        
        return summary
    
    def check_session_timeout(self):
        """Kiểm tra timeout của session"""
        if self.session_active and self.session_start_time:
            if time.time() - self.session_start_time > self.session_timeout:
                print("⏰ Session timeout!")
                return self.end_session()
        return None
    
    def calculate_crop_area(self, crop_img):
        """Tính diện tích của crop image"""
        if crop_img.size == 0:
            return 0
        h, w = crop_img.shape[:2]
        return w * h

    def process_frame(self, frame):
        """
        Xử lý một frame từ camera/video
        
        Args:
            frame: Frame ảnh từ camera
            
        Returns:
            annotated_frame: Frame đã được vẽ bounding box và track ID
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        if len(detections) == 0:
            return frame
        
        # Convert detections to DeepSORT format
        detection_list = []
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            detection_list.append([[x1, y1, x2-x1, y2-y1], conf, class_id])
        
        # Update tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Process tracked objects
        annotated_frame = frame.copy()
        current_time = time.time()
        
        for track in tracks:
            if not track.is_confirmed():
                continue

            crop_score = 0.0
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Crop object image
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size > 0:
                # Calculate crop quality score
                crop_score = self.calculate_crop_quality(crop_img)
                crop_area = self.calculate_crop_area(crop_img)
                
                # Initialize tracked object if not exists
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'crops': [],
                        'best_crop': None,
                        'best_score': 0,
                        'max_area': 0
                    }
                
                obj_info = self.tracked_objects[track_id]
                obj_info['last_seen'] = current_time
                
                # Save crop if session is active and area is larger than previous max
                if self.session_active and crop_area > obj_info['max_area']:
                    crop_filename = f"track_{track_id}_best.jpg"
                    crop_path = os.path.join(self.current_session_dir, crop_filename)
                    cv2.imwrite(crop_path, crop_img)
                    
                    # Update object info
                    obj_info['crops'] = [{'path': crop_path,
                                        'timestamp': current_time,
                                        'bbox': (x1, y1, x2, y2),
                                        'score': crop_score,
                                        'area': crop_area}]
                    obj_info['best_crop'] = crop_path
                    obj_info['best_score'] = crop_score
                    obj_info['max_area'] = crop_area

            # Draw bounding box and track ID
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"ID: {track_id}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw crop quality score
            cv2.putText(annotated_frame, f"Q: {crop_score:.2f}", 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw session status
        status_text = "SESSION ACTIVE" if self.session_active else "SESSION INACTIVE"
        status_color = (0, 255, 0) if self.session_active else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw object count
        if self.session_active:
            cv2.putText(annotated_frame, f"Objects: {len(self.tracked_objects)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def calculate_crop_quality(self, crop_img):
        """
        Tính toán điểm quality của crop image
        Dựa trên kích thước và độ sắc nét
        """
        if crop_img.size == 0:
            return 0.0
            
        h, w = crop_img.shape[:2]
        
        # Size score (prefer larger objects)
        size_score = min(1.0, (w * h) / (100 * 100))  # Normalize to 100x100
        
        # Sharpness score using Laplacian variance
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)  # Normalize
        
        # Combined score
        quality_score = 0.6 * size_score + 0.4 * sharpness_score
        return quality_score
    
    def run_camera(self, camera_id=0):
        """
        Chạy với camera real-time
        
        Args:
            camera_id: ID của camera (0 cho camera mặc định)
        """
        cap = self.camera_manager.open_camera(camera_id)
        if cap is None:
            print("❌ Cannot open camera!")
            return
        
        print("🎥 Camera started. Press 's' to start session, 'e' to end session, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Check session timeout
            self.check_session_timeout()
            
            # Display frame
            cv2.imshow('AutoCheckout Stage 1', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not self.session_active:
                    self.start_session()
            elif key == ord('e'):
                if self.session_active:
                    summary = self.end_session()
                    if summary:
                        print(f"Session summary: {summary}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        """
        Chạy với video file
        
        Args:
            video_path: Đường dẫn đến video file
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Auto start session for video processing
        self.start_session()
        
        print(f"🎬 Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("📹 Video ended")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('AutoCheckout Stage 1 - Video', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
        
        # End session and show summary
        summary = self.end_session()
        if summary:
            print(f"\n📊 Video processing completed!")
            print(f"Objects tracked: {summary['objects_count']}")
            print(f"Session duration: {summary['end_time'] - summary['start_time']:.2f}s")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function để test Stage 1"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoCheckout Stage 1 - Detection and Tracking')
    parser.add_argument('--source', type=str, default='camera', 
                       help='Source: "camera" or path to video file')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--model', type=str, default=r'weights\best.pt',
                       help='YOLOv8 model path')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold')
    parser.add_argument('--crops-dir', type=str, default='crops',
                       help='Directory to save cropped images')
    
    args = parser.parse_args()
    
    # Initialize Stage 1
    stage1 = AutoCheckoutStage1(
        model_path=args.model,
        confidence_threshold=args.confidence,
        crops_dir=args.crops_dir
    )
    
    # Run based on source
    if args.source == 'camera':
        stage1.run_camera(args.camera_id)
    else:
        if os.path.exists(args.source):
            stage1.run_video(args.source)
        else:
            print(f"❌ Video file not found: {args.source}")


if __name__ == "__main__":
    main()