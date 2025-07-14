import cv2
import numpy as np
import time

class CameraManager:
    def __init__(self):
        """Camera manager để xử lý camera input"""
        self.cap = None
        self.camera_id = None
        
    def open_camera(self, camera_id=0, width=640, height=480, fps=30):
        """
        Mở camera
        
        Args:
            camera_id: ID của camera
            width: Độ rộng frame
            height: Chiều cao frame
            fps: FPS
            
        Returns:
            cap: Camera capture object hoặc None nếu lỗi
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            self.camera_id = camera_id
            
            if not self.cap.isOpened():
                print(f"❌ Cannot open camera {camera_id}")
                return None
            
            # Cài đặt độ phân giải
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Kiểm tra độ phân giải thực tế
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera {camera_id} opened successfully")
            print(f"📐 Resolution: {actual_width}x{actual_height}")
            print(f"🎬 FPS: {actual_fps}")
            
            return self.cap
            
        except Exception as e:
            print(f"❌ Error opening camera: {e}")
            return None
    
    def release_camera(self):
        """Giải phóng camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"📷 Camera {self.camera_id} released")
    
    def get_frame(self):
        """
        Lấy frame từ camera
        
        Returns:
            frame: Frame ảnh hoặc None nếu lỗi
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
    
    def is_opened(self):
        """Kiểm tra camera có mở không"""
        return self.cap is not None and self.cap.isOpened()
    
    def list_cameras(self, max_check=10):
        """
        Liệt kê các camera có sẵn
        
        Args:
            max_check: Số lượng camera tối đa cần kiểm tra
            
        Returns:
            available_cameras: List các camera ID có sẵn
        """
        available_cameras = []
        
        print("🔍 Checking available cameras...")
        
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                print(f"✅ Camera {i} available")
                cap.release()
            else:
                print(f"❌ Camera {i} not available")
        
        return available_cameras
    
    def get_camera_info(self):
        """
        Lấy thông tin camera hiện tại
        
        Returns:
            info: Dict thông tin camera
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        info = {
            'camera_id': self.camera_id,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'hue': self.cap.get(cv2.CAP_PROP_HUE)
        }
        
        return info
    
    def set_camera_property(self, property_id, value):
        """
        Cài đặt thuộc tính camera
        
        Args:
            property_id: ID thuộc tính (cv2.CAP_PROP_*)
            value: Giá trị cần cài đặt
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(property_id, value)
            print(f"📹 Camera property {property_id} set to {value}")
    
    def calibrate_camera_angle(self, frame):
        """
        Hiệu chỉnh góc camera (cho camera nhìn chéo)
        
        Args:
            frame: Frame ảnh từ camera
            
        Returns:
            corrected_frame: Frame đã hiệu chỉnh
        """
        # Placeholder cho perspective correction
        # Có thể implement perspective transform nếu cần
        return frame
    
    def apply_enhancement(self, frame, brightness=0, contrast=1.0, saturation=1.0):
        """
        Áp dụng enhancement cho frame
        
        Args:
            frame: Frame input
            brightness: Độ sáng (-100 to 100)
            contrast: Độ tương phản (0.0 to 3.0)
            saturation: Độ bão hòa (0.0 to 3.0)
            
        Returns:
            enhanced_frame: Frame đã enhance
        """
        # Brightness
        if brightness != 0:
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
        
        # Contrast
        if contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
        
        # Saturation
        if saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], saturation)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return frame


class VideoProcessor:
    def __init__(self, video_path):
        """
        Xử lý video file
        
        Args:
            video_path: Đường dẫn đến video file
        """
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        
    def open_video(self):
        """Mở video file"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"❌ Cannot open video: {self.video_path}")
                return False
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Video opened: {self.video_path}")
            print(f"📊 Total frames: {self.total_frames}")
            print(f"🎬 FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error opening video: {e}")
            return False
    
    def get_frame(self):
        """Lấy frame tiếp theo"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return frame
        else:
            return None
    
    def seek_frame(self, frame_number):
        """Nhảy đến frame cụ thể"""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
    
    def get_progress(self):
        """Lấy tiến độ xử lý video"""
        if self.total_frames == 0:
            return 0.0
        return self.current_frame / self.total_frames
    
    def release(self):
        """Giải phóng video"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# Test functions
def test_camera_manager():
    """Test camera manager"""
    camera_manager = CameraManager()
    
    # List available cameras
    available_cameras = camera_manager.list_cameras()
    print(f"Available cameras: {available_cameras}")
    
    if len(available_cameras) > 0:
        # Test first available camera
        cap = camera_manager.open_camera(available_cameras[0])
        if cap is not None:
            print("🎥 Testing camera. Press 'q' to quit")
            
            while True:
                frame = camera_manager.get_frame()
                if frame is not None:
                    # Apply some enhancements
                    enhanced_frame = camera_manager.apply_enhancement(
                        frame, brightness=10, contrast=1.1
                    )
                    
                    cv2.imshow('Camera Test', enhanced_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("❌ Failed to get frame")
                    break
            
            camera_manager.release_camera()
            cv2.destroyAllWindows()
    else:
        print("❌ No cameras available")


def test_video_processor():
    """Test video processor"""
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "test.mp4"  # Default test video
    
    processor = VideoProcessor(video_path)
    
    if processor.open_video():
        print("🎬 Processing video. Press 'q' to quit, 'space' to pause")
        
        paused = False
        while True:
            if not paused:
                frame = processor.get_frame()
                if frame is not None:
                    # Show progress
                    progress = processor.get_progress()
                    cv2.putText(frame, f"Progress: {progress:.1%}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Video Test', frame)
                else:
                    print("📹 Video ended")
                    break
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"⏸️ {'Paused' if paused else 'Resumed'}")
        
        processor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Select test:")
    print("1. Test camera")
    print("2. Test video processor")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        test_camera_manager()
    elif choice == "2":
        test_video_processor()
    else:
        print("❌ Invalid choice")