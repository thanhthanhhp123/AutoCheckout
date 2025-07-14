import cv2
import numpy as np
import os
import json
from datetime import datetime
import hashlib

class CropManager:
    def __init__(self, crops_dir="crops"):
        """
        Quản lý việc crop và lưu ảnh objects
        
        Args:
            crops_dir: Thư mục lưu ảnh crop
        """
        self.crops_dir = crops_dir
        self.ensure_directory_exists()
        
    def ensure_directory_exists(self):
        """Tạo thư mục crops nếu chưa có"""
        os.makedirs(self.crops_dir, exist_ok=True)
        
    def crop_object(self, frame, bbox, padding=10):
        """
        Crop object từ frame
        
        Args:
            frame: Frame ảnh gốc
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding xung quanh object
            
        Returns:
            cropped_img: Ảnh object đã crop
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Thêm padding và đảm bảo không vượt quá frame
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop image
        cropped_img = frame[y1:y2, x1:x2]
        
        return cropped_img
    
    def save_crop(self, cropped_img, filename=None, subfolder=None):
        """
        Lưu ảnh crop
        
        Args:
            cropped_img: Ảnh đã crop
            filename: Tên file (tự động tạo nếu None)
            subfolder: Thư mục con
            
        Returns:
            saved_path: Đường dẫn file đã lưu
        """
        if cropped_img is None or cropped_img.size == 0:
            return None
        
        # Tạo tên file nếu không có
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"crop_{timestamp}.jpg"
        
        # Tạo đường dẫn đầy đủ
        if subfolder is not None:
            save_dir = os.path.join(self.crops_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.crops_dir
        
        save_path = os.path.join(save_dir, filename)
        
        # Lưu ảnh
        try:
            cv2.imwrite(save_path, cropped_img)
            return save_path
        except Exception as e:
            print(f"❌ Error saving crop: {e}")
            return None
    
    def crop_and_save(self, frame, bbox, track_id=None, session_id=None, 
                     quality_score=None, padding=10):
        """
        Crop và lưu object trong một lệnh
        
        Args:
            frame: Frame ảnh gốc
            bbox: Bounding box
            track_id: ID của track
            session_id: ID của session
            quality_score: Điểm quality
            padding: Padding
            
        Returns:
            crop_info: Dict thông tin crop đã lưu
        """
        # Crop object
        cropped_img = self.crop_object(frame, bbox, padding)
        
        if cropped_img is None or cropped_img.size == 0:
            return None
        
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if track_id is not None:
            filename = f"track_{track_id}_{timestamp}.jpg"
        else:
            filename = f"crop_{timestamp}.jpg"
        
        # Tạo subfolder
        subfolder = None
        if session_id is not None:
            subfolder = f"session_{session_id}"
        
        # Lưu ảnh
        saved_path = self.save_crop(cropped_img, filename, subfolder)
        
        if saved_path is None:
            return None
        
        # Tạo thông tin crop
        crop_info = {
            'path': saved_path,
            'filename': filename,
            'bbox': bbox,
            'track_id': track_id,
            'session_id': session_id,
            'timestamp': timestamp,
            'quality_score': quality_score,
            'image_size': cropped_img.shape[:2],  # (height, width)
            'file_size': os.path.getsize(saved_path)
        }
        
        return crop_info
    
    def calculate_image_hash(self, image):
        """
        Tính hash của ảnh để check duplicate
        
        Args:
            image: Ảnh input
            
        Returns:
            hash_value: Hash string
        """
        # Resize ảnh để tính hash nhanh hơn
        small_img = cv2.resize(image, (64, 64))
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        # Tính hash
        hash_value = hashlib.md5(gray.tobytes()).hexdigest()
        
        return hash_value
    
    def is_duplicate(self, image, existing_hashes, threshold=0.95):
        """
        Kiểm tra ảnh có duplicate không
        
        Args:
            image: Ảnh cần kiểm tra
            existing_hashes: Dict {hash: image_info}
            threshold: Ngưỡng similarity
            
        Returns:
            is_dup: True nếu duplicate
        """
        current_hash = self.calculate_image_hash(image)
        
        # Simple hash comparison
        if current_hash in existing_hashes:
            return True
        
        # Có thể implement advanced similarity check ở đây
        return False
    
    def enhance_crop(self, cropped_img, enhance_type="auto"):
        """
        Enhance chất lượng ảnh crop
        
        Args:
            cropped_img: Ảnh crop
            enhance_type: Loại enhancement
            
        Returns:
            enhanced_img: Ảnh đã enhance
        """
        if cropped_img is None or cropped_img.size == 0:
            return cropped_img
        
        enhanced_img = cropped_img.copy()
        
        if enhance_type == "auto" or enhance_type == "contrast":
            # Tăng contrast
            enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=1.2, beta=10)
        
        if enhance_type == "auto" or enhance_type == "sharpen":
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
        
        if enhance_type == "auto" or enhance_type == "denoise":
            # Denoise
            enhanced_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)
        
        return enhanced_img
    
    def resize_crop(self, cropped_img, target_size=(224, 224), maintain_aspect=True):
        """
        Resize ảnh crop về kích thước chuẩn
        
        Args:
            cropped_img: Ảnh crop
            target_size: Kích thước đích (width, height)
            maintain_aspect: Giữ nguyên tỷ lệ
            
        Returns:
            resized_img: Ảnh đã resize
        """
        if cropped_img is None or cropped_img.size == 0:
            return cropped_img
        
        if maintain_aspect:
            # Resize giữ nguyên tỷ lệ
            h, w = cropped_img.shape[:2]
            target_w, target_h = target_size
            
            # Tính scale factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized_img = cv2.resize(cropped_img, (new_w, new_h))
            
            # Pad to target size
            if new_w != target_w or new_h != target_h:
                # Tạo black canvas
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Tính toán vị trí center
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                # Paste image vào center
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                resized_img = canvas
        else:
            # Resize trực tiếp
            resized_img = cv2.resize(cropped_img, target_size)
        
        return resized_img
    
    def create_crop_metadata(self, crop_info_list, session_id=None):
        """
        Tạo metadata file cho session crops
        
        Args:
            crop_info_list: List thông tin các crop
            session_id: ID session
            
        Returns:
            metadata_path: Đường dẫn file metadata
        """
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'total_crops': len(crop_info_list),
            'crops': crop_info_list
        }
        
        # Tạo tên file metadata
        if session_id is not None:
            metadata_filename = f"session_{session_id}_metadata.json"
            metadata_dir = os.path.join(self.crops_dir, f"session_{session_id}")
        else:
            metadata_filename = f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            metadata_dir = self.crops_dir
        
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        
        # Lưu metadata
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return metadata_path
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
            return None
    
    def load_crop_metadata(self, metadata_path):
        """
        Load metadata từ file
        
        Args:
            metadata_path: Đường dẫn file metadata
            
        Returns:
            metadata: Dict metadata
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None
    
    def cleanup_old_crops(self, days_old=7):
        """
        Xóa các crop cũ
        
        Args:
            days_old: Số ngày cũ
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        deleted_count = 0
        for root, dirs, files in os.walk(self.crops_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ Error deleting {file_path}: {e}")
        
        print(f"🗑️ Deleted {deleted_count} old crop files")


# Test function
def test_crop_manager():
    """Test crop manager"""
    # Tạo test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Tạo crop manager
    crop_manager = CropManager("test_crops")
    
    # Test crop and save
    bbox = (100, 100, 300, 300)
    crop_info = crop_manager.crop_and_save(
        test_img, bbox, track_id=1, session_id="test_session"
    )
    
    if crop_info:
        print(f"✅ Crop saved: {crop_info['path']}")
        print(f"📊 Crop info: {crop_info}")
        
        # Test load và enhance
        saved_img = cv2.imread(crop_info['path'])
        enhanced_img = crop_manager.enhance_crop(saved_img, "auto")
        resized_img = crop_manager.resize_crop(enhanced_img, (224, 224))
        
        # Test save enhanced
        enhanced_path = crop_manager.save_crop(
            enhanced_img, "enhanced_test.jpg", "test_session"
        )
        print(f"✅ Enhanced crop saved: {enhanced_path}")
        
        # Test metadata
        metadata_path = crop_manager.create_crop_metadata([crop_info], "test_session")
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Display results
        cv2.imshow('Original Crop', saved_img)
        cv2.imshow('Enhanced Crop', enhanced_img)
        cv2.imshow('Resized Crop', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("❌ Failed to crop and save")


if __name__ == "__main__":
    test_crop_manager()