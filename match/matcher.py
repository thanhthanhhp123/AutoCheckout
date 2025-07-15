import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
from sklearn.metrics.pairwise import cosine_similarity

class ProductMatcher:
    def __init__(self, model_name="ViT-B/32", device=None, features_dir="features", augment_times=3):
        """
        Khởi tạo ProductMatcher với CLIP model
        
        Args:
            model_name: Tên model CLIP (ViT-B/32, ViT-B/16, RN50, etc.)
            device: Device để chạy model (cuda/cpu)
            features_dir: Thư mục lưu features database
            augment_times: Số lần augment mỗi ảnh khi tạo features
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.features_dir = features_dir
        self.augment_times = augment_times

        # Load CLIP model
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print("ProductMatcher initialized successfully!")

    def ensure_feature_database(self, product_db):
        """
        Đảm bảo đã có features cho từng sản phẩm, nếu chưa thì tạo với augment data
        Args:
            product_db: dict {product_id: [image_paths]}
        Returns:
            feature_db: dict {product_id: [features]}
        """
        import os
        feature_db = {}
        os.makedirs(self.features_dir, exist_ok=True)
        for product_id, image_paths in product_db.items():
            feature_path = os.path.join(self.features_dir, f"{product_id}.npy")
            if os.path.exists(feature_path):
                # Load features nếu đã có
                features = np.load(feature_path)
                feature_db[product_id] = features
            else:
                # Tạo features với augment data
                all_features = []
                for img_path in image_paths:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    # Gốc
                    all_features.append(self.extract_features(image)[0])
                    # Augment
                    for _ in range(self.augment_times):
                        aug_img = augment_image(image)
                        all_features.append(self.extract_features(aug_img)[0])
                features_arr = np.stack(all_features) if all_features else np.empty((0,512))
                np.save(feature_path, features_arr)
                feature_db[product_id] = features_arr
        return feature_db
    
    def preprocess_image(self, image):
        """
        Tiền xử lý ảnh cho CLIP model
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def extract_features(self, image):
        """
        Trích xuất đặc trưng từ ảnh sử dụng CLIP
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=1, keepdim=True)  # Normalize
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        return features_np
    
    def calculate_similarity(self, features1, features2):
        """
        Tính độ tương đồng giữa 2 feature vectors
        
        Args:
            features1: Feature vector 1
            features2: Feature vector 2
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Ensure features are 2D arrays
        if features1.ndim == 1:
            features1 = features1.reshape(1, -1)
        if features2.ndim == 1:
            features2 = features2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(features1, features2)[0][0]
        
        return similarity
    
    def match_images(self, query_image, reference_images):
        """
        Match một ảnh query với danh sách ảnh reference
        
        Args:
            query_image: OpenCV image (query)
            reference_images: List of OpenCV images (reference)
            
        Returns:
            list: List of similarity scores
        """
        # Extract features from query image
        query_features = self.extract_features(query_image)
        
        similarities = []
        
        # Compare with each reference image
        for ref_image in reference_images:
            ref_features = self.extract_features(ref_image)
            similarity = self.calculate_similarity(query_features, ref_features)
            similarities.append(similarity)
        
        return similarities
    
    def find_best_match(self, query_image, reference_images, threshold=0.7):
        """
        Tìm ảnh reference phù hợp nhất với query image
        
        Args:
            query_image: OpenCV image (query)
            reference_images: List of OpenCV images (reference)
            threshold: Minimum similarity threshold
            
        Returns:
            dict: Best match info {index, similarity} or None
        """
        similarities = self.match_images(query_image, reference_images)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= threshold:
            return {
                'index': best_idx,
                'similarity': best_similarity
            }
        
        return None
    
    def batch_extract_features(self, images):
        """
        Trích xuất đặc trưng từ nhiều ảnh cùng lúc (batch processing)
        
        Args:
            images: List of OpenCV images
            
        Returns:
            numpy.ndarray: Feature matrix (n_images, feature_dim)
        """
        if not images:
            return np.array([])
        
        # Preprocess all images
        image_tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            image_tensors.append(tensor)
        
        # Batch process
        batch_tensor = torch.cat(image_tensors, dim=0)
        
        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features = features / features.norm(dim=1, keepdim=True)  # Normalize
        
        return features.cpu().numpy()
    
    def create_feature_database(self, image_paths):
        """
        Tạo database đặc trưng từ danh sách đường dẫn ảnh
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            dict: Feature database {path: features}
        """
        feature_db = {}
        
        print(f"Creating feature database from {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Cannot load image {image_path}")
                    continue
                
                # Extract features
                features = self.extract_features(image)
                feature_db[image_path] = features
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Feature database created with {len(feature_db)} images")
        return feature_db
    
    def save_features(self, features, save_path):
        """
        Lưu features vào file
        
        Args:
            features: Feature array hoặc feature database
            save_path: Đường dẫn file để lưu
        """
        np.save(save_path, features)
        print(f"Features saved to {save_path}")
    
    def load_features(self, load_path):
        """
        Load features từ file
        
        Args:
            load_path: Đường dẫn file để load
            
        Returns:
            numpy.ndarray: Loaded features
        """
        features = np.load(load_path, allow_pickle=True)
        print(f"Features loaded from {load_path}")
        return features

# Utility functions
def resize_image(image, target_size=(224, 224)):
    """
    Resize ảnh về kích thước target
    
    Args:
        image: OpenCV image
        target_size: Target size (width, height)
        
    Returns:
        OpenCV image: Resized image
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Normalize ảnh về range [0, 1]
    
    Args:
        image: OpenCV image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    return image.astype(np.float32) / 255.0

def augment_image(image, rotation_range=10, brightness_range=0.2):
    """
    Augment ảnh để tăng tính robust
    
    Args:
        image: OpenCV image
        rotation_range: Rotation range in degrees
        brightness_range: Brightness change range
        
    Returns:
        OpenCV image: Augmented image
    """
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness
    if brightness_range > 0:
        brightness_factor = np.random.uniform(1-brightness_range, 1+brightness_range)
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    return image