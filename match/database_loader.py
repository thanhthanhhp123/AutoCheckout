import os
import cv2
import json
import glob
from collections import defaultdict
import numpy as np

class DatabaseLoader:
    def __init__(self, database_path="database/"):
        """
        Khởi tạo DatabaseLoader
        
        Args:
            database_path: Đường dẫn thư mục database
        """
        self.database_path = database_path
        self.products = {}
        self.product_info = {}
        
        # Load database
        self.load_database()
    
    def load_database(self):
        """Load tất cả sản phẩm từ database"""
        if not os.path.exists(self.database_path):
            print(f"Database path {self.database_path} not found!")
            return
        
        print(f"Loading database from {self.database_path}...")
        
        # Tìm tất cả thư mục con (mỗi thư mục = 1 sản phẩm)
        product_dirs = [d for d in os.listdir(self.database_path) 
                       if os.path.isdir(os.path.join(self.database_path, d))]
        
        for product_id in product_dirs:
            product_path = os.path.join(self.database_path, product_id)
            
            # Load tất cả ảnh trong thư mục sản phẩm
            image_files = self.get_image_files(product_path)
            
            if image_files:
                self.products[product_id] = image_files
                print(f"Loaded {len(image_files)} images for product '{product_id}'")
            
            # Load thông tin sản phẩm nếu có file info.json
            info_file = os.path.join(product_path, "info.json")
            if os.path.exists(info_file):
                self.product_info[product_id] = self.load_product_info(info_file)
        
        print(f"Database loaded: {len(self.products)} products")
    
    def get_image_files(self, directory):
        """
        Lấy tất cả file ảnh trong thư mục
        
        Args:
            directory: Đường dẫn thư mục
            
        Returns:
            list: Danh sách đường dẫn file ảnh
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory, ext)
            image_files.extend(glob.glob(pattern))
            
            # Thêm uppercase extensions
            pattern_upper = os.path.join(directory, ext.upper())
            image_files.extend(glob.glob(pattern_upper))
        
        return sorted(image_files)
    
    def load_product_info(self, info_file):
        """
        Load thông tin sản phẩm từ file info.json
        
        Args:
            info_file: Đường dẫn file info.json
            
        Returns:
            dict: Thông tin sản phẩm
        """
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading product info from {info_file}: {e}")
            return {}
    
    def get_all_products(self):
        """
        Lấy tất cả sản phẩm trong database
        
        Returns:
            dict: {product_id: [image_paths]}
        """
        return self.products.copy()
    
    def get_product_images(self, product_id):
        """
        Lấy tất cả ảnh của một sản phẩm
        
        Args:
            product_id: ID sản phẩm
            
        Returns:
            list: Danh sách đường dẫn ảnh
        """
        return self.products.get(product_id, [])
    
    def get_product_info(self, product_id):
        """
        Lấy thông tin của một sản phẩm
        
        Args:
            product_id: ID sản phẩm
            
        Returns:
            dict: Thông tin sản phẩm
        """
        return self.product_info.get(product_id, {})
    
    def get_product_list(self):
        """
        Lấy danh sách tất cả product ID
        
        Returns:
            list: Danh sách product ID
        """
        return list(self.products.keys())
    
    def load_product_images(self, product_id):
        """
        Load tất cả ảnh của một sản phẩm vào memory
        
        Args:
            product_id: ID sản phẩm
            
        Returns:
            list: Danh sách OpenCV images
        """
        image_paths = self.get_product_images(product_id)
        images = []
        
        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                else:
                    print(f"Warning: Cannot load image {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        return images
    
    def get_database_stats(self):
        """
        Lấy thống kê database
        
        Returns:
            dict: Thống kê database
        """
        stats = {
            'total_products': len(self.products),
            'total_images': sum(len(images) for images in self.products.values()),
            'products_detail': {}
        }
        
        for product_id, images in self.products.items():
            stats['products_detail'][product_id] = {
                'image_count': len(images),
                'has_info': product_id in self.product_info
            }
        
        return stats
    
    def validate_database(self):
        """
        Kiểm tra tính hợp lệ của database
        
        Returns:
            dict: Kết quả validation
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Kiểm tra thư mục database có tồn tại
        if not os.path.exists(self.database_path):
            validation_results['valid'] = False
            validation_results['issues'].append(f"Database directory {self.database_path} does not exist")
            return validation_results
        
        # Kiểm tra từng sản phẩm
        for product_id, image_paths in self.products.items():
            # Kiểm tra có ảnh không
            if not image_paths:
                validation_results['warnings'].append(f"Product {product_id} has no images")
                continue
            
            # Kiểm tra từng ảnh có load được không
            valid_images = 0
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            valid_images += 1
                        else:
                            validation_results['warnings'].append(f"Cannot load image {img_path}")
                    except Exception as e:
                        validation_results['warnings'].append(f"Error loading image {img_path}: {e}")
                else:
                    validation_results['issues'].append(f"Image file {img_path} does not exist")
            
            # Cảnh báo nếu sản phẩm có ít ảnh
            if valid_images < 3:
                validation_results['warnings'].append(f"Product {product_id} has only {valid_images} valid images")
        
        if validation_results['issues']:
            validation_results['valid'] = False
        
        return validation_results
    
    def reload_database(self):
        """Reload database từ đầu"""
        print("Reloading database...")
        self.products = {}
        self.product_info = {}
        self.load_database()
    
    def add_product_images(self, product_id, image_paths):
        """
        Thêm ảnh cho sản phẩm
        
        Args:
            product_id: ID sản phẩm
            image_paths: Danh sách đường dẫn ảnh mới
        """
        if product_id not in self.products:
            self.products[product_id] = []
        
        # Kiểm tra và thêm ảnh hợp lệ
        valid_paths = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        valid_paths.append(img_path)
                    else:
                        print(f"Warning: Cannot load image {img_path}")
                except Exception as e:
                    print(f"Error validating image {img_path}: {e}")
            else:
                print(f"Warning: Image file {img_path} does not exist")
        
        self.products[product_id].extend(valid_paths)
        print(f"Added {len(valid_paths)} images to product {product_id}")
    
    def remove_product(self, product_id):
        """
        Xóa sản phẩm khỏi database
        
        Args:
            product_id: ID sản phẩm cần xóa
        """
        if product_id in self.products:
            del self.products[product_id]
            print(f"Removed product {product_id}")
        
        if product_id in self.product_info:
            del self.product_info[product_id]
    
    def get_random_product_image(self, product_id):
        """
        Lấy một ảnh ngẫu nhiên của sản phẩm
        
        Args:
            product_id: ID sản phẩm
            
        Returns:
            str: Đường dẫn ảnh ngẫu nhiên hoặc None
        """
        images = self.get_product_images(product_id)
        if images:
            return np.random.choice(images)
        return None
    
    def search_products_by_name(self, search_term):
        """
        Tìm kiếm sản phẩm theo tên
        
        Args:
            search_term: Từ khóa tìm kiếm
            
        Returns:
            list: Danh sách product ID phù hợp
        """
        results = []
        search_term = search_term.lower()
        
        for product_id in self.products.keys():
            # Tìm trong product_id
            if search_term in product_id.lower():
                results.append(product_id)
                continue
            
            # Tìm trong product info nếu có
            if product_id in self.product_info:
                info = self.product_info[product_id]
                if 'name' in info and search_term in info['name'].lower():
                    results.append(product_id)
                elif 'description' in info and search_term in info['description'].lower():
                    results.append(product_id)
        
        return results
    
    def export_database_info(self, output_file="database_info.json"):
        """
        Xuất thông tin database ra file JSON
        
        Args:
            output_file: Đường dẫn file output
        """
        db_info = {
            'database_path': self.database_path,
            'stats': self.get_database_stats(),
            'products': {}
        }
        
        for product_id, images in self.products.items():
            db_info['products'][product_id] = {
                'images': [os.path.basename(img) for img in images],
                'image_count': len(images),
                'info': self.get_product_info(product_id)
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(db_info, f, ensure_ascii=False, indent=2)
        
        print(f"Database info exported to {output_file}")

class DatabaseManager:
    """Helper class để quản lý database"""
    
    def __init__(self, database_path="database/"):
        self.database_path = database_path
        self.loader = DatabaseLoader(database_path)
    
    def create_product_structure(self, product_id):
        """
        Tạo cấu trúc thư mục cho sản phẩm mới
        
        Args:
            product_id: ID sản phẩm
        """
        product_dir = os.path.join(self.database_path, product_id)
        os.makedirs(product_dir, exist_ok=True)
        
        # Tạo file info.json template
        info_file = os.path.join(product_dir, "info.json")
        if not os.path.exists(info_file):
            default_info = {
                "name": product_id,
                "description": "",
                "category": "",
                "price": 0,
                "created_date": "",
                "tags": []
            }
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(default_info, f, ensure_ascii=False, indent=2)
        
        print(f"Created product structure for {product_id}")
    
    def copy_images_to_product(self, product_id, source_images, copy_images=True):
        """
        Copy ảnh vào thư mục sản phẩm
        
        Args:
            product_id: ID sản phẩm
            source_images: Danh sách đường dẫn ảnh nguồn
            copy_images: True để copy, False để chỉ symlink
        """
        product_dir = os.path.join(self.database_path, product_id)
        
        # Tạo thư mục nếu chưa có
        self.create_product_structure(product_id)
        
        copied_count = 0
        for i, src_img in enumerate(source_images):
            if not os.path.exists(src_img):
                print(f"Warning: Source image {src_img} does not exist")
                continue
            
            # Tạo tên file đích
            file_ext = os.path.splitext(src_img)[1]
            dst_img = os.path.join(product_dir, f"{product_id}_{i:03d}{file_ext}")
            
            try:
                if copy_images:
                    import shutil
                    shutil.copy2(src_img, dst_img)
                else:
                    os.symlink(src_img, dst_img)
                
                copied_count += 1
                
            except Exception as e:
                print(f"Error copying {src_img}: {e}")
        
        print(f"Copied {copied_count} images to product {product_id}")
        
        # Reload database
        self.loader.reload_database()
    
    def update_product_info(self, product_id, info_dict):
        """
        Cập nhật thông tin sản phẩm
        
        Args:
            product_id: ID sản phẩm
            info_dict: Dictionary chứa thông tin mới
        """
        product_dir = os.path.join(self.database_path, product_id)
        info_file = os.path.join(product_dir, "info.json")
        
        # Load thông tin hiện tại
        current_info = {}
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                current_info = json.load(f)
        
        # Cập nhật thông tin
        current_info.update(info_dict)
        
        # Lưu lại
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(current_info, f, ensure_ascii=False, indent=2)
        
        # Reload database
        self.loader.reload_database()
        
        print(f"Updated info for product {product_id}")
    
    def delete_product(self, product_id, confirm=True):
        """
        Xóa sản phẩm khỏi database
        
        Args:
            product_id: ID sản phẩm
            confirm: Yêu cầu xác nhận trước khi xóa
        """
        product_dir = os.path.join(self.database_path, product_id)
        
        if not os.path.exists(product_dir):
            print(f"Product {product_id} does not exist")
            return
        
        if confirm:
            response = input(f"Are you sure you want to delete product {product_id}? (y/N): ")
            if response.lower() != 'y':
                print("Delete cancelled")
                return
        
        try:
            import shutil
            shutil.rmtree(product_dir)
            print(f"Deleted product {product_id}")
            
            # Reload database
            self.loader.reload_database()
            
        except Exception as e:
            print(f"Error deleting product {product_id}: {e}")

def main():
    """Test DatabaseLoader"""
    # Test load database
    loader = DatabaseLoader("database/")
    
    # Print stats
    stats = loader.get_database_stats()
    print("\nDatabase Stats:")
    print(f"Total products: {stats['total_products']}")
    print(f"Total images: {stats['total_images']}")
    
    # Print product details
    print("\nProduct Details:")
    for product_id, detail in stats['products_detail'].items():
        print(f"  {product_id}: {detail['image_count']} images")
    
    # Validate database
    validation = loader.validate_database()
    print(f"\nDatabase valid: {validation['valid']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

if __name__ == "__main__":
    main()