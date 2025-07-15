import os
import json
import csv
import cv2
import numpy as np
from datetime import datetime
from match.matcher import ProductMatcher
from match.database_loader import DatabaseLoader
import glob

class AutoCheckoutMatcher:
    def __init__(self, database_path="database/", prices_file="prices.json", crops_path="crops/"):
        self.database_path = database_path
        self.prices_file = prices_file
        self.base_crops_path = crops_path

        # Nếu có biến môi trường CROPS_PATH thì dùng session đó, ngược lại lấy session mới nhất
        crops_env = os.environ.get("CROPS_PATH")
        if crops_env and os.path.isdir(crops_env):
            self.crops_path = crops_env
            self.session_id = os.path.basename(self.crops_path).replace("session_", "")
        else:
            # Lấy session gần nhất trong crops/
            self.crops_path, self.session_id = self.get_latest_session_dir_and_id(self.base_crops_path)
            if self.crops_path is None:
                # Nếu không có session nào thì tạo mới
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.crops_path = os.path.join(self.base_crops_path, f"session_{self.session_id}")
                os.makedirs(self.crops_path, exist_ok=True)

        # Load database và prices
        self.db_loader = DatabaseLoader(database_path)
        self.matcher = ProductMatcher()
        self.prices = self.load_prices()

        # Tạo hoặc load features database cho matching
        self.feature_db = self.matcher.ensure_feature_database(self.db_loader.get_all_products())

        # Danh sách sản phẩm đã match trong session hiện tại
        self.matched_products = []

    @staticmethod
    def get_latest_session_dir_and_id(crops_base_dir):
        """Lấy thư mục session gần nhất và session_id trong crops/"""
        if not os.path.isdir(crops_base_dir):
            return None, None
        session_dirs = [d for d in os.listdir(crops_base_dir) if d.startswith("session_") and os.path.isdir(os.path.join(crops_base_dir, d))]
        if not session_dirs:
            return None, None
        session_dirs.sort(reverse=True)
        latest_dir = session_dirs[0]
        return os.path.join(crops_base_dir, latest_dir), latest_dir.replace("session_", "")
        
    def load_prices(self):
        """Load giá sản phẩm từ file JSON"""
        try:
            with open(self.prices_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Không tìm thấy file {self.prices_file}")
            return {}
    
    def get_crop_images(self):
        """Lấy tất cả ảnh crop từ thư mục crops/"""
        crop_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            crop_files.extend(glob.glob(os.path.join(self.crops_path, ext)))
        return sorted(crop_files)
    
    def match_single_product(self, crop_image_path, confidence_threshold=0.7):
        """
        Match 1 sản phẩm crop với database
        
        Args:
            crop_image_path: Đường dẫn ảnh crop
            confidence_threshold: Ngưỡng confidence để accept match
            
        Returns:
            dict: Thông tin match (product_id, confidence, price)
        """
        crop_image = cv2.imread(crop_image_path)
        if crop_image is None:
            return None

        # Trích xuất đặc trưng từ ảnh crop
        crop_features = self.matcher.extract_features(crop_image)

        best_match = None
        best_confidence = 0
        best_db_image_path = None

        # So sánh với tất cả sản phẩm trong feature_db
        for product_id, features_arr in self.feature_db.items():
            if features_arr.shape[0] == 0:
                continue
            # Tính similarity với từng vector đặc trưng của sản phẩm
            sims = [self.matcher.calculate_similarity(crop_features, features_arr[i]) for i in range(features_arr.shape[0])]
            max_sim = max(sims)
            if max_sim > best_confidence:
                best_confidence = max_sim
                best_match = {
                    'product_id': product_id,
                    'confidence': max_sim,
                    'price': self.prices.get(product_id, 0),
                    'db_image_path': ''  # Không lưu cụ thể ảnh gốc nào, chỉ lưu product_id
                }

        # Chỉ trả về nếu confidence >= threshold
        if best_match and best_confidence >= confidence_threshold:
            return best_match

        return None
    
    def match_all_products(self, confidence_threshold=0.7):
        """
        Match tất cả sản phẩm crop với database
        
        Args:
            confidence_threshold: Ngưỡng confidence để accept match
            
        Returns:
            list: Danh sách các sản phẩm đã match
        """
        crop_images = self.get_crop_images()
        matched_products = []
        
        print(f"Đang xử lý {len(crop_images)} ảnh crop...")
        
        for i, crop_path in enumerate(crop_images):
            print(f"Đang xử lý ảnh {i+1}/{len(crop_images)}: {os.path.basename(crop_path)}")
            
            match_result = self.match_single_product(crop_path, confidence_threshold)
            
            if match_result:
                match_result['crop_image_path'] = crop_path
                matched_products.append(match_result)
                print(f"  -> Match: {match_result['product_id']} (confidence: {match_result['confidence']:.3f})")
            else:
                print(f"  -> Không tìm thấy match phù hợp")
        
        self.matched_products = matched_products
        return matched_products
    
    def calculate_total(self):
        """Tính tổng tiền của tất cả sản phẩm đã match"""
        total = 0
        product_count = {}
        
        for product in self.matched_products:
            product_id = product['product_id']
            price = product['price']
            
            if product_id in product_count:
                product_count[product_id]['count'] += 1
                product_count[product_id]['total_price'] += price
            else:
                product_count[product_id] = {
                    'count': 1,
                    'unit_price': price,
                    'total_price': price
                }
            
            total += price
        
        return total, product_count
    
    def get_session_path(self, filename):
        """
        Tạo đường dẫn file trong thư mục session hiện tại
        
        Args:
            filename: Tên file cần tạo đường dẫn
            
        Returns:
            str: Đường dẫn đầy đủ trong thư mục session
        """
        return os.path.join(self.crops_path, filename)
    
    def generate_receipt(self, output_file=None):
        """
        Tạo hóa đơn và lưu vào file CSV
        
        Args:
            output_file: Đường dẫn file CSV output. Nếu None, sẽ lưu trong thư mục session
        """
        if not self.matched_products:
            print("Không có sản phẩm nào để tạo hóa đơn")
            return
            
        if output_file is None:
            output_file = self.get_session_path("receipt.csv")
            
        total, product_count = self.calculate_total()
        
        # Tạo hóa đơn
        receipt_data = []
        
        # Header
        receipt_data.append(['=== HÓA ĐƠN THANH TOÁN ==='])
        receipt_data.append(['Thời gian:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        receipt_data.append([''])
        receipt_data.append(['Sản phẩm', 'Số lượng', 'Đơn giá', 'Thành tiền'])
        receipt_data.append(['=' * 50])
        
        # Chi tiết sản phẩm
        for product_id, info in product_count.items():
            receipt_data.append([
                product_id,
                info['count'],
                f"{info['unit_price']:,.0f} VND",
                f"{info['total_price']:,.0f} VND"
            ])
        
        # Tổng cộng
        receipt_data.append(['=' * 50])
        receipt_data.append(['TỔNG CỘNG', '', '', f"{total:,.0f} VND"])
        
        # Lưu vào file CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(receipt_data)
        
        print(f"Đã tạo hóa đơn: {output_file}")
        
        # In ra console
        print("\n" + "="*50)
        print("HÓA ĐƠN THANH TOÁN")
        print("="*50)
        print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"{'Sản phẩm':<20} {'SL':<5} {'Đơn giá':<15} {'Thành tiền':<15}")
        print("-"*50)
        
        for product_id, info in product_count.items():
            print(f"{product_id:<20} {info['count']:<5} {info['unit_price']:>10,.0f} VND {info['total_price']:>10,.0f} VND")
        
        print("-"*50)
        print(f"{'TỔNG CỘNG':<20} {'':<5} {'':<15} {total:>10,.0f} VND")
        print("="*50)
    
    def save_match_results(self, output_file=None):
        """
        Lưu kết quả matching vào file JSON
        
        Args:
            output_file: Đường dẫn file JSON output. Nếu None, sẽ lưu trong thư mục session
        """
        if not self.matched_products:
            print("Không có kết quả matching để lưu")
            return
            
        if output_file is None:
            output_file = self.get_session_path("match_results.json")
        
        # Chuẩn bị dữ liệu
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_products': len(self.matched_products),
            'total_amount': self.calculate_total()[0],
            'products': []
        }
        
        for product in self.matched_products:
            results['products'].append({
                'product_id': product['product_id'],
                'confidence': product['confidence'],
                'price': product['price'],
                'crop_image': os.path.basename(product['crop_image_path']),
                'db_image': os.path.basename(product['db_image_path'])
            })
        
        # Lưu vào file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu kết quả matching: {output_file}")
    
    def clear_crops(self):
        """Xóa thư mục crop của session hiện tại sau khi xử lý xong"""
        if os.path.exists(self.crops_path):
            try:
                import shutil
                shutil.rmtree(self.crops_path)
                print(f"Đã xóa thư mục session: {self.crops_path}")
            except Exception as e:
                print(f"Lỗi khi xóa thư mục session {self.crops_path}: {e}")
    
    def run_checkout_process(self, confidence_threshold=0.7, clear_crops_after=True):
        """
        Chạy toàn bộ quy trình checkout
        
        Args:
            confidence_threshold: Ngưỡng confidence để accept match
            clear_crops_after: Có xóa thư mục session sau khi xử lý không
        """
        print(f"Bắt đầu quá trình checkout... [Session: {self.session_id}]")
        print(f"Thư mục session: {self.crops_path}")
        
        # 1. Match tất cả sản phẩm
        matched_products = self.match_all_products(confidence_threshold)
        
        if not matched_products:
            print("Không tìm thấy sản phẩm nào để thanh toán")
            if clear_crops_after:
                self.clear_crops()
            return
        
        # 2. Tạo hóa đơn
        self.generate_receipt()
        
        # 3. Lưu kết quả matching
        self.save_match_results()
        
        # 4. Xóa thư mục session nếu cần
        if clear_crops_after:
            self.clear_crops()
            
        print(f"Hoàn thành quá trình checkout! [Session: {self.session_id}]")

def main():
    """Hàm main để test"""
    # Khởi tạo matcher
    matcher = AutoCheckoutMatcher()
    
    # Chạy quá trình checkout
    matcher.run_checkout_process(confidence_threshold=0.7, clear_crops_after=False)

if __name__ == "__main__":
    main()