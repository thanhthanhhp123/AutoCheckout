import subprocess
import sys
import os

"""
Chạy đồng thời Stage 1 (main.py) và Stage 2 (match_all.py)
- Stage 1: Xử lý detection, crop và lưu ảnh vào thư mục session
- Stage 2: Thực hiện matching và xuất hóa đơn dựa trên session mới nhất
"""

def get_latest_session_dir(crops_base_dir="crops"):
    """Lấy thư mục session mới nhất trong crops/"""
    session_dirs = [d for d in os.listdir(crops_base_dir) if d.startswith("session_")]
    if not session_dirs:
        return None
    session_dirs.sort(reverse=True)
    return os.path.join(crops_base_dir, session_dirs[0])

def run_stage1():
    """Chạy Stage 1 (main.py) ở chế độ camera"""
    return subprocess.Popen([sys.executable, "main.py", "--source", r"videos\test2.mp4"])

def run_stage2(session_dir):
    """Chạy Stage 2 (match_all.py) với crops_path là session_dir"""
    return subprocess.Popen([sys.executable, "match_all.py"], env={**os.environ, "CROPS_PATH": session_dir})

def main():
    # Chạy Stage 1
    print("[Pipeline] Khởi động Stage 1 (main.py)...")
    p1 = run_stage1()
    
    # Đợi user kết thúc session ở Stage 1 (hoặc có thể polling thư mục crops/)
    input("\nNhấn Enter sau khi kết thúc session ở Stage 1 để tiếp tục sang Stage 2...")
    
    # Lấy session mới nhất
    session_dir = get_latest_session_dir()
    if not session_dir:
        print("Không tìm thấy session nào trong crops/!")
        p1.terminate()
        return
    print(f"[Pipeline] Sử dụng session: {session_dir}")
    
    # Chạy Stage 2 với session_dir
    print("[Pipeline] Khởi động Stage 2 (match_all.py)...")
    p2 = run_stage2(session_dir)
    p2.wait()
    
    # Kết thúc Stage 1 nếu còn chạy
    if p1.poll() is None:
        p1.terminate()
    print("[Pipeline] Đã hoàn thành cả hai stage.")

if __name__ == "__main__":
    main()
