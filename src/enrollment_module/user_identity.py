import sqlite3
from typing import Dict, Any
import time

# --- Configuration Constants ---
# Đảm bảo file database đã được khởi tạo bằng sqlite_setup.py trước khi chạy
DATABASE_FILE = "/home/un1/projects/facial_recognition/database/metadata.db" 

# --- Metadata Client Wrapper (SQLite) ---

class MetadataClientWrapper:    
    def __init__(self):
        """Thiết lập kết nối database SQLite."""
        try:
            self.conn = sqlite3.connect(DATABASE_FILE)
            self.conn.row_factory = sqlite3.Row # Cho phép truy cập cột bằng tên
            self.cursor = self.conn.cursor()
            print(f"SUCCESS: Đã kết nối Metadata DB: {DATABASE_FILE}")
        except sqlite3.Error as e:
            print(f"ERROR: Lỗi kết nối SQLite. Hãy đảm bảo đã chạy sqlite_setup.py. Chi tiết: {e}")
            self.conn = None

    def get_user_by_code(self, user_code: str) -> Dict[str, Any] | None:
        """Truy vấn hồ sơ người dùng bằng user_code (mã 10 số)."""
        if not self.conn: return None
        try:
            # Chỉ cần SELECT user_id để xác nhận sự tồn tại.
            self.cursor.execute("SELECT user_id FROM user_profiles WHERE user_code = ?", (user_code,))
            result = self.cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            print(f"SQLite ERROR: Lỗi truy vấn user_code {user_code}: {e}")
            return None

    def close(self):
        """Đóng kết nối database và lưu lại thay đổi."""
        if self.conn:
            self.conn.close()
            print(f"INFO: Đã đóng kết nối tới {DATABASE_FILE}.")

class UserCodeCheckSystem:
    """Thực hiện Bước 1: User Identity Claim & Non-Biometric Uniqueness."""
    
    def __init__(self):
        self.metadata_client = MetadataClientWrapper()
        
    def check_user_code_uniqueness(self, user_code: str) -> str:
        """
        Kiểm tra tính duy nhất của user_code (mã 10 số) trong Metadata DB.
        Sử dụng thông báo lỗi chung chung khi trùng lặp.
        """
        print(f"\n--- BƯỚC 1: KIỂM TRA TÍNH DUY NHẤT (Mã: {user_code}) ---")
        
        existing_profile = self.metadata_client.get_user_by_code(user_code)
        
        if existing_profile:
            return f"THẤT BẠI: Mã người dùng '{user_code}' đã được đăng ký. Vui lòng kiểm tra lại thông tin hoặc liên hệ quản trị."
        else:
            return f"THÀNH CÔNG: Mã người dùng '{user_code}' là duy nhất. Có thể tiến hành các bước đăng ký tiếp theo."

# --- Execution Example ---

def run_check_example():
    """Minh họa kiểm tra mã người dùng duy nhất và trùng lặp."""
    
    system = UserCodeCheckSystem()
    
    # 1. Kiểm tra mã DUY NHẤT (Giả định không có trong DB)
    NEW_UNIQUE_CODE = "9998887770" 
    result_unique = system.check_user_code_uniqueness(NEW_UNIQUE_CODE)
    print(result_unique)
    
    # 2. Tạo một bản ghi mẫu (chèn tạm thời) để kiểm tra lỗi trùng lặp
    DUPLICATE_CODE = "1112223330" 
    mock_id = "mock-uuid-12345"
    
    try:
        system.metadata_client.cursor.execute("""
            INSERT OR IGNORE INTO user_profiles (user_id, user_code, account_status, is_enrolled, enrollment_date, biometric_consent_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            mock_id,
            DUPLICATE_CODE,
            'ACTIVE',
            0,
            int(time.time()),
            int(time.time())
        ))
        system.metadata_client.conn.commit()
        print(f"\n[INFO]: Đã chèn tạm thời user_code '{DUPLICATE_CODE}' để thử nghiệm trùng lặp.")
    except Exception as e:
        print(f"[INFO]: Không thể chèn tạm thời user_code (có thể đã tồn tại hoặc lỗi): {e}")

    # 3. Kiểm tra mã TRÙNG LẶP (Sẽ thất bại với thông báo an toàn)
    result_duplicate = system.check_user_code_uniqueness(DUPLICATE_CODE)
    print(result_duplicate)
    
    # 4. Dọn dẹp kết nối
    system.metadata_client.close()
    
if __name__ == "__main__":
    run_check_example()
