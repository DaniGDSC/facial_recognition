import sqlite3
import os
import time

DATABASE_FILE = "database/metadata.db"

class MetadataDBSetup:

    def __init__(self):
        # Kết nối đến database (tạo file nếu chưa tồn tại)
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()
        print(f"SUCCESS: Đã kết nối/tạo file database SQLite: {DATABASE_FILE}")
        
        # Thiết lập để truy xuất các cột dưới dạng dictionary (rất hữu ích)
        self.conn.row_factory = sqlite3.Row 

    def create_user_profiles_table(self):
        """Tạo bảng để lưu trữ thông tin người dùng và liên kết với Milvus."""
        try:
            # Xóa bảng cũ nếu tồn tại để đảm bảo schema mới được áp dụng
            self.cursor.execute("DROP TABLE IF EXISTS user_profiles")

            self.cursor.execute("""
                CREATE TABLE user_profiles (
                    user_id TEXT PRIMARY KEY,               -- UUID Liên kết chính với Milvus Vector
                    user_code TEXT UNIQUE NOT NULL,         -- MÃ 10 SỐ CỦA USER (PII), phải là DUY NHẤT
                    first_name TEXT,
                    last_name TEXT,
                    account_status TEXT NOT NULL,           -- Ví dụ: 'ACTIVE', 'LOCKED', 'PENDING'
                    enrollment_date INTEGER,                -- Timestamp ngày đăng ký
                    biometric_consent_date INTEGER          -- Timestamp ngày đồng ý sinh trắc học 
                );
            """)
            print("SUCCESS: Đã tạo bảng 'user_profiles' với cột 'user_code' (mã 10 số) duy nhất.")
        except sqlite3.Error as e:
            print(f"ERROR: Lỗi khi tạo bảng user_profiles: {e}")

    def create_audit_log_table(self):
        """Tạo bảng để ghi lại tất cả các lần truy cập và kiểm tra sinh trắc học."""
        try:
            # Xóa bảng cũ nếu tồn tại để đảm bảo schema mới được áp dụng
            self.cursor.execute("DROP TABLE IF EXISTS biometric_audit_log")

            self.cursor.execute("""
                CREATE TABLE biometric_audit_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,                               -- Liên kết đến user_profiles
                    timestamp INTEGER NOT NULL,                 -- Thời điểm sự kiện xảy ra
                    event_type TEXT NOT NULL,                   -- Ví dụ: 'Enroll', 'Verify_Success', 'Verify_Denied', 'Liveness_Fail'
                    similarity_score REAL,                      -- Điểm số sinh trắc học 1:1 hoặc 1:N
                    spoof_detected INTEGER,                     -- 1 nếu phát hiện giả mạo, 0 nếu không
                    device_ip TEXT,                             -- Địa chỉ IP của thiết bị
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );
            """)
            print("SUCCESS: Đã tạo hoặc xác nhận bảng 'biometric_audit_log'.")
        except sqlite3.Error as e:
            print(f"ERROR: Lỗi khi tạo bảng biometric_audit_log: {e}")

    def close(self):
        """Đóng kết nối database và lưu lại thay đổi."""
        self.conn.commit()
        self.conn.close()
        print(f"INFO: Đã đóng kết nối tới {DATABASE_FILE}.")

def main():
    """Chạy quy trình thiết lập database Metadata."""
    print("--- KHỞI TẠO METADATA DATABASE (SQLITE) ---")
    setup = None
    # Xóa file cũ để bắt đầu sạch sẽ với schema mới (chỉ dùng khi phát triển)
    if os.path.exists(DATABASE_FILE):
         os.remove(DATABASE_FILE)
         print(f"INFO: Đã xóa file cũ '{DATABASE_FILE}' để đảm bảo schema mới.")

    try:
        # Khởi tạo kết nối
        setup = MetadataDBSetup()

        # Tạo các bảng
        setup.create_user_profiles_table()
        setup.create_audit_log_table()

    except RuntimeError:
        print("\nSETUP FAILED: Không thể tiếp tục.")
    finally:
        if setup:
            setup.close()

if __name__ == "__main__":
    main()
