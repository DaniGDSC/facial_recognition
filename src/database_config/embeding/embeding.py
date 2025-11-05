import psycopg2
from psycopg2 import sql
import os

# --- Cấu hình PostgreSQL ---
# Cần thay thế các giá trị này bằng thông tin kết nối thực tế của PostgreSQL server
PG_HOST = "localhost"
PG_PORT = "5432"
PG_DATABASE = "face_db"  # Tên database để chứa bảng vector
PG_USER = "admin"
PG_PASSWORD = "Daniel@2410" # Vui lòng thay đổi mật khẩu
COLLECTION_NAME = "face_templates"
VECTOR_DIMENSION = 512  # Kích thước vector cho FaceNet embedding

class PgVectorDBSetup:
    def __init__(self):
        self.conn = None
        # 1. Thiết lập kết nối PostgreSQL
        try:
            self.conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                database=PG_DATABASE,
                user=PG_USER,
                password=PG_PASSWORD
            )
            self.conn.autocommit = True
            print(f"SUCCESS: Đã kết nối đến PostgreSQL DB '{PG_DATABASE}'.")
        except psycopg2.OperationalError as e:
            print(f"ERROR: Không thể kết nối đến PostgreSQL. Vui lòng kiểm tra dịch vụ và thông tin đăng nhập.")
            print(f"Chi tiết: {e}")
            raise RuntimeError("Không thể kết nối PostgreSQL.")
        except Exception as e:
            print(f"ERROR: Xảy ra lỗi không xác định khi kết nối DB: {e}")
            raise RuntimeError("Không thể kết nối PostgreSQL.")
        
    def _execute_sql(self, query: sql.SQL, *args):
        """Hàm hỗ trợ thực thi SQL an toàn."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, args)
                return True
        except psycopg2.Error as e:
            print(f"PostgreSQL ERROR: Lỗi thực thi SQL: {e}")
            return False

    def setup_pgvector(self):
        """Đảm bảo extension pgvector đã được kích hoạt."""
        print(f"INFO: Đang kiểm tra và kích hoạt extension 'vector'...")
        # Lệnh SQL để tạo extension vector
        create_extension_query = sql.SQL("CREATE EXTENSION IF NOT EXISTS vector;")
        if self._execute_sql(create_extension_query):
            print("SUCCESS: Extension 'vector' đã sẵn sàng.")
        else:
            print("FATAL: Không thể tạo extension 'vector'. Đảm bảo pgvector đã được cài đặt trên PostgreSQL server.")
            raise RuntimeError("Cần cài đặt pgvector.")


    def create_face_collection(self):
        """
        Tạo bảng 'face_templates' với cột vector 512 chiều.
        """
        
        # 2. Định nghĩa Schema (Cấu trúc bảng)
        # Sử dụng UUID làm Primary Key và user_id cho mục đích liên kết.
        # face_vector là cột vector 512 chiều.
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- ID chính của vector
                user_id VARCHAR(100) UNIQUE NOT NULL,         -- UUID liên kết với Metadata DB (giống Milvus)
                enrollment_date BIGINT NOT NULL,              -- Timestamp ngày đăng ký
                face_vector vector({dim})                     -- Cột vector 512 chiều
            );
        """).format(
            table_name=sql.Identifier(COLLECTION_NAME),
            dim=sql.Literal(VECTOR_DIMENSION)
        )

        print(f"INFO: Đang tạo hoặc xác nhận bảng '{COLLECTION_NAME}'...")
        if self._execute_sql(create_table_query):
            print(f"SUCCESS: Đã tạo bảng '{COLLECTION_NAME}'.")
        else:
            print("ERROR: Không thể tạo bảng. Vui lòng kiểm tra lại cấu hình.")
            return


        index_query = sql.SQL("""
            CREATE INDEX IF NOT EXISTS face_vector_ivfflat_idx 
            ON {table_name} USING ivfflat (face_vector) WITH (lists = 100);
        """).format(table_name=sql.Identifier(COLLECTION_NAME))
        
        print("INFO: Đang tạo chỉ mục IVFFlat trên cột vector...")
        if self._execute_sql(index_query):
            print("SUCCESS: Đã tạo Index IVFFlat trên trường 'face_vector'.")
        else:
            print("WARNING: Không thể tạo Index. Hiệu suất tìm kiếm có thể bị ảnh hưởng.")

    def close(self):
        """Đóng kết nối database."""
        if self.conn:
            self.conn.close()
            print("INFO: Đã đóng kết nối tới PostgreSQL.")

def main():
    """Chạy quy trình thiết lập database pgvector."""
    print("--- KHỞI TẠO EMBEDDING DATABASE (POSTGRESQL + PGVECTOR) ---")
    setup = None
    try:
        setup = PgVectorDBSetup()
        setup.setup_pgvector()
        setup.create_face_collection()
        print("\nSETUP COMPLETE: Database PostgreSQL + pgvector đã sẵn sàng cho vector khuôn mặt 512 chiều.")
    except RuntimeError:
        print("\nSETUP FAILED: Không thể tiếp tục do lỗi kết nối hoặc cấu hình PostgreSQL.")
    except Exception as e:
        print(f"\nSETUP FAILED: Xảy ra lỗi không xác định: {e}")
    finally:
        if setup:
            setup.close()

if __name__ == "__main__":
    main()
