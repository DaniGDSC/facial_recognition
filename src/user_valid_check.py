import logging
import sqlite3
from config import DATABASE_FILE

logger = logging.getLogger(__name__)

class UserCodeCheckSystem:
    def __init__(self):
        self.metadata_client = sqlite3.connect(DATABASE_FILE)
        self.metadata_client.row_factory = sqlite3.Row

    def check_user_code_uniqueness(self, user_code: str) -> str:
        """Check if user code already exists in database"""
        try:
            cursor = self.metadata_client.cursor()
            cursor.execute(
                "SELECT user_code FROM user_profiles WHERE user_code = ?", 
                (user_code,)
            )
            
            existing_user = cursor.fetchone()
            
            if existing_user:
                return f"THẤT BẠI: User code {user_code} already exists"
            else:
                return f"THÀNH CÔNG: User code {user_code} is available"
                
        except Exception as e:
            logger.error(f"Database error during user code check: {e}")
            return "THẤT BẠI: Database error occurred"

    def get_user_info(self, user_code: str) -> dict:
        """Get user information by user code"""
        try:
            cursor = self.metadata_client.cursor()
            cursor.execute("""
                SELECT user_id, user_code, first_name, last_name, account_status
                FROM user_profiles 
                WHERE user_code = ?
            """, (user_code,))
            
            user = cursor.fetchone()
            
            if user:
                return {
                    'found': True,
                    'user_id': user['user_id'],
                    'user_code': user['user_code'],
                    'first_name': user['first_name'],
                    'last_name': user['last_name'],
                    'full_name': f"{user['first_name']} {user['last_name']}",
                    'account_status': user['account_status']
                }
            else:
                return {'found': False, 'message': 'User not found'}
                
        except Exception as e:
            logger.error(f"Database error during user info lookup: {e}")
            return {'found': False, 'message': 'Database error occurred'}

    def close(self):
        if hasattr(self, 'metadata_client'):
            self.metadata_client.close()