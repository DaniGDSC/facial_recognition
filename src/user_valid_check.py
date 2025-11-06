import sqlite3

class UserCodeCheckSystem:
    def __init__(self):
        self.metadata_client = sqlite3.connect(
            "/home/un1/projects/facial_recognition/src/database/metadata.db"
        )
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
            return f"THẤT BẠI: Database error - {str(e)}"

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
            return {'found': False, 'message': f'Database error: {str(e)}'}

    def close(self):
        if hasattr(self, 'metadata_client'):
            self.metadata_client.close()