import os
import time
import uuid
import numpy as np
from typing import Optional, Dict, Any
import sqlite3
import psycopg2
import cv2

from facial_detection import LiveCapture, FacialLiveness
from facial_recognition import FacialRecognizer
from user_valid_check import UserCodeCheckSystem

class UserEnrollmentSystem:
    def __init__(self):
        print("🔧 Initializing enrollment system...")
        
        # Initialize components
        self.user_checker = UserCodeCheckSystem()
        self.recognizer = FacialRecognizer(device='cpu')
        
        # Database connections
        self.setup_databases()
        print("✅ Enrollment system ready")
        
    def setup_databases(self):
        """Initialize database connections"""
        try:
            # SQLite for metadata
            self.metadata_conn = sqlite3.connect("/home/un1/projects/facial_recognition/src/database/metadata.db")
            self.metadata_conn.row_factory = sqlite3.Row
            
            # PostgreSQL for embeddings
            self.vector_conn = psycopg2.connect(
                host="localhost",
                port="5432", 
                database="face_db",
                user="admin",
                password="Daniel@2410"
            )
            self.vector_conn.autocommit = True
            
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}")
    
    def enroll_new_user(self, user_code: str, first_name: str, last_name: str, 
                       enable_liveness: bool = True) -> Dict[str, Any]:
        """Complete user enrollment pipeline with proper validation"""
        
        print(f"\n{'='*50}")
        print(f"  STARTING USER ENROLLMENT")
        print(f"  User Code: {user_code}")
        print(f"  Name: {first_name} {last_name}")
        print(f"{'='*50}")
        
        try:
            print("🔍 STEP 1: Checking user code uniqueness...")
            
            uniqueness_result = self.user_checker.check_user_code_uniqueness(user_code)
            print(f"   Validation result: {uniqueness_result}")
            
            # FIXED: Check for failure message properly
            if "THẤT BẠI" in uniqueness_result or "already exists" in uniqueness_result:
                print("❌ ENROLLMENT BLOCKED: User code already exists!")
                return {
                    'success': False,
                    'user_id': None,
                    'message': f"User code {user_code} already exists in database",
                    'similarity_score': 0.0,
                    'liveness_passed': False
                }
            
            print("✅ STEP 1: User code is unique and available")
            
            # STEP 2: Capture face
            print("\n📷 STEP 2: Starting face capture...")
            print("   Position face in camera and press SPACE when ready")
            
            live_capture = LiveCapture(enable_liveness=enable_liveness, camera_index=0)
            captured_face = live_capture.start()
            
            if captured_face is None:
                return {
                    'success': False,
                    'user_id': None,
                    'message': "Face capture failed or cancelled",
                    'similarity_score': 0.0,
                    'liveness_passed': False
                }
            
            print("✅ STEP 2: Face captured successfully")
            
            # STEP 3: Extract features
            print("\n🧠 STEP 3: Extracting facial features...")
            
            embeddings = self.recognizer.process_and_extract_from_array(captured_face)
            if embeddings is None:
                return {
                    'success': False,
                    'user_id': None,
                    'message': "Feature extraction failed",
                    'similarity_score': 0.0,
                    'liveness_passed': False
                }
            
            embedding_vector = embeddings[0].cpu().numpy()
            print(f"✅ STEP 3: Generated {len(embedding_vector)}-D embedding")
            
            # STEP 4: Store in databases
            print("\n💾 STEP 4: Storing user data...")
            
            user_id = str(uuid.uuid4())
            current_timestamp = int(time.time())
            
            # Store metadata
            success_metadata = self._store_user_metadata(
                user_id, user_code, first_name, last_name, current_timestamp
            )
            
            if not success_metadata:
                return {
                    'success': False,
                    'user_id': None,
                    'message': "Failed to store user metadata",
                    'similarity_score': 0.0,
                    'liveness_passed': False
                }
            
            # Store embedding
            success_embedding = self._store_user_embedding(
                user_id, user_code, embedding_vector, current_timestamp
            )
            
            if not success_embedding:
                self._rollback_metadata(user_id)
                return {
                    'success': False,
                    'user_id': None,
                    'message': "Failed to store face embedding",
                    'similarity_score': 0.0,
                    'liveness_passed': False
                }
            
            print("✅ STEP 4: User data stored successfully")
            
            # STEP 5: Finalize
            self._log_enrollment_event(user_id, True, 1.0, "127.0.0.1")
            self._save_enrollment_image(captured_face, user_id)
            
            print(f"\n🎉 ENROLLMENT COMPLETED SUCCESSFULLY!")
            print(f"   User ID: {user_id}")
            print(f"   User Code: {user_code}")
            print(f"   Name: {first_name} {last_name}")
            
            return {
                'success': True,
                'user_id': user_id,
                'message': "User enrolled successfully",
                'similarity_score': 1.0,
                'liveness_passed': enable_liveness
            }
            
        except Exception as e:
            print(f"💥 ERROR: Enrollment failed: {e}")
            return {
                'success': False,
                'user_id': None,
                'message': f"Enrollment error: {str(e)}",
                'similarity_score': 0.0,
                'liveness_passed': False
            }
    
    def _store_user_metadata(self, user_id: str, user_code: str, first_name: str, 
                           last_name: str, timestamp: int) -> bool:
        """Store user profile in SQLite"""
        try:
            cursor = self.metadata_conn.cursor()
            cursor.execute("""
                INSERT INTO user_profiles 
                (user_id, user_code, first_name, last_name, account_status, enrollment_date, biometric_consent_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, user_code, first_name, last_name, 'ACTIVE', timestamp, timestamp))
            
            self.metadata_conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Metadata storage failed: {e}")
            return False
    
    def _store_user_embedding(self, user_id: str, user_code: str, 
                            embedding: np.ndarray, timestamp: int) -> bool:
        """Store embedding in PostgreSQL"""
        try:
            cursor = self.vector_conn.cursor()
            vector_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
            
            cursor.execute("""
                INSERT INTO face_templates (user_id, enrollment_date, face_vector)
                VALUES (%s, %s, %s::vector)
            """, (user_code, timestamp, vector_str))
            
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Embedding storage failed: {e}")
            return False
    
    def _rollback_metadata(self, user_id: str):
        """Remove metadata on failure"""
        try:
            cursor = self.metadata_conn.cursor()
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            self.metadata_conn.commit()
            print("🔄 Metadata rollback completed")
        except Exception as e:
            print(f"⚠️  Rollback failed: {e}")
    
    def _log_enrollment_event(self, user_id: str, success: bool, score: float, ip: str):
        """Log enrollment event"""
        try:
            cursor = self.metadata_conn.cursor()
            cursor.execute("""
                INSERT INTO biometric_audit_log 
                (user_id, timestamp, event_type, similarity_score, spoof_detected, device_ip)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, int(time.time()), 'Enroll', score, 0, ip))
            self.metadata_conn.commit()
        except Exception:
            pass
    
    def _save_enrollment_image(self, face_image: np.ndarray, user_id: str):
        """Save enrollment image"""
        try:
            output_dir = "/home/un1/projects/facial_recognition/data/enrolled_faces"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"enrolled_{user_id}_{int(time.time())}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, face_image)
        except Exception:
            pass
    
    def close(self):
        """Close connections"""
        if hasattr(self, 'metadata_conn'):
            self.metadata_conn.close()
        if hasattr(self, 'vector_conn'):
            self.vector_conn.close()
        if hasattr(self, 'user_checker'):
            self.user_checker.close()