import os
import time
import uuid
import logging
import numpy as np
from typing import Dict, Any
import sqlite3
import psycopg2
import cv2

from facial_detection import LiveCapture
from facial_recognition import FacialRecognizer
from user_valid_check import UserCodeCheckSystem
from db import get_metadata_connection, get_vector_connection
from crypto import encrypt_image, compute_embedding_hmac

logger = logging.getLogger(__name__)

def _get_optional_key(env_var: str) -> bytes | None:
    """Load an optional hex-encoded key from env, return None if not set."""
    raw = os.environ.get(env_var)
    if not raw:
        return None
    return bytes.fromhex(raw)


class UserEnrollmentSystem:
    def __init__(self):
        logger.info("Initializing enrollment system...")

        # Initialize components
        self.user_checker = UserCodeCheckSystem()
        self.recognizer = FacialRecognizer(device='cpu')

        # Database connections
        self.metadata_conn = get_metadata_connection()
        self.vector_conn = get_vector_connection()
        logger.info("Enrollment system ready")

    def enroll_new_user(self, user_code: str, first_name: str, last_name: str,
                       enable_liveness: bool = True) -> Dict[str, Any]:
        """Complete user enrollment pipeline with proper validation"""

        logger.info("Starting enrollment for user_code=%s, name=%s %s",
                     user_code, first_name, last_name)

        try:
            # STEP 1: Check uniqueness
            logger.info("STEP 1: Checking user code uniqueness...")
            uniqueness_result = self.user_checker.check_user_code_uniqueness(user_code)
            logger.info("Validation result: %s", uniqueness_result)

            if "THẤT BẠI" in uniqueness_result or "already exists" in uniqueness_result:
                logger.warning("Enrollment blocked: user code %s already exists", user_code)
                return {
                    'success': False, 'user_id': None,
                    'message': f"User code {user_code} already exists in database",
                    'similarity_score': 0.0, 'liveness_passed': False
                }

            logger.info("STEP 1 passed: user code is unique")

            # STEP 2: Capture face
            logger.info("STEP 2: Starting face capture...")
            live_capture = LiveCapture(enable_liveness=enable_liveness, camera_index=0)
            captured_face = live_capture.start()

            if captured_face is None:
                return {
                    'success': False, 'user_id': None,
                    'message': "Face capture failed or cancelled",
                    'similarity_score': 0.0, 'liveness_passed': False
                }

            logger.info("STEP 2 passed: face captured")

            # STEP 3: Extract features
            logger.info("STEP 3: Extracting facial features...")
            embeddings = self.recognizer.process_and_extract_from_array(captured_face)
            if embeddings is None:
                return {
                    'success': False, 'user_id': None,
                    'message': "Feature extraction failed",
                    'similarity_score': 0.0, 'liveness_passed': False
                }

            embedding_vector = embeddings[0].cpu().numpy()
            logger.info("STEP 3 passed: generated %d-D embedding", len(embedding_vector))

            # STEP 4: Store in databases
            logger.info("STEP 4: Storing user data...")

            user_id = str(uuid.uuid4())
            current_timestamp = int(time.time())

            success_metadata = self._store_user_metadata(
                user_id, user_code, first_name, last_name, current_timestamp
            )
            if not success_metadata:
                return {
                    'success': False, 'user_id': None,
                    'message': "Failed to store user metadata",
                    'similarity_score': 0.0, 'liveness_passed': False
                }

            success_embedding = self._store_user_embedding(
                user_id, user_code, embedding_vector, current_timestamp
            )
            if not success_embedding:
                self._rollback_metadata(user_id)
                return {
                    'success': False, 'user_id': None,
                    'message': "Failed to store face embedding",
                    'similarity_score': 0.0, 'liveness_passed': False
                }

            logger.info("STEP 4 passed: user data stored")

            # STEP 5: Finalize
            self._log_enrollment_event(user_id, True, 1.0, "127.0.0.1")
            self._save_enrollment_image(captured_face, user_id)

            logger.info("Enrollment completed: user_id=%s, user_code=%s", user_id, user_code)

            return {
                'success': True, 'user_id': user_id,
                'message': "User enrolled successfully",
                'similarity_score': 1.0, 'liveness_passed': enable_liveness
            }

        except Exception as e:
            logger.exception("Enrollment failed for user_code=%s", user_code)
            return {
                'success': False, 'user_id': None,
                'message': f"Enrollment error: {str(e)}",
                'similarity_score': 0.0, 'liveness_passed': False
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
            logger.error("Metadata storage failed: %s", e)
            return False

    def _store_user_embedding(self, user_id: str, user_code: str,
                            embedding: np.ndarray, timestamp: int) -> bool:
        """Store embedding in PostgreSQL with optional HMAC integrity tag."""
        try:
            cursor = self.vector_conn.cursor()
            embedding_list = embedding.tolist()
            vector_str = '[' + ','.join(map(str, embedding_list)) + ']'

            # Compute HMAC if key is available
            hmac_key = _get_optional_key("EMBEDDING_HMAC_KEY")
            hmac_tag = compute_embedding_hmac(embedding_list, hmac_key) if hmac_key else None

            if hmac_tag:
                cursor.execute("""
                    INSERT INTO face_templates (user_id, enrollment_date, face_vector, hmac_tag)
                    VALUES (%s, %s, %s::vector, %s)
                """, (user_code, timestamp, vector_str, hmac_tag))
                logger.info("Stored embedding with HMAC integrity tag")
            else:
                cursor.execute("""
                    INSERT INTO face_templates (user_id, enrollment_date, face_vector)
                    VALUES (%s, %s, %s::vector)
                """, (user_code, timestamp, vector_str))
                logger.warning("Stored embedding WITHOUT HMAC — set EMBEDDING_HMAC_KEY for integrity protection")

            return True
        except psycopg2.Error as e:
            logger.error("Embedding storage failed: %s", e)
            return False

    def _rollback_metadata(self, user_id: str):
        """Remove metadata on failure"""
        try:
            cursor = self.metadata_conn.cursor()
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            self.metadata_conn.commit()
            logger.info("Metadata rollback completed for user_id=%s", user_id)
        except Exception as e:
            logger.error("Rollback failed for user_id=%s: %s", user_id, e)

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
        except Exception as e:
            logger.error("Failed to log enrollment event for user %s: %s", user_id, e)

    def _save_enrollment_image(self, face_image: np.ndarray, user_id: str):
        """Save enrollment image, encrypted at rest if key is available."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.environ.get(
                "ENROLLED_FACES_DIR",
                os.path.join(base_dir, "data", "enrolled_faces")
            )
            os.makedirs(output_dir, exist_ok=True)

            enc_key = _get_optional_key("BIOMETRIC_ENCRYPTION_KEY")
            _, img_bytes = cv2.imencode(".jpg", face_image)
            raw_bytes = img_bytes.tobytes()

            if enc_key:
                encrypted = encrypt_image(raw_bytes, enc_key)
                filename = f"enrolled_{user_id}_{int(time.time())}.jpg.enc"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(encrypted)
                os.chmod(filepath, 0o600)
                logger.info("Saved encrypted enrollment image: %s", filepath)
            else:
                filename = f"enrolled_{user_id}_{int(time.time())}.jpg"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(raw_bytes)
                os.chmod(filepath, 0o600)
                logger.warning("Saved UNENCRYPTED enrollment image — set BIOMETRIC_ENCRYPTION_KEY for encryption")

        except Exception as e:
            logger.error("Failed to save enrollment image for user %s: %s", user_id, e)

    def close(self):
        """Close connections"""
        if hasattr(self, 'metadata_conn'):
            self.metadata_conn.close()
        if hasattr(self, 'vector_conn'):
            self.vector_conn.close()
        if hasattr(self, 'user_checker'):
            self.user_checker.close()
        logger.info("Enrollment system closed")
