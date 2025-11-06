import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import sqlite3
import psycopg2
import cv2
from PIL import Image
import faiss

from facial_detection import LiveCapture, FacialLiveness, FacialDetectorMTCNN
from facial_recognition import FacialRecognizer

class FacialRecognitionSystem:
    def __init__(self, similarity_threshold: float = 0.6, use_faiss: bool = True):
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss
        self.faiss_index = None
        self.user_code_mapping = []
        
        # Initialize components
        self.recognizer = FacialRecognizer(device='cpu')
        self.detector = FacialDetectorMTCNN()
        self.liveness_detector = FacialLiveness()
        
        # Setup databases and load users
        self.setup_databases()
        self.enrolled_users = self.load_enrolled_users()
        
        # Build FAISS index
        if self.use_faiss:
            self._build_faiss_index()
        
    def setup_databases(self):
        self.metadata_conn = sqlite3.connect("/home/un1/projects/facial_recognition/src/database/metadata.db")
        self.metadata_conn.row_factory = sqlite3.Row
        
        self.vector_conn = psycopg2.connect(
            host="localhost", port="5432", database="face_db",
            user="admin", password="Daniel@2410"
        )
        self.vector_conn.autocommit = True
    
    def load_enrolled_users(self) -> Dict[str, Dict]:
        try:
            cursor = self.metadata_conn.cursor()
            cursor.execute("""
                SELECT user_id, user_code, first_name, last_name, account_status, enrollment_date
                FROM user_profiles WHERE account_status = 'ACTIVE'
            """)
            
            users = {}
            for row in cursor.fetchall():
                users[row['user_code']] = {
                    'user_id': row['user_id'],
                    'user_code': row['user_code'],
                    'first_name': row['first_name'],
                    'last_name': row['last_name'],
                    'full_name': f"{row['first_name']} {row['last_name']}",
                    'account_status': row['account_status'],
                    'enrollment_date': row['enrollment_date']
                }
            
            return users
        except Exception:
            return {}
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        try:
            cursor = self.vector_conn.cursor()
            cursor.execute("SELECT user_id, face_vector FROM face_templates")
            
            embeddings = []
            for row in cursor.fetchall():
                user_code = row[0]
                vector_str = row[1]
                vector_values = vector_str.strip('[]').split(',')
                embedding = np.array([float(x.strip()) for x in vector_values])
                embeddings.append((user_code, embedding))
            
            return embeddings
        except Exception:
            return []
    
    def _build_faiss_index(self):
        """Build FAISS index for O(log n) similarity search"""
        try:
            all_embeddings = self.get_all_embeddings()
            if not all_embeddings:
                return
            
            # Prepare embeddings matrix
            embeddings_matrix = []
            self.user_code_mapping = []
            
            for user_code, embedding in all_embeddings:
                embeddings_matrix.append(embedding)
                self.user_code_mapping.append(user_code)
            
            embeddings_matrix = np.array(embeddings_matrix).astype('float32')
            n_users, embedding_dim = embeddings_matrix.shape
            
            # Choose index type based on database size
            if n_users < 1000:
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Exact search
            else:
                # Approximate search for large datasets
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, min(100, n_users//10))
                self.faiss_index.train(embeddings_matrix)
            
            # Normalize for cosine similarity and add to index
            faiss.normalize_L2(embeddings_matrix)
            self.faiss_index.add(embeddings_matrix)
            
        except Exception:
            self.faiss_index = None
    
    def fast_similarity_search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Fast O(log n) similarity search using FAISS"""
        if self.faiss_index is None:
            return self._linear_search(query_vector, top_k)
        
        try:
            # Normalize query vector
            query_vector = query_vector.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.faiss_index.search(query_vector, min(top_k, len(self.user_code_mapping)))
            
            # Convert results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != -1 and idx < len(self.user_code_mapping):
                    user_code = self.user_code_mapping[idx]
                    similarity_score = (similarity + 1) / 2  # Convert to 0-1 range
                    results.append((user_code, float(similarity_score)))
            
            return results
        except Exception:
            return self._linear_search(query_vector, top_k)
    
    def _linear_search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Fallback linear search"""
        all_embeddings = self.get_all_embeddings()
        similarities = [(user_code, self.calculate_similarity(query_vector, embedding)) 
                       for user_code, embedding in all_embeddings]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float((similarity + 1) / 2)
    
    def detect_and_extract_face(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            pil_image = Image.fromarray(rgb_image)
            
            boxes, probs, landmarks = self.detector.detect(pil_image)
            
            if boxes is None or len(boxes) == 0:
                return None, None, None
            
            best_idx = np.argmax(probs) if probs is not None else 0
            face_box = boxes[best_idx].astype(int)
            face_landmarks = landmarks[best_idx] if landmarks is not None else None
            
            x1, y1, x2, y2 = face_box
            margin = 10
            h, w = rgb_image.shape[:2]
            
            x1_exp = max(0, x1 - margin)
            y1_exp = max(0, y1 - margin) 
            x2_exp = min(w, x2 + margin)
            y2_exp = min(h, y2 + margin)
            
            face_region = rgb_image[y1_exp:y2_exp, x1_exp:x2_exp]
            
            return face_region, np.array([x1, y1, x2, y2]), face_landmarks
        except Exception:
            return None, None, None
    
    def check_liveness(self, image: np.ndarray) -> Dict[str, Any]:
        face_region, face_box, landmarks = self.detect_and_extract_face(image)
        
        if face_region is None or face_box is None:
            return {'is_live': False, 'confidence': 0.0, 'reason': 'No face detected'}
        
        return self.liveness_detector.check_liveness(image, face_box, landmarks)
    
    def recognize_face(self, face_image: np.ndarray, enable_liveness: bool = True, 
                      top_k: int = 5) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Liveness check
            liveness_result = {'is_live': True, 'confidence': 1.0}
            
            if enable_liveness:
                liveness_result = self.check_liveness(face_image)
                if not liveness_result['is_live']:
                    return {
                        'success': False, 'recognized': False, 'user_info': None,
                        'matches': [], 'liveness_result': liveness_result,
                        'similarity_score': 0.0, 'processing_time': time.time() - start_time,
                        'message': f'Liveness check failed: {liveness_result.get("reason")}'
                    }
            
            # Extract features
            query_embedding = self.recognizer.process_and_extract_from_array(face_image)
            if query_embedding is None:
                return {
                    'success': False, 'recognized': False, 'user_info': None,
                    'matches': [], 'liveness_result': liveness_result,
                    'similarity_score': 0.0, 'processing_time': time.time() - start_time,
                    'message': 'Feature extraction failed'
                }
            
            query_vector = query_embedding[0].cpu().numpy()
            
            # Fast similarity search with FAISS
            search_start = time.time()
            
            if self.use_faiss and self.faiss_index is not None:
                similar_users = self.fast_similarity_search(query_vector, top_k)
                search_method = "FAISS"
            else:
                similar_users = self._linear_search(query_vector, top_k)
                search_method = "Linear"
            
            search_time = time.time() - search_start
            
            if not similar_users:
                return {
                    'success': True, 'recognized': False, 'user_info': None,
                    'matches': [], 'liveness_result': liveness_result,
                    'similarity_score': 0.0, 'processing_time': time.time() - start_time,
                    'message': 'No enrolled users'
                }
            
            # Get top matches
            top_matches = []
            for i, (user_code, similarity) in enumerate(similar_users):
                if user_code in self.enrolled_users:
                    user_info = self.enrolled_users[user_code].copy()
                    user_info['similarity_score'] = similarity
                    user_info['rank'] = i + 1
                    top_matches.append(user_info)
            
            # Check best match
            best_match = top_matches[0] if top_matches else None
            best_similarity = best_match['similarity_score'] if best_match else 0.0
            recognized = best_similarity >= self.similarity_threshold
            
            # Log event
            self._log_event('Recognition', best_match['user_code'] if best_match else None,
                          recognized, best_similarity, liveness_result['is_live'])
            
            processing_time = time.time() - start_time
            message = f"User recognized: {best_match['full_name']}" if recognized else "No match found"
            
            return {
                'success': True,
                'recognized': recognized,
                'user_info': best_match,
                'matches': top_matches,
                'liveness_result': liveness_result,
                'similarity_score': best_similarity,
                'processing_time': processing_time,
                'search_time': search_time,
                'search_method': search_method,
                'database_size': len(self.enrolled_users),
                'message': message
            }
            
        except Exception as e:
            return {
                'success': False, 'recognized': False, 'user_info': None,
                'matches': [], 'liveness_result': {'is_live': False, 'confidence': 0.0},
                'similarity_score': 0.0, 'processing_time': time.time() - start_time,
                'message': f'Recognition failed: {str(e)}'
            }
    
    def recognize_from_camera(self, enable_liveness: bool = True, top_k: int = 5) -> Dict[str, Any]:
        try:
            live_capture = LiveCapture(enable_liveness=False, camera_index=0)
            captured_face = live_capture.start()
            
            if captured_face is None:
                return {'success': False, 'message': 'Face capture failed'}
            
            return self.recognize_face(captured_face, enable_liveness, top_k)
        except Exception as e:
            return {'success': False, 'message': f'Camera error: {str(e)}'}
    
    def _log_event(self, event_type: str, user_code: str, success: bool, 
                   similarity: float, liveness_passed: bool):
        try:
            cursor = self.metadata_conn.cursor()
            cursor.execute("""
                INSERT INTO biometric_audit_log 
                (user_code, timestamp, event_type, similarity_score, spoof_detected, device_ip)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_code, int(time.time()), 
                  f'{event_type}_{"Success" if success else "Failed"}',
                  similarity, 0 if liveness_passed else 1, "127.0.0.1"))
            self.metadata_conn.commit()
        except Exception:
            pass
    
    def close(self):
        if hasattr(self, 'metadata_conn'):
            self.metadata_conn.close()
        if hasattr(self, 'vector_conn'):
            self.vector_conn.close()

if __name__ == "__main__":
    # Test both FAISS and linear search
    system = FacialRecognitionSystem(similarity_threshold=0.6, use_faiss=True)
    
    test_image = cv2.imread("/home/un1/projects/facial_recognition/data/captured_faces/aligned_face_20251105_113243.jpg")
    if test_image is not None:
        result = system.recognize_face(test_image, enable_liveness=False)
        print(f"Result: {result['message']}")
        print(f"Method: {result.get('search_method', 'Unknown')}")
        print(f"Time: {result['processing_time']:.3f}s")
        print(f"Search: {result.get('search_time', 0):.4f}s")
    
    system.close()