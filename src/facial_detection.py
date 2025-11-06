import os
import cv2
import math
import time
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from facenet_pytorch import MTCNN
from config import (
    CAMERA_INDEX, TARGET_BOX_COLOR, SUCCESS_COLOR, FAILURE_COLOR,
    TARGET_BOX_SIZE_RATIO, MIN_FACE_SIZE_RATIO, DESIRED_FACE_SIZE,
    DESIRED_LEFT_EYE, CONFIDENT_THRESHOLD, DEVICE, 
    REFLECTION_THRESHOLD, EYE_THRESHOLD
)

class FacialDetectorMTCNN:
    def __init__(self):
        self.detector = MTCNN(
            image_size=DESIRED_FACE_SIZE,
            margin=14,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=DEVICE,
        )

    def detect(self, pil_image: Image.Image):
        return self.detector.detect(pil_image, landmarks=True)

class FacialLiveness:
    def __init__(self):
        self.reflection_threshold = REFLECTION_THRESHOLD
        self.eye_threshold = EYE_THRESHOLD

    def check_liveness(self, image: np.ndarray, face_box: np.ndarray, landmarks: Optional[np.ndarray] = None) -> dict:
        # Basic validation
        if image is None or face_box is None or len(face_box) != 4:
            return {'is_live': False, 'confidence': 0.0, 'reason': 'Invalid input'}
        
        x1, y1, x2, y2 = face_box.astype(int)
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return {'is_live': False, 'confidence': 0.0, 'reason': 'Invalid face region'}
        
        # Check reflection patterns
        reflection_score = self._check_reflection(face_region)
        
        # Check eye regions if landmarks available
        eye_score = 0.5  # Default
        if landmarks is not None:
            eye_score = self._check_eyes(image, landmarks)
        
        # Calculate overall confidence
        confidence = (reflection_score * 0.6 + eye_score * 0.4)
        is_live = confidence >= 0.5
        
        return {
            'is_live': is_live,
            'confidence': confidence,
            'scores': {'reflection': reflection_score, 'eye_region': eye_score},
            'reason': 'Live face detected' if is_live else 'Possible spoof detected'
        }
    
    def _check_reflection(self, face_region: np.ndarray) -> float:
        # Convert to LAB color space
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Find bright spots
        threshold = np.percentile(l_channel, 95)
        bright_spots = (l_channel > threshold).astype(np.uint8)
        
        # Analyze bright spot patterns
        contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.7  # No reflections - likely real
        
        # Count large vs small reflections
        large_reflections = sum(1 for c in contours if cv2.contourArea(c) > 100)
        
        if large_reflections > 2:
            return 0.2  # Too many large reflections - likely fake
        else:
            return 0.8  # Normal reflection pattern
    
    def _check_eyes(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        try:
            left_eye = landmarks[0].astype(int)
            right_eye = landmarks[1].astype(int)
            
            eye_scores = []
            for eye_center in [left_eye, right_eye]:
                x, y = eye_center
                # Extract eye region
                eye_region = image[y-15:y+15, x-15:x+15]
                
                if eye_region.size == 0:
                    continue
                
                # Convert to grayscale and detect edges
                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Real eyes have moderate edge density
                score = 0.8 if 0.1 < edge_density < 0.4 else 0.3
                eye_scores.append(score)
            
            return np.mean(eye_scores) if eye_scores else 0.5
            
        except Exception:
            return 0.5

class FacialAlignment:
    def __init__(self):
        self.desired_face_size = DESIRED_FACE_SIZE
        self.desired_left_eye = DESIRED_LEFT_EYE

    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate angle between eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dY, dX))
            
            # Calculate scale
            dist = math.hypot(dX, dY)
            desired_dist = (1.0 - 2 * self.desired_left_eye[0]) * self.desired_face_size
            scale = desired_dist / dist
            
            # Get rotation center
            eyes_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
            
            # Create transformation matrix
            M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
            
            # Calculate translation
            desired_center_x = self.desired_face_size * 0.5
            desired_center_y = self.desired_left_eye[1] * self.desired_face_size
            
            M[0, 2] += desired_center_x - eyes_center[0]
            M[1, 2] += desired_center_y - eyes_center[1]
            
            # Apply transformation
            aligned = cv2.warpAffine(image, M, (self.desired_face_size, self.desired_face_size))
            return aligned
            
        except Exception:
            return None

class LiveCapture:
    def __init__(self, enable_liveness: bool = True, camera_index: int = CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.detector = FacialDetectorMTCNN()
        self.aligner = FacialAlignment()
        self.enable_liveness = enable_liveness
        
        if enable_liveness:
            self.liveness_detector = FacialLiveness()

    def start(self) -> Optional[np.ndarray]:
        if not self.cap.isOpened():
            return None
            
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate target box
        target_size = int(frame_width * TARGET_BOX_SIZE_RATIO)
        center_x, center_y = frame_width // 2, frame_height // 2
        tx1 = center_x - target_size // 2
        ty1 = center_y - target_size // 2
        tx2 = center_x + target_size // 2
        ty2 = center_y + target_size // 2
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Detect face
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            boxes, probs, landmarks = self.detector.detect(pil_image)
            
            status_color = TARGET_BOX_COLOR
            status_message = "Position face in box"
            is_ready = False
            
            # Process detection
            if boxes is not None and len(boxes) == 1 and probs[0] >= CONFIDENT_THRESHOLD:
                face_box = boxes[0].astype(int)
                x1, y1, x2, y2 = face_box
                
                # Check if face is properly positioned
                face_in_box = (x1 >= tx1 and y1 >= ty1 and x2 <= tx2 and y2 <= ty2)
                face_size_ok = (x2 - x1) >= target_size * MIN_FACE_SIZE_RATIO
                
                if face_in_box and face_size_ok:
                    # Run liveness check if enabled
                    if self.enable_liveness:
                        liveness_result = self.liveness_detector.check_liveness(
                            frame, face_box, landmarks[0] if landmarks is not None else None
                        )
                        
                        if liveness_result['is_live']:
                            status_color = SUCCESS_COLOR
                            status_message = "Ready! Press SPACE"
                            is_ready = True
                        else:
                            status_color = FAILURE_COLOR
                            status_message = f"Liveness failed: {liveness_result['reason']}"
                    else:
                        status_color = SUCCESS_COLOR
                        status_message = "Ready! Press SPACE"
                        is_ready = True
                
                # Draw face box
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            
            # Draw target box
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), status_color, 3)
            
            # Draw status
            cv2.putText(frame, status_message, (10, frame_height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, "Press SPACE to capture | ESC to exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Face Capture", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord(' ') and is_ready and landmarks is not None:
                # Capture and align face
                aligned = self.aligner.align_face(frame, landmarks[0])
                if aligned is not None:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return aligned
        
        self.cap.release()
        cv2.destroyAllWindows()
        return None

def main():
    print("Starting face capture...")
    
    live_capture = LiveCapture(enable_liveness=True, camera_index=0)
    captured_face = live_capture.start()
    
    if captured_face is not None:
        print("Face captured successfully!")
        cv2.imshow("Captured Face", captured_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Face capture failed or cancelled")

if __name__ == "__main__":
    main()
