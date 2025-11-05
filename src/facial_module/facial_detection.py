import os
import cv2
import math
import time
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import sys
import argparse
from facenet_pytorch import MTCNN
from config import (
    CAMERA_INDEX, TARGET_BOX_COLOR, SUCCESS_COLOR, FAILURE_COLOR,
    TARGET_BOX_SIZE_RATIO, MIN_FACE_SIZE_RATIO, DESIRED_FACE_SIZE,
    DESIRED_LEFT_EYE, CONFIDENT_THRESHOLD, DEVICE, 
    REFLECTION_THRESHOLD, EYE_THRESHOLD, MIN_CONFIDENCE
)

class FacialDetectorMTCNN:
    def __init__(self, detection_scale: float = 0.75):
        self.detector = MTCNN(
            image_size=DESIRED_FACE_SIZE,
            margin=14,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=DEVICE,
        )
        self.detection_scale = detection_scale

    def detect(self, pil_image: Image.Image):
        original_size = pil_image.size

        if self.detection_scale != 1.0:
            new_size = (
                int(original_size[0] * self.detection_scale),
                int(original_size[1] * self.detection_scale),
            )
            pil_image = pil_image.resize(new_size, Image.BILINEAR)
            boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
            if boxes is not None:
                scale_factor = 1 / self.detection_scale
                boxes = boxes * scale_factor
                if landmarks is not None:
                    landmarks = landmarks * scale_factor
            return boxes, probs, landmarks
        else:
            return self.detector.detect(pil_image, landmarks=True)

class FacialLiveness:
    def __init__(self, reflection_threshold: float = REFLECTION_THRESHOLD,
                 eye_threshold: float = EYE_THRESHOLD):
        self.reflection_threshold = reflection_threshold
        self.eye_threshold = eye_threshold

    def check_liveness(self, image_bgr: np.ndarray, face_box: np.ndarray,
                       landmarks: Optional[np.ndarray] = None) -> dict:
        if not self._validate_inputs(image_bgr, face_box):
            return self._create_result(False, 0.0, {}, "Invalid input")
        x1, y1, x2, y2 = face_box.astype(int)
        if not self._validate_box_coords(x1, y1, x2, y2, image_bgr):
            return self._create_result(False, 0.0, {}, "Face box outside image bounds")
        margin = 10
        x1_m, y1_m = max(0, x1 - margin), max(0, y1 - margin)
        x2_m, y2_m = min(image_bgr.shape[1], x2 + margin), min(image_bgr.shape[0], y2 + margin)
        face_region = image_bgr[y1_m:y2_m, x1_m:x2_m]
        if face_region.size == 0:
            return self._create_result(False, 0.0, {}, "Invalid face region")
        reflection_score = self._check_reflection_pattern(face_region)
        if landmarks is None:
            print("WARNING: No landmarks detected - eye region check skipped")
            eye_score, weights = 0.5, {'reflection': 1.0, 'eye_region': 0.0}
        else:
            eye_score = self._check_eye_region(image_bgr, landmarks)
            weights = {'reflection': 0.55, 'eye_region': 0.45}
        scores = {'reflection': reflection_score, 'eye_region': eye_score}
        confidence = sum(scores[k] * weights[k] for k in scores.keys())
        min_confidence = 0.6 if landmarks is None else 0.5
        is_live = (reflection_score >= self.reflection_threshold and
                   eye_score >= self.eye_threshold and
                   confidence >= min_confidence)
        reason = self._generate_reason(is_live, scores)
        return self._create_result(is_live, confidence, scores, reason)

    def _validate_inputs(self, image_bgr, face_box):
        return (image_bgr is not None and isinstance(image_bgr, np.ndarray) and
                image_bgr.ndim == 3 and image_bgr.shape[2] == 3 and
                face_box is not None and isinstance(face_box, np.ndarray) and
                len(face_box) == 4)

    def _validate_box_coords(self, x1, y1, x2, y2, image_bgr):
        return (x1 >= 0 and y1 >= 0 and x2 <= image_bgr.shape[1] and y2 <= image_bgr.shape[0] and
                x1 < x2 and y1 < y2)

    def _check_reflection_pattern(self, face_region: np.ndarray) -> float:
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        threshold = np.percentile(l_channel, 95)
        bright_spots = (l_channel > threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_spots)
        if num_labels <= 1:
            return 0.7
        areas = stats[1:, cv2.CC_STAT_AREA]
        SMALL_REFLECTION_MAX, LARGE_REFLECTION_MIN = 50, 200
        small_reflections = np.sum(areas < SMALL_REFLECTION_MAX)
        large_reflections = np.sum(areas > LARGE_REFLECTION_MIN)
        if large_reflections > 2:
            return 0.3
        elif small_reflections > 3:
            return 0.9
        else:
            return 0.6

    def _check_eye_region(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> float:
        try:
            left_eye, right_eye = landmarks[0].astype(int), landmarks[1].astype(int)
            eye_size, scores = 30, []
            for eye_center in [left_eye, right_eye]:
                x, y = eye_center
                x1, y1 = max(0, x - eye_size // 2), max(0, y - eye_size // 2)
                x2, y2 = min(image_bgr.shape[1], x + eye_size // 2), min(image_bgr.shape[0], y + eye_size // 2)
                eye_region = image_bgr[y1:y2, x1:x2]
                if eye_region.size == 0 or eye_region.shape[0] < 20 or eye_region.shape[1] < 20:
                    continue
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_eye, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                scores.append(0.9 if 0.05 < edge_density < 0.3 else 0.5)
            return np.mean(scores) if scores else 0.5
        except Exception as e:
            print(f"WARNING: Eye region check failed: {e}")
            return 0.5

    def _generate_reason(self, is_live: bool, scores: dict) -> str:
        if is_live:
            return "Face appears to be live (passed all checks)"
        failures = []
        if scores['reflection'] < self.reflection_threshold:
            failures.append(f"suspicious reflection pattern ({scores['reflection']:.2f})")
        if scores['eye_region'] < self.eye_threshold:
            failures.append(f"unusual eye region ({scores['eye_region']:.2f})")
        return f"Possible spoof: {', '.join(failures)}" if failures else "Low overall confidence score"

    @staticmethod
    def _create_result(is_live: bool, confidence: float, scores: dict, reason: str) -> dict:
        return {'is_live': is_live, 'confidence': confidence, 'scores': scores, 'reason': reason}

    def get_liveness_score(self, image_bgr: np.ndarray, face_box: np.ndarray,
                          landmarks: Optional[np.ndarray] = None) -> float:
        return self.check_liveness(image_bgr, face_box, landmarks)['confidence']

class FacialAlignment:
    def __init__(self):
        self.desired_face_size = DESIRED_FACE_SIZE
        self.desired_left_eye = DESIRED_LEFT_EYE

    def align_face(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        try:
            left_eye, right_eye = landmarks[0], landmarks[1]
            angle, scale, eyes_center = self._calculate_alignment_params(left_eye, right_eye)
            M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
            desired_left_eye_x = self.desired_left_eye[0] * self.desired_face_size
            desired_right_eye_x = (1.0 - self.desired_left_eye[0]) * self.desired_face_size
            desired_eyes_center = (
                (desired_left_eye_x + desired_right_eye_x) * 0.5,
                self.desired_left_eye[1] * self.desired_face_size,
            )
            M[0, 2] += desired_eyes_center[0] - eyes_center[0]
            M[1, 2] += desired_eyes_center[1] - eyes_center[1]
            aligned = cv2.warpAffine(
                image_bgr, M, (self.desired_face_size, self.desired_face_size),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned
        except Exception as e:
            print(f"Face alignment failed: {e}")
            return None

    def _calculate_alignment_params(self, left_eye: np.ndarray, right_eye: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        left_eye, right_eye = np.asarray(left_eye, dtype=np.float32), np.asarray(right_eye, dtype=np.float32)
        dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dY, dX))
        dist = math.hypot(dX, dY)
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_size
        scale = desired_dist / max(dist, 1e-6)
        eyes_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
        return angle, scale, eyes_center

class LiveCapture:
    ALLOWED_CAMERAS = [0, 1]
    def __init__(self, enable_liveness: bool = True, camera_index: int = CAMERA_INDEX):
        if camera_index not in self.ALLOWED_CAMERAS:
            raise ValueError(f"Camera index {camera_index} is not allowed. Allowed indices: {self.ALLOWED_CAMERAS}")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.detector = FacialDetectorMTCNN()
        self.aligner = FacialAlignment()
        self.enable_liveness = enable_liveness
        if enable_liveness:
            self.liveness_detector = FacialLiveness(reflection_threshold=0.4, eye_threshold=0.5)

    def start(self) -> Optional[np.ndarray]:
        cap = self.cap
        if not cap.isOpened():
            print("Cannot open camera, exiting...")
            return None
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_size = int(frame_width * TARGET_BOX_SIZE_RATIO)
        center_x, center_y = frame_width // 2, frame_height // 2
        tx1, ty1, tx2, ty2 = self._get_target_box(center_x, center_y, target_size)
        print(f"INFO: Camera activated. Standard box: {target_size}x{target_size} pixels.")
        if self.enable_liveness:
            print(f"INFO: Liveness detection enabled.")
        captured_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error! Frame capture failed, exiting...")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            boxes, probs, landmarks = self.detector.detect(pil)
            status_color, status_message = TARGET_BOX_COLOR, "No face detected."
            is_aligned, face_box, face_landmarks, liveness_result = False, None, None, None
            if boxes is not None and len(boxes) == 1 and probs is not None and probs[0] is not None and probs[0] >= CONFIDENT_THRESHOLD:
                face_box = boxes[0].astype(int)
                x1, y1, x2, y2 = face_box
                w, h = x2 - x1, y2 - y1
                margin = 5
                target_size_now = tx2 - tx1
                is_size_ok = (w >= target_size_now * MIN_FACE_SIZE_RATIO and h >= target_size_now * MIN_FACE_SIZE_RATIO)
                is_position_ok = (x1 >= tx1 + margin and y1 >= ty1 + margin and x2 <= tx2 - margin and y2 <= ty2 - margin)
                if self.enable_liveness and is_size_ok and is_position_ok:
                    liveness_result = self.liveness_detector.check_liveness(
                        frame, face_box, landmarks[0] if landmarks is not None else None
                    )
                    if not liveness_result['is_live']:
                        is_aligned, status_color = False, FAILURE_COLOR
                        status_message = f"⚠ {liveness_result['reason']}"
                    else:
                        is_aligned, status_color = True, SUCCESS_COLOR
                        status_message = f"✓ Live face detected ({liveness_result['confidence']:.2f}). Press SPACE."
                elif is_size_ok and is_position_ok:
                    is_aligned, status_color = True, SUCCESS_COLOR
                    status_message = "Aligned. Press SPACE to capture."
                else:
                    status_color = FAILURE_COLOR if not is_position_ok else TARGET_BOX_COLOR
                    status_message = "Move closer to the camera." if not is_size_ok else "Center your face in the box."
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                if liveness_result and self.enable_liveness:
                    self._draw_liveness_info(frame, liveness_result, x2 + 10, y1)
            elif boxes is not None and len(boxes) > 1:
                status_color, status_message = FAILURE_COLOR, "Multiple faces detected."
            if landmarks is not None and len(landmarks) == 1:
                face_landmarks = landmarks[0]
                for (px, py) in face_landmarks.astype(int):
                    cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), status_color, 3)
            cv2.putText(frame, "Face Alignment Guide", (tx1, max(20, ty1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TARGET_BOX_COLOR, 2)
            cv2.putText(frame, status_message, (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, "Press SPACE to capture | ESC to exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Live Face Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Capture cancelled by user.")
                break
            if key == ord(' ') and is_aligned and face_landmarks is not None:
                aligned = self.aligner.align_face(frame, face_landmarks)
                if aligned is not None:
                    print("Face captured and aligned successfully!")
                    if liveness_result:
                        print(f"Liveness confidence: {liveness_result['confidence']:.3f}")
                        print(f"Scores: {liveness_result['scores']}")
                    captured_frame = aligned
                    break
                else:
                    print("Alignment failed. Please try again.")
        cap.release()
        cv2.destroyAllWindows()
        return captured_frame

    def _draw_liveness_info(self, frame: np.ndarray, liveness_result: dict, x: int, y: int):
        scores, confidence = liveness_result['scores'], liveness_result['confidence']
        num_scores = len(scores)
        box_height = 40 + (num_scores * 18)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 180, y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        y_offset = y + 20
        cv2.putText(frame, f"Liveness: {confidence:.2f}", (x + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        for key, value in scores.items():
            color = (0, 255, 0) if value > 0.5 else (0, 165, 255)
            display_name = key.replace('_', ' ').title()
            cv2.putText(frame, f"{display_name}: {value:.2f}", (x + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 18

    @staticmethod
    def _get_target_box(center_x: int, center_y: int, target_size: int) -> Tuple[int, int, int, int]:
        half = target_size // 2
        return center_x - half, center_y - half, center_x + half, center_y + half

def main():
    parser = argparse.ArgumentParser(
        description='Facial Detection - Live Capture System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--no-liveness', '-n', action='store_true', help='Disable liveness detection (NOT RECOMMENDED for production)')
    parser.add_argument('--camera', type=int, default=0, choices=[0, 1], help='Camera index to use (default: 0)')
    parser.add_argument('--output-dir', type=str, default="/home/un1/projects/facial_recognition/data/captured_faces",
                        help='Output directory for captured faces')
    args = parser.parse_args()
    enable_liveness = not args.no_liveness
    print(f"\n{'='*60}")
    print(f"  FACIAL DETECTION - Live Capture System")
    print(f"  Liveness Detection: {'ENABLED ✓' if enable_liveness else 'DISABLED'}")
    print(f"{'='*60}\n")
    if not enable_liveness:
        print("WARNING: Running without liveness detection (spoofing possible)")
    print("Usage: python facial_detection.py [--no-liveness|-n]\n")
    live_capture = LiveCapture(enable_liveness=enable_liveness, camera_index=args.camera)
    captured_face = live_capture.start()
    if captured_face is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.abspath(args.output_dir)
        base_dir = "/home/un1/projects/facial_recognition/data"
        if not output_dir.startswith(base_dir):
            print(f"\n✗ SECURITY ERROR: Invalid output directory")
            sys.exit()
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.chmod(output_dir, 0o700)
            test_file = os.path.join(output_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            print(f"\n✗ SECURITY ERROR: No write permission to {output_dir}")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ ERROR: Failed to prepare output directory: {e}")
            sys.exit(1)
        safe_timestamp = timestamp.replace('/', '').replace('\\', '').replace('..', '')
        filename = f"aligned_face_{safe_timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        if not os.path.abspath(output_path).startswith(base_dir):
            print(f"\n✗ SECURITY ERROR: Path traversal detected")
            sys.exit()
        success = cv2.imwrite(output_path, captured_face)
        if success:
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS: Face captured and saved")
            print(f"  File: {output_path}")
            print(f"  Size: {captured_face.shape[1]}x{captured_face.shape[0]}")
            print(f"{'='*60}\n")
        else:
            print(f"\n✗ ERROR: Failed to save image to {output_path}\n")
        cv2.imshow("Captured Aligned Face (160x160)", captured_face)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"\n{'='*60}")
        print(f"✗ FAILED: No face captured")
        print(f"  Possible reasons:")
        print(f"  • User cancelled (ESC pressed)")
        print(f"  • Camera error")
        print(f"  • Liveness check failed")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
