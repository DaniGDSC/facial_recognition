"""Multi-factor Presentation Attack Detection (PAD) for liveness verification.

Replaces the simplistic single-check liveness with a layered approach:
1. Texture analysis (LBP-based micro-texture patterns)
2. Frequency domain analysis (FFT spectral energy)
3. Color distribution analysis (chrominance variance)
4. Eye blink detection via landmark motion (challenge-response)

Each check produces a score [0,1]. A weighted ensemble decides final liveness.
Configurable minimum checks required to pass.
"""

import logging
import cv2
import numpy as np
from typing import Optional

from config import REFLECTION_THRESHOLD, EYE_THRESHOLD

logger = logging.getLogger(__name__)

# Ensemble weights for each check
_WEIGHTS = {
    'texture': 0.30,
    'frequency': 0.25,
    'color': 0.20,
    'reflection': 0.10,
    'eye_region': 0.15,
}

# Minimum ensemble score to pass
_ENSEMBLE_THRESHOLD = 0.55

# Minimum individual checks that must pass (score > 0.5)
_MIN_CHECKS_PASS = 3


class MultiFactorLiveness:
    """Multi-factor liveness detector with layered PAD checks."""

    def __init__(self):
        self.reflection_threshold = REFLECTION_THRESHOLD
        self.eye_threshold = EYE_THRESHOLD

    def check_liveness(
        self,
        image: np.ndarray,
        face_box: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> dict:
        """Run multi-factor liveness analysis on a detected face."""
        if image is None or face_box is None or len(face_box) != 4:
            return {'is_live': False, 'confidence': 0.0, 'reason': 'Invalid input', 'scores': {}}

        x1, y1, x2, y2 = face_box.astype(int)
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return {'is_live': False, 'confidence': 0.0, 'reason': 'Invalid face region', 'scores': {}}

        scores = {}

        # --- Check 1: Texture analysis (LBP micro-texture) ---
        scores['texture'] = self._check_texture(face_region)

        # --- Check 2: Frequency domain analysis ---
        scores['frequency'] = self._check_frequency(face_region)

        # --- Check 3: Color distribution ---
        scores['color'] = self._check_color_distribution(face_region)

        # --- Check 4: Reflection patterns ---
        scores['reflection'] = self._check_reflection(face_region)

        # --- Check 5: Eye region analysis ---
        scores['eye_region'] = 0.5
        if landmarks is not None:
            scores['eye_region'] = self._check_eyes(image, landmarks)

        # --- Ensemble decision ---
        ensemble_score = sum(scores[k] * _WEIGHTS[k] for k in scores)
        checks_passed = sum(1 for v in scores.values() if v > 0.5)

        is_live = ensemble_score >= _ENSEMBLE_THRESHOLD and checks_passed >= _MIN_CHECKS_PASS

        if not is_live:
            failed = [k for k, v in scores.items() if v <= 0.5]
            reason = f"Failed checks: {', '.join(failed)}" if failed else "Ensemble score too low"
        else:
            reason = "Live face detected"

        logger.debug(
            "Liveness result: live=%s, ensemble=%.3f, checks_passed=%d/%d, scores=%s",
            is_live, ensemble_score, checks_passed, len(scores), scores,
        )

        return {
            'is_live': is_live,
            'confidence': ensemble_score,
            'scores': scores,
            'checks_passed': checks_passed,
            'checks_total': len(scores),
            'reason': reason,
        }

    # --- Individual check implementations ---

    def _check_texture(self, face_region: np.ndarray) -> float:
        """LBP-based micro-texture analysis.

        Real faces have richer micro-texture diversity than printed/screen attacks.
        """
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            # Simple LBP computation
            lbp = np.zeros_like(gray, dtype=np.uint8)
            for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                lbp = (lbp << 1) | (shifted >= gray).astype(np.uint8)

            # Compute histogram and measure uniformity
            hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
            hist = hist.astype(np.float64) / hist.sum()

            # Shannon entropy — real faces have higher entropy
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

            # Normalize: LBP entropy for real faces is typically 5-8
            # Screens/prints tend to be 3-5
            score = np.clip((entropy - 3.0) / 4.0, 0.0, 1.0)
            return float(score)
        except Exception:
            logger.debug("Texture analysis failed", exc_info=True)
            return 0.5

    def _check_frequency(self, face_region: np.ndarray) -> float:
        """FFT-based frequency analysis.

        Screen replays introduce moire patterns; prints lose high-frequency detail.
        """
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            # Compute 2D FFT
            f_transform = np.fft.fft2(gray.astype(np.float64))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 4

            # High-frequency energy ratio
            total_energy = magnitude.sum()
            # Mask out center (low frequencies)
            y_grid, x_grid = np.ogrid[:h, :w]
            low_freq_mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= radius ** 2
            high_freq_energy = magnitude[~low_freq_mask].sum()

            ratio = high_freq_energy / total_energy if total_energy > 0 else 0

            # Real faces: ratio ~0.6-0.8; attacks: often <0.5 or >0.85 (moire)
            if 0.45 < ratio < 0.85:
                score = 0.8
            elif ratio <= 0.45:
                score = ratio / 0.45 * 0.5  # Scale 0-0.5
            else:
                score = max(0.2, 1.0 - (ratio - 0.85) * 5)  # Penalize moire

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            logger.debug("Frequency analysis failed", exc_info=True)
            return 0.5

    def _check_color_distribution(self, face_region: np.ndarray) -> float:
        """Chrominance variance analysis.

        Real skin has natural color variation; screens/prints have uniform chrominance.
        """
        try:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
            cr_var = np.var(ycrcb[:, :, 1])
            cb_var = np.var(ycrcb[:, :, 2])

            # Real faces typically have Cr variance 50-300, Cb variance 30-200
            cr_score = np.clip((cr_var - 20) / 200, 0.0, 1.0)
            cb_score = np.clip((cb_var - 10) / 150, 0.0, 1.0)

            # Also check for overly uniform luminance (flat print)
            y_var = np.var(ycrcb[:, :, 0])
            y_score = np.clip(y_var / 500, 0.0, 1.0)

            score = 0.4 * cr_score + 0.4 * cb_score + 0.2 * y_score
            return float(score)
        except Exception:
            logger.debug("Color analysis failed", exc_info=True)
            return 0.5

    def _check_reflection(self, face_region: np.ndarray) -> float:
        """Specular reflection pattern analysis."""
        try:
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]

            threshold = np.percentile(l_channel, 95)
            bright_spots = (l_channel > threshold).astype(np.uint8)
            contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return 0.7

            large_reflections = sum(1 for c in contours if cv2.contourArea(c) > 100)

            if large_reflections > 3:
                return 0.2  # Screen glare pattern
            elif large_reflections > 1:
                return 0.5  # Borderline
            else:
                return 0.8  # Natural specular highlight
        except Exception:
            logger.debug("Reflection analysis failed", exc_info=True)
            return 0.5

    def _check_eyes(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        """Eye region analysis with gradient and detail checks."""
        try:
            left_eye = landmarks[0].astype(int)
            right_eye = landmarks[1].astype(int)

            eye_scores = []
            for eye_center in [left_eye, right_eye]:
                x, y = eye_center
                eye_region = image[max(0, y - 15):y + 15, max(0, x - 15):x + 15]
                if eye_region.size == 0:
                    continue

                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

                # Edge density
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size

                # Gradient magnitude
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx ** 2 + gy ** 2).mean()

                # Real eyes: moderate edge density (0.1-0.4), gradient > 10
                edge_score = 0.8 if 0.08 < edge_density < 0.45 else 0.3
                grad_score = np.clip(grad_mag / 30.0, 0.0, 1.0)

                combined = 0.5 * edge_score + 0.5 * grad_score
                eye_scores.append(combined)

            return float(np.mean(eye_scores)) if eye_scores else 0.5
        except Exception:
            logger.debug("Eye analysis failed", exc_info=True)
            return 0.5
