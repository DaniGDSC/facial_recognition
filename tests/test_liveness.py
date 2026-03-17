import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault("PG_PASSWORD", "test_password")

try:
    from liveness import MultiFactorLiveness
except ImportError:
    MultiFactorLiveness = None


@unittest.skipIf(MultiFactorLiveness is None, "cv2 not installed")
class TestMultiFactorLiveness(unittest.TestCase):

    def setUp(self):
        self.detector = MultiFactorLiveness()

    def test_invalid_input_returns_not_live(self):
        result = self.detector.check_liveness(None, None)
        self.assertFalse(result['is_live'])
        self.assertEqual(result['confidence'], 0.0)

    def test_invalid_face_box_returns_not_live(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = self.detector.check_liveness(image, np.array([1, 2]))  # Wrong shape
        self.assertFalse(result['is_live'])

    def test_empty_face_region_returns_not_live(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = self.detector.check_liveness(image, np.array([100, 100, 100, 100]))
        self.assertFalse(result['is_live'])

    def test_returns_all_score_keys(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        box = np.array([10, 10, 190, 190])
        result = self.detector.check_liveness(image, box)

        self.assertIn('is_live', result)
        self.assertIn('confidence', result)
        self.assertIn('scores', result)
        self.assertIn('checks_passed', result)
        self.assertIn('checks_total', result)
        self.assertIn('reason', result)

        scores = result['scores']
        self.assertIn('texture', scores)
        self.assertIn('frequency', scores)
        self.assertIn('color', scores)
        self.assertIn('reflection', scores)
        self.assertIn('eye_region', scores)

    def test_scores_in_valid_range(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        box = np.array([10, 10, 190, 190])
        result = self.detector.check_liveness(image, box)

        for name, score in result['scores'].items():
            self.assertGreaterEqual(score, 0.0, f"{name} score below 0")
            self.assertLessEqual(score, 1.0, f"{name} score above 1")

    def test_confidence_in_valid_range(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        box = np.array([10, 10, 190, 190])
        result = self.detector.check_liveness(image, box)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_checks_total_equals_5(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        box = np.array([10, 10, 190, 190])
        result = self.detector.check_liveness(image, box)
        self.assertEqual(result['checks_total'], 5)

    def test_with_landmarks(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        box = np.array([10, 10, 190, 190])
        landmarks = np.array([[60, 80], [140, 80], [100, 120], [70, 150], [130, 150]], dtype=np.float32)
        result = self.detector.check_liveness(image, box, landmarks)

        self.assertIn('eye_region', result['scores'])
        # With landmarks, eye score should not be the default 0.5
        # (though it might coincidentally be — this just verifies no crash)
        self.assertIsInstance(result['scores']['eye_region'], float)

    def test_uniform_image_low_texture_score(self):
        """A uniformly colored image (like a blank screen) should score low on texture."""
        uniform = np.ones((200, 200, 3), dtype=np.uint8) * 128
        box = np.array([10, 10, 190, 190])
        result = self.detector.check_liveness(uniform, box)
        self.assertLess(result['scores']['texture'], 0.5)


if __name__ == "__main__":
    unittest.main()
