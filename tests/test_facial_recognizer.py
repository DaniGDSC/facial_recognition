import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Patch env vars before importing
os.environ.setdefault("PG_PASSWORD", "test_password")

try:
    from facial_recognition import FacialRecognizer
except ImportError:
    FacialRecognizer = None


@unittest.skipIf(FacialRecognizer is None, "torch not installed")
class TestFacialRecognizerProcessing(unittest.TestCase):
    """Test image preprocessing in FacialRecognizer (no model weights needed for some tests)."""

    def setUp(self):
        self.recognizer = FacialRecognizer(device='cpu')

    def test_process_valid_bgr_image(self):
        """A valid 3-channel image should produce a tensor."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        tensor = self.recognizer.process_numpy_array(image)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 160, 160))

    def test_process_invalid_grayscale_image(self):
        """A grayscale image (2D) should return None."""
        image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        tensor = self.recognizer.process_numpy_array(image)
        self.assertIsNone(tensor)

    def test_process_invalid_4channel_image(self):
        """A 4-channel (BGRA) image should return None."""
        image = np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8)
        tensor = self.recognizer.process_numpy_array(image)
        self.assertIsNone(tensor)

    def test_tensor_value_range(self):
        """Output tensor values should be normalized to approximately [-1, 1]."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        tensor = self.recognizer.process_numpy_array(image)
        self.assertGreaterEqual(tensor.min().item(), -1.01)
        self.assertLessEqual(tensor.max().item(), 1.01)

    def test_extract_features_returns_none_for_none_input(self):
        """extract_features should handle None input gracefully."""
        result = self.recognizer.extract_features(None)
        self.assertIsNone(result)

    def test_extract_features_output_shape(self):
        """Feature extraction should produce a 512-D normalized vector."""
        image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        tensor = self.recognizer.process_numpy_array(image)
        embeddings = self.recognizer.extract_features(tensor)
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (1, 512))

    def test_extract_features_are_normalized(self):
        """Embeddings should be L2-normalized (unit vectors)."""
        image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        tensor = self.recognizer.process_numpy_array(image)
        embeddings = self.recognizer.extract_features(tensor)
        norm = embeddings.norm(dim=1).item()
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_process_and_extract_from_array(self):
        """End-to-end: numpy array -> 512-D embedding."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        embeddings = self.recognizer.process_and_extract_from_array(image)
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[1], 512)

    def test_deterministic_output(self):
        """Same input should produce the same output (model in eval mode)."""
        image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        emb1 = self.recognizer.process_and_extract_from_array(image)
        emb2 = self.recognizer.process_and_extract_from_array(image)
        diff = (emb1 - emb2).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
