import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Patch database_config before any src module tries to import it
os.environ.setdefault("PG_PASSWORD", "test_password")


class TestSimilarityCalculation(unittest.TestCase):
    """Test cosine similarity calculation used in recognition_controller."""

    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Standalone copy of FacialRecognitionSystem.calculate_similarity for unit testing."""
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float((similarity + 1) / 2)

    def test_identical_vectors_return_1(self):
        vec = np.random.randn(512).astype(np.float32)
        score = self.calculate_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_opposite_vectors_return_0(self):
        vec = np.random.randn(512).astype(np.float32)
        score = self.calculate_similarity(vec, -vec)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_orthogonal_vectors_return_half(self):
        vec1 = np.zeros(512, dtype=np.float32)
        vec2 = np.zeros(512, dtype=np.float32)
        vec1[0] = 1.0
        vec2[1] = 1.0
        score = self.calculate_similarity(vec1, vec2)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_result_in_valid_range(self):
        for _ in range(100):
            v1 = np.random.randn(512).astype(np.float32)
            v2 = np.random.randn(512).astype(np.float32)
            score = self.calculate_similarity(v1, v2)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_symmetry(self):
        v1 = np.random.randn(512).astype(np.float32)
        v2 = np.random.randn(512).astype(np.float32)
        self.assertAlmostEqual(
            self.calculate_similarity(v1, v2),
            self.calculate_similarity(v2, v1),
            places=7
        )

    def test_scaled_vectors_same_result(self):
        v1 = np.random.randn(512).astype(np.float32)
        v2 = np.random.randn(512).astype(np.float32)
        score_original = self.calculate_similarity(v1, v2)
        score_scaled = self.calculate_similarity(v1 * 5.0, v2 * 0.3)
        self.assertAlmostEqual(score_original, score_scaled, places=5)

    def test_threshold_boundary(self):
        """Test that similar vectors exceed the 0.8 recognition threshold."""
        vec = np.random.randn(512).astype(np.float32)
        # Add small noise
        noise = np.random.randn(512).astype(np.float32) * 0.05
        similar_vec = vec + noise
        score = self.calculate_similarity(vec, similar_vec)
        self.assertGreater(score, 0.8, "Slightly noisy vector should still exceed threshold")


if __name__ == "__main__":
    unittest.main()
