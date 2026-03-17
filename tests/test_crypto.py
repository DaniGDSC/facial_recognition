import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Patch env before importing src modules
os.environ.setdefault("PG_PASSWORD", "test_password")

from crypto import (
    encrypt_image,
    decrypt_image,
    compute_embedding_hmac,
    verify_embedding_hmac,
)


class TestImageEncryption(unittest.TestCase):

    def setUp(self):
        self.key = os.urandom(32)
        self.plaintext = b"fake image data " * 100

    def test_encrypt_decrypt_roundtrip(self):
        encrypted = encrypt_image(self.plaintext, self.key)
        decrypted = decrypt_image(encrypted, self.key)
        self.assertEqual(decrypted, self.plaintext)

    def test_encrypted_differs_from_plaintext(self):
        encrypted = encrypt_image(self.plaintext, self.key)
        self.assertNotEqual(encrypted, self.plaintext)

    def test_encrypted_includes_nonce(self):
        encrypted = encrypt_image(self.plaintext, self.key)
        # nonce (12) + ciphertext (>= plaintext len) + tag (16)
        self.assertGreater(len(encrypted), 12 + len(self.plaintext))

    def test_wrong_key_fails(self):
        encrypted = encrypt_image(self.plaintext, self.key)
        wrong_key = os.urandom(32)
        with self.assertRaises(Exception):
            decrypt_image(encrypted, wrong_key)

    def test_tampered_data_fails(self):
        encrypted = encrypt_image(self.plaintext, self.key)
        tampered = bytearray(encrypted)
        tampered[-1] ^= 0xFF
        with self.assertRaises(Exception):
            decrypt_image(bytes(tampered), self.key)

    def test_too_short_data_raises(self):
        with self.assertRaises(ValueError):
            decrypt_image(b"short", self.key)

    def test_different_encryptions_differ(self):
        enc1 = encrypt_image(self.plaintext, self.key)
        enc2 = encrypt_image(self.plaintext, self.key)
        # Different nonces should produce different ciphertexts
        self.assertNotEqual(enc1, enc2)


class TestEmbeddingHMAC(unittest.TestCase):

    def setUp(self):
        self.key = os.urandom(32)
        self.embedding = [float(x) for x in range(512)]

    def test_compute_and_verify(self):
        tag = compute_embedding_hmac(self.embedding, self.key)
        self.assertTrue(verify_embedding_hmac(self.embedding, tag, self.key))

    def test_wrong_key_fails_verification(self):
        tag = compute_embedding_hmac(self.embedding, self.key)
        wrong_key = os.urandom(32)
        self.assertFalse(verify_embedding_hmac(self.embedding, tag, wrong_key))

    def test_tampered_embedding_fails(self):
        tag = compute_embedding_hmac(self.embedding, self.key)
        tampered = self.embedding.copy()
        tampered[0] += 0.001
        self.assertFalse(verify_embedding_hmac(tampered, tag, self.key))

    def test_deterministic(self):
        tag1 = compute_embedding_hmac(self.embedding, self.key)
        tag2 = compute_embedding_hmac(self.embedding, self.key)
        self.assertEqual(tag1, tag2)

    def test_different_vectors_different_tags(self):
        other = [float(x + 1) for x in range(512)]
        tag1 = compute_embedding_hmac(self.embedding, self.key)
        tag2 = compute_embedding_hmac(other, self.key)
        self.assertNotEqual(tag1, tag2)

    def test_tag_is_hex_string(self):
        tag = compute_embedding_hmac(self.embedding, self.key)
        self.assertEqual(len(tag), 64)  # SHA-256 hex = 64 chars
        int(tag, 16)  # Should not raise


if __name__ == "__main__":
    unittest.main()
