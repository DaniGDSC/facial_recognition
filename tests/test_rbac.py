import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Patch env before importing src modules
os.environ.setdefault("PG_PASSWORD", "test_password")

from rbac import RBACManager, ROLE_OPERATOR, ROLE_ADMIN, ROLE_USER


class TestRBACManager(unittest.TestCase):

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.rbac = RBACManager(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_no_operators_initially(self):
        self.assertFalse(self.rbac.has_operators())

    def test_register_operator(self):
        result = self.rbac.register_operator("op1", "1234")
        self.assertTrue(result)
        self.assertTrue(self.rbac.has_operators())

    def test_register_duplicate_fails(self):
        self.rbac.register_operator("op1", "1234")
        result = self.rbac.register_operator("op1", "5678")
        self.assertFalse(result)

    def test_register_invalid_role_fails(self):
        result = self.rbac.register_operator("op1", "1234", role=ROLE_USER)
        self.assertFalse(result)

    def test_authenticate_success(self):
        self.rbac.register_operator("op1", "secret123")
        result = self.rbac.authenticate_operator("op1", "secret123")
        self.assertTrue(result)
        self.assertEqual(self.rbac.current_operator, "op1")
        self.assertEqual(self.rbac.current_role, ROLE_OPERATOR)

    def test_authenticate_wrong_pin(self):
        self.rbac.register_operator("op1", "secret123")
        result = self.rbac.authenticate_operator("op1", "wrong")
        self.assertFalse(result)
        self.assertIsNone(self.rbac.current_operator)

    def test_authenticate_nonexistent_operator(self):
        result = self.rbac.authenticate_operator("ghost", "1234")
        self.assertFalse(result)

    def test_admin_permissions(self):
        self.rbac.register_operator("admin1", "admin", role=ROLE_ADMIN)
        self.rbac.authenticate_operator("admin1", "admin")

        self.assertTrue(self.rbac.check_permission("enroll"))
        self.assertTrue(self.rbac.check_permission("recognize"))
        self.assertTrue(self.rbac.check_permission("authenticate"))
        self.assertTrue(self.rbac.check_permission("view_statistics"))
        self.assertTrue(self.rbac.check_permission("manage_users"))

    def test_operator_permissions(self):
        self.rbac.register_operator("op1", "1234", role=ROLE_OPERATOR)
        self.rbac.authenticate_operator("op1", "1234")

        self.assertTrue(self.rbac.check_permission("enroll"))
        self.assertTrue(self.rbac.check_permission("recognize"))
        self.assertTrue(self.rbac.check_permission("view_statistics"))
        self.assertFalse(self.rbac.check_permission("manage_users"))

    def test_unauthenticated_has_no_permissions(self):
        self.assertFalse(self.rbac.check_permission("enroll"))
        self.assertFalse(self.rbac.check_permission("recognize"))

    def test_require_permission_raises(self):
        with self.assertRaises(PermissionError):
            self.rbac.require_permission("enroll")

    def test_require_permission_passes_when_authorized(self):
        self.rbac.register_operator("op1", "1234")
        self.rbac.authenticate_operator("op1", "1234")
        # Should not raise
        self.rbac.require_permission("enroll")

    def test_logout(self):
        self.rbac.register_operator("op1", "1234")
        self.rbac.authenticate_operator("op1", "1234")
        self.assertIsNotNone(self.rbac.current_operator)

        self.rbac.logout()
        self.assertIsNone(self.rbac.current_operator)
        self.assertIsNone(self.rbac.current_role)
        self.assertFalse(self.rbac.check_permission("enroll"))

    def test_unknown_action(self):
        self.rbac.register_operator("op1", "1234")
        self.rbac.authenticate_operator("op1", "1234")
        self.assertFalse(self.rbac.check_permission("nonexistent_action"))


if __name__ == "__main__":
    unittest.main()
