import unittest

from main import get_package_version


class VersionHelpersTests(unittest.TestCase):
    def test_get_package_version_returns_string(self):
        version = get_package_version("langgraph")
        self.assertIsInstance(version, str)
        self.assertTrue(version)


if __name__ == "__main__":
    unittest.main()
