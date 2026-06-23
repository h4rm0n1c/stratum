import unittest

import stratum


class PackageExportsTest(unittest.TestCase):
    def test_all_exports_exist(self):
        missing = [name for name in stratum.__all__ if not hasattr(stratum, name)]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
