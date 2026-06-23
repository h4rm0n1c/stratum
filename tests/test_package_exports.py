import unittest
import inspect

import stratum


class PackageExportsTest(unittest.TestCase):
    def test_all_exports_exist(self):
        missing = [name for name in stratum.__all__ if not hasattr(stratum, name)]
        self.assertEqual(missing, [])

    def test_build_pipeline_accepts_train_forwarded_kwargs(self):
        params = inspect.signature(stratum.build_pipeline).parameters
        for name in [
            "volta_layers",
            "volta_window_left",
            "volta_window_right",
            "dense_attention_masks",
            "stage_memory_limit_gib",
        ]:
            self.assertIn(name, params)


if __name__ == "__main__":
    unittest.main()
