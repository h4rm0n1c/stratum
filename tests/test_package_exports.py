import unittest
import inspect

import stratum
from stratum.model.registry import ModelArch
from stratum.telemetry import parse_int_set


class PackageExportsTest(unittest.TestCase):
    def test_all_exports_exist(self):
        missing = [name for name in stratum.__all__ if not hasattr(stratum, name)]
        self.assertEqual(missing, [])

    def test_build_pipeline_accepts_train_forwarded_kwargs(self):
        params = inspect.signature(stratum.build_pipeline).parameters
        for name in [
            "flash_layers",
            "flash_window_left",
            "flash_window_right",
            "dense_attention_masks",
            "stage_memory_limit_gib",
            "hf_model_name_or_path",
        ]:
            self.assertIn(name, params)

    def test_base_model_arch_accepts_adapter_forwarded_kwargs(self):
        params = inspect.signature(ModelArch.build).parameters
        self.assertTrue(
            any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        )

    def test_parse_int_set_accepts_disabled_sentinels(self):
        for value in ["none", "off", "false", " NONE "]:
            self.assertEqual(parse_int_set(value), set())


if __name__ == "__main__":
    unittest.main()
