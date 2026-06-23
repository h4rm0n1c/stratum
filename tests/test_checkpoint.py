import tempfile
import unittest
from pathlib import Path

import torch

from stratum.checkpoint import load_checkpoint, save_checkpoint


class CheckpointMetadataTest(unittest.TestCase):
    def test_save_and_load_step_metadata_without_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            save_checkpoint({}, optimizer=None, step=123, out_dir=out_dir)

            self.assertTrue((out_dir / "meta.pt").exists())
            self.assertEqual(torch.load(out_dir / "meta.pt", map_location="cpu")["step"], 123)
            self.assertEqual(load_checkpoint({}, checkpoint_dir=out_dir), 123)


if __name__ == "__main__":
    unittest.main()
