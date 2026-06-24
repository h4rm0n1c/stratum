import tempfile
import unittest
import json
from pathlib import Path

import torch
from torch import nn

from stratum.checkpoint import load_checkpoint, save_checkpoint


class CheckpointMetadataTest(unittest.TestCase):
    def test_default_checkpoint_uses_json_metadata_without_pt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            save_checkpoint({}, optimizer=None, step=123, out_dir=out_dir)

            self.assertTrue((out_dir / "trainer_state.json").exists())
            self.assertFalse((out_dir / "meta.pt").exists())
            self.assertEqual(list(out_dir.glob("*.pt")), [])
            state = json.loads((out_dir / "trainer_state.json").read_text())
            self.assertEqual(state["step"], 123)
            self.assertEqual(load_checkpoint({}, checkpoint_dir=out_dir), 123)

    def test_legacy_device_state_only_saves_trainable_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            module[0].weight.requires_grad_(False)
            module[0].bias.requires_grad_(False)

            save_checkpoint(
                {0: [module]},
                optimizer=None,
                step=7,
                out_dir=out_dir,
                save_legacy_device_state=True,
            )

            legacy = torch.load(out_dir / "device_0.pt", map_location="cpu")
            saved = legacy["module_0"]
            self.assertIn("1.weight", saved)
            self.assertIn("1.bias", saved)
            self.assertNotIn("0.weight", saved)
            self.assertNotIn("0.bias", saved)
            self.assertTrue((out_dir / "meta.pt").exists())

    def test_adapter_only_checkpoint_defaults_to_step_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "adapter_model.safetensors").write_bytes(b"placeholder")

            self.assertEqual(load_checkpoint({}, checkpoint_dir=out_dir), 0)


if __name__ == "__main__":
    unittest.main()
