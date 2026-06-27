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

    def test_optimizer_state_loads_without_legacy_device_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Linear(2, 2)
            saved_opt = torch.optim.AdamW(module.parameters(), lr=0.123)

            class SavedOptimizer:
                cpu_offload = True
                optimizers = {0: saved_opt}

            save_checkpoint(
                {0: [module]},
                optimizer=SavedOptimizer(),
                step=4,
                out_dir=out_dir,
                save_optimizer_state=True,
                save_legacy_device_state=False,
            )
            self.assertFalse((out_dir / "device_0.pt").exists())

            loaded_module = nn.Linear(2, 2)
            loaded_opt = torch.optim.AdamW(loaded_module.parameters(), lr=0.001)

            class LoadedOptimizer:
                cpu_offload = True
                optimizers = {0: loaded_opt}

            step = load_checkpoint(
                {0: [loaded_module]},
                optimizer=LoadedOptimizer(),
                checkpoint_dir=out_dir,
            )

            self.assertEqual(step, 4)
            self.assertEqual(loaded_opt.param_groups[0]["lr"], 0.123)

    def test_optimizer_adam_moments_round_trip_without_legacy_device_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Linear(2, 2)
            saved_opt = torch.optim.AdamW(module.parameters(), lr=0.123)

            x = torch.ones(1, 2)
            module(x).sum().backward()
            saved_opt.step()
            saved_state = saved_opt.state_dict()

            class SavedOptimizer:
                cpu_offload = True
                optimizers = {0: saved_opt}

            save_checkpoint(
                {0: [module]},
                optimizer=SavedOptimizer(),
                step=9,
                out_dir=out_dir,
                save_optimizer_state=True,
                save_legacy_device_state=False,
            )

            loaded_module = nn.Linear(2, 2)
            loaded_opt = torch.optim.AdamW(loaded_module.parameters(), lr=0.001)

            class LoadedOptimizer:
                cpu_offload = True
                optimizers = {0: loaded_opt}

            step = load_checkpoint(
                {0: [loaded_module]},
                optimizer=LoadedOptimizer(),
                checkpoint_dir=out_dir,
            )
            loaded_state = loaded_opt.state_dict()

            self.assertEqual(step, 9)
            self.assertFalse((out_dir / "device_0.pt").exists())
            self.assertEqual(len(saved_state["state"]), len(loaded_state["state"]))
            for saved_param_state, loaded_param_state in zip(
                saved_state["state"].values(),
                loaded_state["state"].values(),
            ):
                self.assertTrue(
                    torch.equal(saved_param_state["step"], loaded_param_state["step"])
                )
                self.assertTrue(
                    torch.allclose(saved_param_state["exp_avg"], loaded_param_state["exp_avg"])
                )
                self.assertTrue(
                    torch.allclose(saved_param_state["exp_avg_sq"], loaded_param_state["exp_avg_sq"])
                )


if __name__ == "__main__":
    unittest.main()
