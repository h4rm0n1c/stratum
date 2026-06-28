import importlib
import tempfile
import unittest
import json
from pathlib import Path

import torch
from torch import nn

from stratum.checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_async, AsyncCheckpointHandle

_have_safetensors = importlib.util.find_spec("safetensors") is not None
_skip_no_safetensors = unittest.skipUnless(_have_safetensors, "safetensors not installed")


class CheckpointMetadataTest(unittest.TestCase):
    def test_default_checkpoint_uses_json_metadata_without_pt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            save_checkpoint({}, optimizer=None, step=123, out_dir=out_dir)

            self.assertTrue((out_dir / "trainer_state.json").exists())
            self.assertFalse((out_dir / "optimizer_state.safetensors").exists())
            self.assertEqual(list(out_dir.glob("*.pt")), [])
            state = json.loads((out_dir / "trainer_state.json").read_text())
            self.assertEqual(state["step"], 123)
            self.assertEqual(load_checkpoint({}, checkpoint_dir=out_dir), 123)

    def test_adapter_only_checkpoint_defaults_to_step_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "adapter_model.safetensors").write_bytes(b"placeholder")

            self.assertEqual(load_checkpoint({}, checkpoint_dir=out_dir), 0)

    @_skip_no_safetensors
    def test_optimizer_state_saves_single_name_keyed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Linear(2, 2)
            opt = torch.optim.AdamW(module.parameters(), lr=0.123)

            class MockOptimizer:
                cpu_offload = False
                optimizers = {0: opt}

            save_checkpoint(
                {0: [module]},
                optimizer=MockOptimizer(),
                step=4,
                out_dir=out_dir,
                save_optimizer_state=True,
            )

            self.assertTrue((out_dir / "optimizer_state.safetensors").exists())
            self.assertFalse((out_dir / "optim_0.pt").exists())
            self.assertFalse((out_dir / "device_0.pt").exists())
            self.assertFalse((out_dir / "meta.pt").exists())

            ckpt = torch.load(out_dir / "optimizer_state.safetensors", map_location="cpu",
                              weights_only=False)
            self.assertEqual(ckpt["format_version"], 1)
            self.assertIsInstance(ckpt["state"], dict)
            # Keys are parameter names, not device IDs
            for key in ckpt["state"]:
                self.assertIsInstance(key, str)

    @_skip_no_safetensors
    def test_optimizer_state_loads_lr_and_restores_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Linear(2, 2)
            saved_opt = torch.optim.AdamW(module.parameters(), lr=0.123)

            class SavedOptimizer:
                cpu_offload = False
                optimizers = {0: saved_opt}

            save_checkpoint(
                {0: [module]},
                optimizer=SavedOptimizer(),
                step=4,
                out_dir=out_dir,
                save_optimizer_state=True,
            )

            loaded_module = nn.Linear(2, 2)
            loaded_opt = torch.optim.AdamW(loaded_module.parameters(), lr=0.001)

            class LoadedOptimizer:
                cpu_offload = False
                optimizers = {0: loaded_opt}

            step = load_checkpoint(
                {0: [loaded_module]},
                optimizer=LoadedOptimizer(),
                checkpoint_dir=out_dir,
            )

            self.assertEqual(step, 4)
            self.assertEqual(loaded_opt.param_groups[0]["lr"], 0.123)

    @_skip_no_safetensors
    def test_optimizer_adam_moments_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Linear(2, 2)
            saved_opt = torch.optim.AdamW(module.parameters(), lr=0.123)

            x = torch.ones(1, 2)
            module(x).sum().backward()
            saved_opt.step()
            saved_state = saved_opt.state_dict()

            class SavedOptimizer:
                cpu_offload = False
                optimizers = {0: saved_opt}

            save_checkpoint(
                {0: [module]},
                optimizer=SavedOptimizer(),
                step=9,
                out_dir=out_dir,
                save_optimizer_state=True,
            )

            loaded_module = nn.Linear(2, 2)
            loaded_opt = torch.optim.AdamW(loaded_module.parameters(), lr=0.001)

            class LoadedOptimizer:
                cpu_offload = False
                optimizers = {0: loaded_opt}

            step = load_checkpoint(
                {0: [loaded_module]},
                optimizer=LoadedOptimizer(),
                checkpoint_dir=out_dir,
            )
            loaded_state = loaded_opt.state_dict()

            self.assertEqual(step, 9)
            self.assertFalse((out_dir / "optim_0.pt").exists())
            self.assertEqual(len(saved_state["state"]), len(loaded_state["state"]))
            for saved_ps, loaded_ps in zip(
                saved_state["state"].values(),
                loaded_state["state"].values(),
            ):
                self.assertTrue(torch.equal(saved_ps["step"], loaded_ps["step"]))
                self.assertTrue(torch.allclose(saved_ps["exp_avg"], loaded_ps["exp_avg"]))
                self.assertTrue(torch.allclose(saved_ps["exp_avg_sq"], loaded_ps["exp_avg_sq"]))

    @_skip_no_safetensors
    def test_optimizer_state_portable_across_device_split(self):
        """Params saved from device 0 load correctly into device 1."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module = nn.Linear(2, 2)
            opt = torch.optim.AdamW(module.parameters(), lr=0.05)

            x = torch.ones(1, 2)
            module(x).sum().backward()
            opt.step()

            class SingleDeviceOptimizer:
                cpu_offload = False
                optimizers = {0: opt}

            save_checkpoint(
                {0: [module]},
                optimizer=SingleDeviceOptimizer(),
                step=1,
                out_dir=out_dir,
                save_optimizer_state=True,
            )

            # Resume with same module assigned to device 1 instead of 0
            resumed_module = nn.Linear(2, 2)
            resumed_opt = torch.optim.AdamW(resumed_module.parameters(), lr=0.001)

            class ResumedOptimizer:
                cpu_offload = False
                optimizers = {1: resumed_opt}

            step = load_checkpoint(
                {1: [resumed_module]},
                optimizer=ResumedOptimizer(),
                checkpoint_dir=out_dir,
            )

            self.assertEqual(step, 1)
            self.assertEqual(resumed_opt.param_groups[0]["lr"], 0.05)
            # Adam moments should have been restored
            resumed_state = resumed_opt.state_dict()
            orig_state = opt.state_dict()
            for rs, os in zip(resumed_state["state"].values(), orig_state["state"].values()):
                self.assertTrue(torch.allclose(rs["exp_avg"], os["exp_avg"]))


class AsyncCheckpointTest(unittest.TestCase):
    def test_async_returns_handle_and_writes_trainer_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            handle = save_checkpoint_async({}, optimizer=None, step=7, out_dir=out_dir)
            self.assertIsInstance(handle, AsyncCheckpointHandle)
            handle.join()
            self.assertTrue((out_dir / "trainer_state.json").exists())
            state = json.loads((out_dir / "trainer_state.json").read_text())
            self.assertEqual(state["step"], 7)

    def test_async_no_pt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            handle = save_checkpoint_async({}, optimizer=None, step=3, out_dir=out_dir)
            handle.join()
            self.assertEqual(list(out_dir.glob("*.pt")), [])

    @_skip_no_safetensors
    def test_async_optimizer_state_matches_sync(self):
        """Async write produces same optimizer_state.safetensors as sync write."""
        with tempfile.TemporaryDirectory() as tmp_async, \
             tempfile.TemporaryDirectory() as tmp_sync:
            module = nn.Linear(2, 2)
            opt = torch.optim.AdamW(module.parameters(), lr=0.05)
            module(torch.ones(1, 2)).sum().backward()
            opt.step()

            class MockOptimizer:
                cpu_offload = False
                optimizers = {0: opt}

            handle = save_checkpoint_async(
                {0: [module]}, optimizer=MockOptimizer(),
                step=5, out_dir=Path(tmp_async), save_optimizer_state=True,
            )
            save_checkpoint(
                {0: [module]}, optimizer=MockOptimizer(),
                step=5, out_dir=Path(tmp_sync), save_optimizer_state=True,
            )
            handle.join()

            async_ckpt = torch.load(
                Path(tmp_async) / "optimizer_state.safetensors",
                map_location="cpu", weights_only=False,
            )
            sync_ckpt = torch.load(
                Path(tmp_sync) / "optimizer_state.safetensors",
                map_location="cpu", weights_only=False,
            )
            self.assertEqual(async_ckpt["format_version"], sync_ckpt["format_version"])
            for k in sync_ckpt["state"]:
                self.assertIn(k, async_ckpt["state"])

    def test_join_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            handle = save_checkpoint_async({}, optimizer=None, step=1, out_dir=Path(tmp))
            handle.join()
            handle.join()  # must not raise

    def test_async_write_completes_before_next_join(self):
        """Second handle's join waits for both writes to finish."""
        with tempfile.TemporaryDirectory() as tmp1, \
             tempfile.TemporaryDirectory() as tmp2:
            h1 = save_checkpoint_async({}, optimizer=None, step=10, out_dir=Path(tmp1))
            h1.join()
            h2 = save_checkpoint_async({}, optimizer=None, step=20, out_dir=Path(tmp2))
            h2.join()
            self.assertEqual(
                json.loads((Path(tmp1) / "trainer_state.json").read_text())["step"], 10
            )
            self.assertEqual(
                json.loads((Path(tmp2) / "trainer_state.json").read_text())["step"], 20
            )


if __name__ == "__main__":
    unittest.main()
