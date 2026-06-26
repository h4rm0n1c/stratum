import unittest

import torch
from torch.utils.checkpoint import checkpoint

from stratum.context import (
    checkpoint_context_fn,
    doing_recompute,
    get_recompute_data,
    save_for_recompute,
    set_recompute_event_recorder,
)


class TestRecomputeContext(unittest.TestCase):
    def test_checkpoint_context_replays_saved_data_during_backward_recompute(self):
        events = []
        x = torch.randn(8, requires_grad=True)
        scale_source = torch.full_like(x, 3.0).detach()

        def fn(t):
            if doing_recompute():
                (scale,) = get_recompute_data()
                events.append("recompute")
            else:
                scale = scale_source
                save_for_recompute(scale)
                events.append("forward")
            return (t * scale).sin().sum()

        y = checkpoint(
            fn,
            x,
            use_reentrant=False,
            context_fn=checkpoint_context_fn,
        )
        y.backward()

        self.assertEqual(events, ["forward", "recompute"])
        self.assertIsNotNone(x.grad)

    def test_save_for_recompute_is_noop_without_forward_context(self):
        save_for_recompute(torch.tensor(1.0))
        self.assertFalse(doing_recompute())

    def test_checkpoint_context_records_recompute_lifecycle(self):
        records = []
        x = torch.randn(8, requires_grad=True)
        saved = torch.ones_like(x).detach()

        def recorder(name, wall_ms, fields):
            records.append((name, wall_ms, dict(fields)))

        def fn(t):
            if doing_recompute():
                (scale,) = get_recompute_data()
            else:
                scale = saved
                save_for_recompute(scale)
            return (t * scale).sin().sum()

        set_recompute_event_recorder(recorder)
        try:
            y = checkpoint(
                fn,
                x,
                use_reentrant=False,
                context_fn=lambda: checkpoint_context_fn(
                    layer_idx=7,
                    stage_device=1,
                    recompute_grain="layer",
                ),
            )
            y.backward()
        finally:
            set_recompute_event_recorder(None)

        names = [record[0] for record in records]
        self.assertEqual(names, ["recompute_save", "recompute_enter", "layer_recompute"])
        for name, wall_ms, fields in records:
            self.assertEqual(fields["layer_idx"], 7)
            self.assertEqual(fields["stage_device"], 1)
            self.assertEqual(fields["recompute_grain"], "layer")
            self.assertEqual(fields["saved_tensors"], 1)
            self.assertEqual(fields["saved_bytes"], saved.numel() * saved.element_size())
            if name == "layer_recompute":
                self.assertGreaterEqual(wall_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
