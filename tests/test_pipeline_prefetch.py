import types
import unittest
from unittest import mock

import torch

import stratum.pipeline as pipeline_mod
from stratum.pipeline import StratumPipeline


class Prefix(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        self.events.append("prefix_forward")
        hidden = input_ids.float().unsqueeze(-1)
        return (hidden, None, input_ids, hidden, {}, labels, None)


class Stage(torch.nn.Module):
    def __init__(self, name, events):
        super().__init__()
        self.name = name
        self.events = events
        self.device_id = 0
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, tuple_data):
        self.events.append(f"{self.name}_forward")
        return tuple_data


class Postfix(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, tuple_data):
        self.events.append("postfix_forward")
        return types.SimpleNamespace(loss=tuple_data[0].sum())


class PipelinePrefetchTests(unittest.TestCase):
    def _pipeline(self, *, prefetch_nf4):
        events = []
        pipe = StratumPipeline(
            Prefix(events),
            [Stage("stage0", events), Stage("stage1", events)],
            Postfix(events),
            prefetch_nf4=prefetch_nf4,
        )
        return pipe, events

    def test_prefetch_flag_schedules_next_module_before_current_forward(self):
        pipe, events = self._pipeline(prefetch_nf4=True)

        def fake_prefetch(module, device_id):
            name = getattr(module, "name", module.__class__.__name__.lower())
            events.append(f"{name}_prefetch")
            return f"prefetch:{name}"

        def fake_ensure(module, device_id, prefetch):
            name = getattr(module, "name", module.__class__.__name__.lower())
            events.append(f"{name}_ensure:{prefetch}")
            return 0

        with mock.patch.object(pipeline_mod, "prefetch_weights", side_effect=fake_prefetch):
            with mock.patch.object(pipeline_mod, "ensure_prefetched_weights", side_effect=fake_ensure):
                out = pipe(torch.tensor([[1, 2]]), labels=torch.tensor([[1, 2]]))

        self.assertEqual(out.loss.item(), 3.0)
        self.assertLess(events.index("stage0_prefetch"), events.index("prefix_forward"))
        self.assertLess(events.index("stage1_prefetch"), events.index("stage0_forward"))
        self.assertLess(events.index("postfix_prefetch"), events.index("stage1_forward"))
        self.assertIn("stage0_ensure:prefetch:stage0", events)
        self.assertIn("stage1_ensure:prefetch:stage1", events)
        self.assertIn("postfix_ensure:prefetch:postfix", events)

    def test_prefetch_disabled_uses_synchronous_ensure_only(self):
        pipe, events = self._pipeline(prefetch_nf4=False)

        def fake_ensure(module, device_id):
            name = getattr(module, "name", module.__class__.__name__.lower())
            events.append(f"{name}_sync_ensure")
            return 0

        with mock.patch.object(pipeline_mod, "prefetch_weights") as prefetch_mock:
            with mock.patch.object(pipeline_mod, "ensure_weights", side_effect=fake_ensure):
                pipe(torch.tensor([[1, 2]]), labels=torch.tensor([[1, 2]]))

        prefetch_mock.assert_not_called()
        self.assertIn("prefix_sync_ensure", events)
        self.assertIn("stage0_sync_ensure", events)
        self.assertIn("stage1_sync_ensure", events)
        self.assertIn("postfix_sync_ensure", events)


if __name__ == "__main__":
    unittest.main()
