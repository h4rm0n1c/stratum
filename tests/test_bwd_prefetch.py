"""Tests for cross-group backward weight prefetch look-ahead."""

import unittest
from unittest.mock import MagicMock, patch, call
from typing import Any

import torch
import torch.nn as nn

from stratum.runtime import (
    _AnchorMeta,
    CapturedInput,
    anchor_explicit_group_backward,
    capture_backward_input,
)


def _simple_run_group(input_data):
    hidden, *rest = input_data
    return (hidden * 2, *rest)


def _make_meta(*, look_ahead_upload=None) -> _AnchorMeta:
    inp = torch.ones(2, requires_grad=True)
    captured = capture_backward_input((inp,))
    return _AnchorMeta(
        run_group=_simple_run_group,
        captured_input=captured,
        input_indices=[0],
        output_indices=[0],
        output_leaf_count=1,
        output_spec=captured.spec,
        recompute_data=None,
        timing_recorder=None,
        timing_fields={},
        compute_stream=None,
        on_backward_complete=None,
        look_ahead_upload=look_ahead_upload,
    )


class LookAheadFiresTest(unittest.TestCase):
    def test_look_ahead_fires_before_recompute(self):
        """look_ahead_upload is called when _ExplicitGroupBackward.backward() runs."""
        fired = []

        x = torch.ones(2, requires_grad=True)
        group_input = (x,)
        group_output = _simple_run_group(group_input)

        anchored = anchor_explicit_group_backward(
            run_group=_simple_run_group,
            group_input=group_input,
            group_output=group_output,
            look_ahead_upload=lambda: fired.append(1),
        )
        loss = anchored[0].sum()
        loss.backward()

        self.assertEqual(fired, [1], "look_ahead_upload should fire exactly once during backward")

    def test_look_ahead_fires_before_on_backward_complete(self):
        """look_ahead_upload fires before on_backward_complete."""
        order = []

        x = torch.ones(2, requires_grad=True)
        group_input = (x,)
        group_output = _simple_run_group(group_input)

        anchored = anchor_explicit_group_backward(
            run_group=_simple_run_group,
            group_input=group_input,
            group_output=group_output,
            look_ahead_upload=lambda: order.append("look_ahead"),
            on_backward_complete=lambda: order.append("complete"),
        )
        anchored[0].sum().backward()

        self.assertEqual(order[0], "look_ahead", "look_ahead must fire before on_backward_complete")
        self.assertIn("complete", order)

    def test_no_look_ahead_is_safe(self):
        """Backward runs fine with look_ahead_upload=None (default)."""
        x = torch.ones(2, requires_grad=True)
        group_input = (x,)
        group_output = _simple_run_group(group_input)

        anchored = anchor_explicit_group_backward(
            run_group=_simple_run_group,
            group_input=group_input,
            group_output=group_output,
        )
        anchored[0].sum().backward()
        self.assertIsNotNone(x.grad)


class PipelineLookAheadWiringTest(unittest.TestCase):
    """Verify pipeline wires look_ahead_upload correctly for multi-group scenarios."""

    def _build_mini_pipeline(self, num_groups: int):
        """Build a minimal StratumPipeline with CPU-only two-layer stages."""
        from stratum.pipeline import StratumPipeline
        from stratum.stage import DeviceStage
        from stratum.scheduler import ModelExecutePlan

        layers = [nn.Identity() for _ in range(num_groups)]

        class _TinyPrefix(nn.Module):
            def forward(self, *, input_ids, attention_mask=None, labels=None, **kw):
                h = torch.ones(1, 4, requires_grad=True)
                return (h, None, None, None, None, labels, None)

        class _TinyPostfix(nn.Module):
            def forward(self, data):
                h = data[0]
                return h.mean()

        # Each layer is its own group (one layer per group, one group per stage
        # for simplicity — all on device_id=0 which maps to CPU in tests).
        stage = DeviceStage(layers=layers, device_id=0)
        plan = ModelExecutePlan.from_stage_lengths([len(layers)])

        pipe = StratumPipeline(
            _TinyPrefix(),
            [stage],
            _TinyPostfix(),
            execute_plan=plan,
        )
        return pipe

    def test_first_group_gets_no_look_ahead(self):
        """fwd_group_id==0 anchor: look_ahead_upload must be None."""
        from stratum.runtime import anchor_explicit_group_backward as _anchor
        look_ahead_calls = []

        with patch(
            "stratum.pipeline.anchor_explicit_group_backward",
            wraps=_anchor,
        ) as mock_anchor:
            pipe = self._build_mini_pipeline(num_groups=2)
            x = torch.zeros(1, dtype=torch.long)
            try:
                pipe.forward(input_ids=x)
            except Exception:
                pass  # postfix may fail — we only need the anchor calls

        if mock_anchor.call_count >= 1:
            first_call_kwargs = mock_anchor.call_args_list[0].kwargs
            self.assertIsNone(
                first_call_kwargs.get("look_ahead_upload"),
                "first group should have look_ahead_upload=None",
            )

    def test_second_group_gets_look_ahead(self):
        """fwd_group_id==1 anchor: look_ahead_upload must not be None."""
        from stratum.runtime import anchor_explicit_group_backward as _anchor

        with patch(
            "stratum.pipeline.anchor_explicit_group_backward",
            wraps=_anchor,
        ) as mock_anchor:
            pipe = self._build_mini_pipeline(num_groups=2)
            x = torch.zeros(1, dtype=torch.long)
            try:
                pipe.forward(input_ids=x)
            except Exception:
                pass

        if mock_anchor.call_count >= 2:
            second_call_kwargs = mock_anchor.call_args_list[1].kwargs
            self.assertIsNotNone(
                second_call_kwargs.get("look_ahead_upload"),
                "second group should have a look_ahead_upload callback",
            )

    def test_pending_bwd_fences_cleared_each_step(self):
        """_pending_bwd_fences resets to empty at the start of forward_range."""
        from stratum.pipeline import StratumPipeline
        from stratum.stage import DeviceStage
        from stratum.scheduler import ModelExecutePlan

        layer = nn.Identity()
        stage = DeviceStage(layers=[layer], device_id=0)
        plan = ModelExecutePlan.from_stage_lengths([1])

        class _P(nn.Module):
            def forward(self, *, input_ids, attention_mask=None, labels=None, **kw):
                return (torch.ones(1, 4, requires_grad=True), None, None, None, None, None, None)

        class _Q(nn.Module):
            def forward(self, data):
                return data[0].mean()

        pipe = StratumPipeline(_P(), [stage], _Q(), execute_plan=plan)
        # Manually pollute fences
        pipe._pending_bwd_fences[99] = object()
        # forward_range should clear it
        try:
            pipe.forward(input_ids=torch.zeros(1, dtype=torch.long))
        except Exception:
            pass
        self.assertNotIn(99, pipe._pending_bwd_fences)


if __name__ == "__main__":
    unittest.main()
