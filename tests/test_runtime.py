import unittest

import torch

from stratum.context import doing_recompute, get_recompute_data
from stratum.runtime import (
    anchor_explicit_group_backward,
    capture_backward_input,
    run_explicit_group_backward,
)
from stratum.timing import TimingRecorder


class ExplicitGroupBackwardTest(unittest.TestCase):
    def test_anchor_cuts_original_graph_and_recomputes_on_backward(self):
        calls = []
        hidden = torch.tensor([2.0], requires_grad=True)
        weight = torch.nn.Parameter(torch.tensor([3.0]))

        def run_group(input_data):
            calls.append("run")
            (x,) = input_data
            return (x * weight,)

        group_input = (hidden,)
        group_output = run_group(group_input)
        anchored = anchor_explicit_group_backward(
            run_group=run_group,
            group_input=group_input,
            group_output=group_output,
        )

        anchored[0].sum().backward()

        self.assertEqual(calls, ["run", "run"])
        self.assertTrue(torch.equal(hidden.grad, torch.tensor([3.0])))
        self.assertTrue(torch.equal(weight.grad, torch.tensor([2.0])))

    def test_anchor_preserves_nested_router_side_channel_gradients(self):
        hidden = torch.tensor([2.0], requires_grad=True)
        prior_router = torch.tensor([5.0], requires_grad=True)

        def run_group(input_data):
            hs, causal, pos_ids, pos_embeds, kwargs, labels, logits_to_keep = input_data
            out_kwargs = dict(kwargs)
            out_kwargs["_router_logits"] = list(kwargs["_router_logits"])
            return (
                hs * 2.0,
                causal,
                pos_ids,
                pos_embeds,
                out_kwargs,
                labels,
                logits_to_keep,
            )

        group_input = (
            hidden,
            None,
            None,
            None,
            {"_router_logits": [prior_router]},
            None,
            0,
        )
        group_output = run_group(group_input)
        anchored = anchor_explicit_group_backward(
            run_group=run_group,
            group_input=group_input,
            group_output=group_output,
        )

        loss = anchored[0].sum() + anchored[4]["_router_logits"][0].sum()
        loss.backward()

        self.assertTrue(torch.equal(hidden.grad, torch.tensor([2.0])))
        self.assertTrue(torch.equal(prior_router.grad, torch.tensor([1.0])))

    def test_anchor_only_replays_new_router_side_channel_tensors(self):
        calls = []
        hidden = torch.tensor([2.0], requires_grad=True)
        prior_router = torch.tensor([5.0], requires_grad=True)

        def run_group(input_data):
            calls.append("run")
            hs, causal, pos_ids, pos_embeds, kwargs, labels, logits_to_keep = input_data
            out_kwargs = dict(kwargs)
            out_kwargs["_router_logits"] = list(kwargs["_router_logits"])
            out_kwargs["_router_logits"].append(hs * 3.0)
            return (
                hs * 2.0,
                causal,
                pos_ids,
                pos_embeds,
                out_kwargs,
                labels,
                logits_to_keep,
            )

        group_input = (
            hidden,
            None,
            None,
            None,
            {"_router_logits": [prior_router]},
            None,
            0,
        )
        group_output = run_group(group_input)
        anchored = anchor_explicit_group_backward(
            run_group=run_group,
            group_input=group_input,
            group_output=group_output,
        )

        loss = (
            anchored[0].sum()
            + anchored[4]["_router_logits"][0].sum()
            + anchored[4]["_router_logits"][1].sum()
        )
        loss.backward()

        self.assertEqual(calls, ["run", "run"])
        self.assertTrue(torch.equal(hidden.grad, torch.tensor([5.0])))
        self.assertTrue(torch.equal(prior_router.grad, torch.tensor([1.0])))

    def test_anchor_runs_param_backward_without_input_requires_grad(self):
        hidden = torch.tensor([2.0])
        bias = torch.nn.Parameter(torch.tensor([4.0]))
        completed = []

        def run_group(input_data):
            (x,) = input_data
            return (x + bias,)

        group_input = (hidden,)
        group_output = run_group(group_input)
        anchored = anchor_explicit_group_backward(
            run_group=run_group,
            group_input=group_input,
            group_output=group_output,
            on_backward_complete=lambda: completed.append("done"),
        )

        anchored[0].sum().backward()

        self.assertTrue(torch.equal(bias.grad, torch.tensor([1.0])))
        self.assertEqual(completed, ["done"])

    def test_recomputes_group_and_returns_input_grads(self):
        hidden = torch.tensor([[1.0, 2.0]], requires_grad=True)
        bias = torch.tensor([[0.5, -0.5]])
        captured = capture_backward_input((hidden + 1.0, bias))

        def run_group(input_data):
            x, b = input_data
            (scale,) = get_recompute_data()
            return (x * scale + b,)

        result = run_explicit_group_backward(
            run_group=run_group,
            captured_input=captured,
            output_grads=(torch.ones_like(hidden),),
            recompute_data=(torch.full_like(hidden, 3.0),),
        )

        input_grad, bias_grad = result.input_grads
        self.assertTrue(torch.equal(input_grad, torch.full_like(hidden, 3.0)))
        self.assertIsNone(bias_grad)
        self.assertIsNone(hidden.grad)

    def test_capture_records_original_tensor_device_and_requires_grad(self):
        hidden = torch.tensor([[1.0, 2.0]], requires_grad=True)
        bias = torch.tensor([[0.5, -0.5]])

        captured = capture_backward_input((hidden, bias), offload_to_cpu=True)
        restored_hidden, restored_bias = captured.restore()

        self.assertEqual(restored_hidden.device, hidden.device)
        self.assertEqual(restored_bias.device, bias.device)
        self.assertTrue(restored_hidden.requires_grad)
        self.assertFalse(restored_bias.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_offloaded_capture_restores_cuda_leaf_for_recompute(self):
        hidden = torch.tensor([[1.0, 2.0]], device="cuda", requires_grad=True)
        captured = capture_backward_input((hidden,), offload_to_cpu=True)

        self.assertEqual(captured.flat[0].device.type, "cpu")
        (restored,) = captured.restore()

        self.assertEqual(restored.device.type, "cuda")
        self.assertTrue(restored.requires_grad)
        self.assertTrue(restored.is_leaf)

    def test_records_recompute_and_backward_timing(self):
        hidden = torch.tensor([2.0], requires_grad=True)
        captured = capture_backward_input((hidden,))
        recorder = TimingRecorder(use_cuda_events=False)

        def run_group(input_data):
            (x,) = input_data
            return (x * x,)

        run_explicit_group_backward(
            run_group=run_group,
            captured_input=captured,
            output_grads=(torch.ones_like(hidden),),
            timing_recorder=recorder,
            timing_fields={"fwd_group_id": 4, "layer_start": 7, "layer_stop": 8},
        )

        records = [(record["name"], record["fwd_group_id"]) for record in recorder.records]
        self.assertEqual(
            records,
            [("stage_recompute", 4), ("stage_backward_explicit", 4)],
        )

    def test_default_recompute_does_not_set_recompute_context(self):
        hidden = torch.tensor([2.0], requires_grad=True)
        captured = capture_backward_input((hidden,))
        phases = []

        def run_group(input_data):
            phases.append(doing_recompute())
            (x,) = input_data
            return (x * 2.0,)

        run_explicit_group_backward(
            run_group=run_group,
            captured_input=captured,
            output_grads=(torch.ones_like(hidden),),
        )

        self.assertEqual(phases, [False])

    def test_scalar_output_allows_missing_grad(self):
        hidden = torch.tensor([2.0], requires_grad=True)
        captured = capture_backward_input((hidden,))

        def run_group(input_data):
            (x,) = input_data
            return x.sum()

        result = run_explicit_group_backward(
            run_group=run_group,
            captured_input=captured,
            output_grads=None,
        )

        (input_grad,) = result.input_grads
        self.assertTrue(torch.equal(input_grad, torch.ones_like(hidden)))

    def test_rejects_mismatched_output_grad_structure(self):
        hidden = torch.tensor([2.0], requires_grad=True)
        captured = capture_backward_input((hidden,))

        def run_group(input_data):
            (x,) = input_data
            return (x,)

        with self.assertRaisesRegex(ValueError, "same pytree structure"):
            run_explicit_group_backward(
                run_group=run_group,
                captured_input=captured,
                output_grads={"hidden": torch.ones_like(hidden)},
            )


if __name__ == "__main__":
    unittest.main()
