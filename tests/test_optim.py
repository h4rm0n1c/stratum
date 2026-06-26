import unittest

import torch

from stratum.attribute import ParamAttribute
from stratum.grad_scaler import GradScaler as CPUOffloadGradScaler
from stratum.optim import PerDeviceOptimizer


class PerDeviceOptimizerTests(unittest.TestCase):
    def test_cpu_offload_step_updates_live_parameter(self):
        module = torch.nn.Linear(2, 1, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=True,
        )

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        self.assertFalse(torch.equal(module.weight.detach(), original))
        attr = ParamAttribute.get(module.weight)
        self.assertIsNotNone(attr)
        self.assertTrue(torch.allclose(module.weight.detach(), attr.optim.detach()))

    def test_cpu_offload_queued_step_updates_before_synchronize_returns(self):
        module = torch.nn.Linear(2, 1, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=True,
        )

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step(async_step=True)
        optimizer.synchronize()

        self.assertFalse(torch.equal(module.weight.detach(), original))
        attr = ParamAttribute.get(module.weight)
        self.assertIsNotNone(attr)
        self.assertTrue(torch.allclose(module.weight.detach(), attr.optim.detach()))

    def test_cpu_offload_waits_grad_ready_events_before_grad_move(self):
        class FakeEvent:
            def __init__(self):
                self.synchronized = False

            def synchronize(self):
                self.synchronized = True

        module = torch.nn.Linear(2, 1, bias=False)
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=True,
        )
        event = FakeEvent()
        optimizer._record_grad_ready_events = lambda: [event]

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step(async_step=True)
        optimizer.synchronize()

        self.assertTrue(event.synchronized)

    def test_cpu_offload_zero_grad_clears_stale_cpu_grad(self):
        class UsesOneParam(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.used = torch.nn.Parameter(torch.tensor([1.0]))
                self.unused = torch.nn.Parameter(torch.tensor([1.0]))

            def forward(self):
                return self.used.sum()

        module = UsesOneParam()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=True,
        )

        optimizer.zero_grad()
        module().backward()
        optimizer.step()
        optimizer.zero_grad()

        unused_attr = ParamAttribute.get(module.unused)
        self.assertIsNotNone(unused_attr)
        self.assertIsNone(unused_attr.optim.grad)

    def test_torch_grad_scaler_records_successful_step(self):
        module = torch.nn.Linear(2, 1, bias=False)
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=False,
        )
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        optimizer.zero_grad()
        loss = scaler.scale(module(torch.ones(1, 2)).sum())
        loss.backward()
        optimizer.step(scaler=scaler)

        self.assertFalse(optimizer.last_step_was_skipped())

    def test_torch_grad_scaler_records_skipped_step(self):
        module = torch.nn.Linear(2, 1, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=False,
        )
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        optimizer.zero_grad()
        loss = scaler.scale(module(torch.ones(1, 2)).sum())
        loss.backward()
        module.weight.grad.fill_(float("inf"))
        optimizer.step(scaler=scaler)

        self.assertTrue(optimizer.last_step_was_skipped())
        self.assertTrue(torch.equal(module.weight.detach(), original))

    def test_cpu_offload_grad_scaler_records_async_skipped_step(self):
        module = torch.nn.Linear(2, 1, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.1,
            weight_decay=0.0,
            scheduler="constant",
            cpu_offload=True,
        )
        scaler = CPUOffloadGradScaler(enabled=True)

        optimizer.zero_grad()
        loss = scaler.scale(module(torch.ones(1, 2)).sum())
        loss.backward()
        module.weight.grad.fill_(float("inf"))
        optimizer.step(async_step=True, scaler=scaler)
        optimizer.synchronize()

        self.assertTrue(optimizer.last_step_was_skipped())
        self.assertTrue(torch.equal(module.weight.detach(), original))


if __name__ == "__main__":
    unittest.main()
