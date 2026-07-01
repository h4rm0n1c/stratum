import unittest

import torch

from stratum.attribute import ParamAttribute
from stratum.grad_scaler import GradScaler as CPUOffloadGradScaler
from stratum.muon import MuonAdamW
from stratum.optim import PerDeviceOptimizer
from stratum.qk_clip import (
    QK_CLIP_ENABLED_ATTR,
    QK_CLIP_STAT_MODE_ATTR,
    QK_CLIP_STATS_ATTR,
    flash_attention_with_qk_clip_stats,
)


class PerDeviceOptimizerTests(unittest.TestCase):
    def test_muon_optimizer_updates_2d_and_falls_back_for_1d(self):
        matrix = torch.nn.Parameter(torch.ones(2, 3))
        vector = torch.nn.Parameter(torch.ones(3))
        opt = MuonAdamW([matrix, vector], lr=0.01, weight_decay=0.0)

        matrix.grad = torch.ones_like(matrix)
        vector.grad = torch.ones_like(vector)
        opt.step()

        matrix_state = opt.state[matrix]
        vector_state = opt.state[vector]
        self.assertIn("momentum_buffer", matrix_state)
        self.assertNotIn("exp_avg", matrix_state)
        self.assertIn("exp_avg", vector_state)
        self.assertIn("exp_avg_sq", vector_state)
        self.assertFalse(torch.equal(matrix.detach(), torch.ones_like(matrix)))
        self.assertFalse(torch.equal(vector.detach(), torch.ones_like(vector)))

    def test_muon_optimizer_updates_3d_as_batched_matrices(self):
        expert_weight = torch.nn.Parameter(torch.ones(4, 2, 3))
        opt = MuonAdamW([expert_weight], lr=0.01, weight_decay=0.0)

        expert_weight.grad = torch.ones_like(expert_weight)
        opt.step()

        state = opt.state[expert_weight]
        self.assertIn("momentum_buffer", state)
        self.assertEqual(state["momentum_buffer"].shape, expert_weight.shape)
        self.assertFalse(torch.equal(expert_weight.detach(), torch.ones_like(expert_weight)))

    def test_muon_optimizer_forced_adamw_matrix_param(self):
        matrix = torch.nn.Parameter(torch.ones(2, 3))
        opt = MuonAdamW(
            [matrix],
            lr=0.01,
            weight_decay=0.0,
            adamw_param_ids={id(matrix)},
        )

        matrix.grad = torch.ones_like(matrix)
        opt.step()

        state = opt.state[matrix]
        self.assertIn("exp_avg", state)
        self.assertIn("exp_avg_sq", state)
        self.assertNotIn("momentum_buffer", state)
        self.assertFalse(torch.equal(matrix.detach(), torch.ones_like(matrix)))

    def test_per_device_optimizer_muon_defaults_qk_params_to_adamw(self):
        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(2, 2, bias=False)
                self.k_proj = torch.nn.Linear(2, 2, bias=False)
                self.v_proj = torch.nn.Linear(2, 2, bias=False)

            def forward(self, x):
                return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

        module = TinyAttention()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            muon_qk_mode="adamw",
            cpu_offload=False,
        )

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        opt = optimizer.optimizers[0]
        self.assertIsNotNone(opt)
        self.assertIn("exp_avg", opt.state[module.q_proj.weight])
        self.assertIn("exp_avg", opt.state[module.k_proj.weight])
        self.assertIn("momentum_buffer", opt.state[module.v_proj.weight])
        self.assertEqual(len(optimizer.muon_adamw_param_names[0]), 2)

    def test_per_device_optimizer_muon_qk_clip_scales_qk_params(self):
        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 2
                self.q_proj = torch.nn.Linear(2, 2, bias=False)
                self.k_proj = torch.nn.Linear(2, 2, bias=False)
                self.v_proj = torch.nn.Linear(2, 2, bias=False)

        module = TinyAttention()
        for param in module.parameters():
            torch.nn.init.ones_(param)
        module._stratum_qk_clip_smax = torch.tensor([400.0])
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.0,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=False,
            muon_qk_mode="clip",
            muon_qk_clip_threshold=100.0,
        )

        for param in module.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()

        opt = optimizer.optimizers[0]
        self.assertIsNotNone(opt)
        self.assertIn("momentum_buffer", opt.state[module.q_proj.weight])
        self.assertIn("momentum_buffer", opt.state[module.k_proj.weight])
        self.assertIn("momentum_buffer", opt.state[module.v_proj.weight])
        self.assertTrue(torch.allclose(module.q_proj.weight, torch.full_like(module.q_proj.weight, 0.5)))
        self.assertTrue(torch.allclose(module.k_proj.weight, torch.full_like(module.k_proj.weight, 0.5)))
        self.assertTrue(torch.allclose(module.v_proj.weight, torch.ones_like(module.v_proj.weight)))
        self.assertEqual(optimizer.last_qk_clip_stats[0]["heads"], 1)
        self.assertAlmostEqual(float(optimizer.last_qk_clip_stats[0]["min_gamma"]), 0.25)
        self.assertEqual(optimizer.last_qk_clip_stats[0]["bound_layers"], 1)

    def test_per_device_optimizer_muonclip_selects_qk_clip(self):
        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 2
                self.q_proj = torch.nn.Linear(2, 2, bias=False)
                self.k_proj = torch.nn.Linear(2, 2, bias=False)

        module = TinyAttention()
        for param in module.parameters():
            torch.nn.init.ones_(param)
            param.grad = torch.ones_like(param)
        module._stratum_qk_clip_smax = torch.tensor([400.0])
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.0,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muonclip",
            cpu_offload=False,
            muon_qk_mode="adamw",
            muon_qk_clip_threshold=100.0,
        )

        self.assertTrue(getattr(module, QK_CLIP_ENABLED_ATTR))
        optimizer.step()

        opt = optimizer.optimizers[0]
        self.assertIsNotNone(opt)
        self.assertIn("momentum_buffer", opt.state[module.q_proj.weight])
        self.assertIn("momentum_buffer", opt.state[module.k_proj.weight])
        self.assertTrue(torch.allclose(module.q_proj.weight, torch.full_like(module.q_proj.weight, 0.5)))
        self.assertTrue(torch.allclose(module.k_proj.weight, torch.full_like(module.k_proj.weight, 0.5)))
        self.assertEqual(optimizer.muon_qk_mode, "clip")
        self.assertEqual(optimizer.last_qk_clip_stats[0]["heads"], 1)

    def test_per_device_optimizer_muonclip_scales_fused_qkv_qk_rows_only(self):
        class FusedAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 1
                self.num_key_value_heads = 2
                self.qkv = torch.nn.Linear(2, 6, bias=False)

        module = FusedAttention()
        torch.nn.init.ones_(module.qkv.weight)
        module.qkv.weight.grad = torch.ones_like(module.qkv.weight)
        module._stratum_qk_clip_smax = torch.tensor([400.0, 100.0])
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.0,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muonclip",
            cpu_offload=False,
            muon_qk_clip_threshold=100.0,
        )

        self.assertTrue(getattr(module, QK_CLIP_ENABLED_ATTR))
        optimizer.step()

        expected = torch.ones_like(module.qkv.weight)
        expected[0].mul_(0.5)  # Q head 0
        expected[2].mul_(0.5)  # K head 0
        self.assertTrue(torch.allclose(module.qkv.weight, expected))
        self.assertEqual(optimizer.last_qk_clip_stats[0]["heads"], 1)

    def test_per_device_optimizer_muon_qk_mode_can_use_muon(self):
        module = torch.nn.Module()
        module.q_proj = torch.nn.Linear(2, 2, bias=False)
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=False,
            muon_qk_mode="muon",
        )

        optimizer.zero_grad()
        module.q_proj(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        opt = optimizer.optimizers[0]
        self.assertIsNotNone(opt)
        self.assertIn("momentum_buffer", opt.state[module.q_proj.weight])
        self.assertEqual(optimizer.muon_adamw_param_names[0], [])

    def test_per_device_optimizer_muon_updates_live_parameters(self):
        module = torch.nn.Linear(2, 2, bias=True)
        original_weight = module.weight.detach().clone()
        original_bias = module.bias.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=False,
        )

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        opt = optimizer.optimizers[0]
        self.assertIsNotNone(opt)
        self.assertIn("momentum_buffer", opt.state[module.weight])
        self.assertIn("exp_avg", opt.state[module.bias])
        self.assertFalse(torch.equal(module.weight.detach(), original_weight))
        self.assertFalse(torch.equal(module.bias.detach(), original_bias))

    def test_cpu_offload_muon_qk_mode_adamw_routes_qk_params_to_adamw(self):
        module = torch.nn.Module()
        module.q_proj = torch.nn.Linear(2, 2, bias=False)
        module.v_proj = torch.nn.Linear(2, 2, bias=False)
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=True,
            muon_qk_mode="adamw",
        )

        optimizer.zero_grad()
        (module.q_proj(torch.ones(1, 2)) + module.v_proj(torch.ones(1, 2))).sum().backward()
        optimizer.step()

        q_attr = ParamAttribute.get(module.q_proj.weight)
        v_attr = ParamAttribute.get(module.v_proj.weight)
        self.assertIsNotNone(q_attr)
        self.assertIsNotNone(v_attr)
        opt = optimizer.optimizers[0]
        self.assertIsNotNone(opt)
        self.assertIn("exp_avg", opt.state[q_attr.optim])
        self.assertIn("momentum_buffer", opt.state[v_attr.optim])

    def test_cpu_offload_muon_qk_clip_updates_optim_copy_and_live_param(self):
        module = torch.nn.Module()
        module.head_dim = 2
        module.q_proj = torch.nn.Linear(2, 2, bias=False)
        module.k_proj = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.ones_(module.q_proj.weight)
        torch.nn.init.ones_(module.k_proj.weight)
        module._stratum_qk_clip_smax = torch.tensor([400.0])
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.0,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=True,
            muon_qk_mode="clip",
            muon_qk_clip_threshold=100.0,
        )

        optimizer.zero_grad()
        (module.q_proj(torch.ones(1, 2)) + module.k_proj(torch.ones(1, 2))).sum().backward()
        optimizer.step()

        q_attr = ParamAttribute.get(module.q_proj.weight)
        k_attr = ParamAttribute.get(module.k_proj.weight)
        self.assertIsNotNone(q_attr)
        self.assertIsNotNone(k_attr)
        self.assertTrue(torch.allclose(q_attr.optim, torch.full_like(q_attr.optim, 0.5)))
        self.assertTrue(torch.allclose(k_attr.optim, torch.full_like(k_attr.optim, 0.5)))
        self.assertTrue(torch.allclose(module.q_proj.weight, torch.full_like(module.q_proj.weight, 0.5)))
        self.assertTrue(torch.allclose(module.k_proj.weight, torch.full_like(module.k_proj.weight, 0.5)))

    def test_qk_clip_flash_helper_records_exact_max_logits(self):
        module = torch.nn.Module()
        setattr(module, QK_CLIP_ENABLED_ATTR, True)
        setattr(module, QK_CLIP_STAT_MODE_ATTR, "exact_flash")
        q = torch.ones(1, 2, 1, 2)
        k = torch.ones(1, 2, 1, 2)
        v = torch.ones(1, 2, 1, 2)

        class Meta:
            max_logits = torch.tensor([125.0])

        def fake_flash(q, k, v, **kwargs):
            self.assertTrue(kwargs["return_max_logits"])
            return q + v, Meta()

        out = flash_attention_with_qk_clip_stats(
            module,
            "fake_flash",
            fake_flash,
            q,
            k,
            v,
            query_states=q.transpose(1, 2),
            key_states=k.transpose(1, 2),
            scaling=1.0,
            causal=True,
        )

        self.assertTrue(torch.equal(out, q + v))
        self.assertTrue(torch.equal(getattr(module, QK_CLIP_STATS_ATTR), torch.tensor([125.0])))

    def test_qk_clip_flash_helper_auto_falls_back_to_bound(self):
        module = torch.nn.Module()
        setattr(module, QK_CLIP_ENABLED_ATTR, True)
        setattr(module, QK_CLIP_STAT_MODE_ATTR, "auto")
        q = torch.ones(1, 2, 1, 2)
        k = torch.ones(1, 2, 1, 2)
        v = torch.ones(1, 2, 1, 2)

        def fake_flash(q, k, v, **kwargs):
            if "return_max_logits" in kwargs:
                raise TypeError("unexpected keyword argument 'return_max_logits'")
            return q + v

        out = flash_attention_with_qk_clip_stats(
            module,
            "fake_flash",
            fake_flash,
            q,
            k,
            v,
            query_states=q.transpose(1, 2),
            key_states=k.transpose(1, 2),
            scaling=1.0,
            causal=True,
        )

        self.assertTrue(torch.equal(out, q + v))
        self.assertTrue(torch.allclose(getattr(module, QK_CLIP_STATS_ATTR), torch.tensor([2.0])))

    def test_qk_clip_flash_helper_exact_mode_requires_backend_support(self):
        module = torch.nn.Module()
        setattr(module, QK_CLIP_ENABLED_ATTR, True)
        setattr(module, QK_CLIP_STAT_MODE_ATTR, "exact_flash")
        q = torch.ones(1, 2, 1, 2)
        k = torch.ones(1, 2, 1, 2)
        v = torch.ones(1, 2, 1, 2)

        def fake_flash(q, k, v, **kwargs):
            if "return_max_logits" in kwargs:
                raise TypeError("unexpected keyword argument 'return_max_logits'")
            return q + v

        with self.assertRaisesRegex(RuntimeError, "return_max_logits"):
            flash_attention_with_qk_clip_stats(
                module,
                "fake_flash",
                fake_flash,
                q,
                k,
                v,
                query_states=q.transpose(1, 2),
                key_states=k.transpose(1, 2),
                scaling=1.0,
                causal=True,
            )

    def test_cpu_offload_muon_step_updates_live_parameter(self):
        module = torch.nn.Linear(2, 2, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=True,
        )

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        self.assertFalse(torch.equal(module.weight.detach(), original))
        attr = ParamAttribute.get(module.weight)
        self.assertIsNotNone(attr)
        self.assertIn("momentum_buffer", optimizer.optimizers[0].state[attr.optim])
        self.assertTrue(torch.allclose(module.weight.detach(), attr.optim.detach()))

    def test_cpu_offload_muon_async_step_updates_before_synchronize_returns(self):
        module = torch.nn.Linear(2, 2, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
            cpu_offload=True,
        )

        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step(async_step=True)
        optimizer.synchronize()

        self.assertFalse(torch.equal(module.weight.detach(), original))

    def test_cpu_offload_grad_scaler_records_async_muon_skipped_step(self):
        module = torch.nn.Linear(2, 2, bias=False)
        original = module.weight.detach().clone()
        optimizer = PerDeviceOptimizer(
            {0: [module]},
            lr=0.01,
            weight_decay=0.0,
            scheduler="constant",
            optimizer="muon",
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
