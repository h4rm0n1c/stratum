import sys
import types
import unittest

import torch
import torch.nn as nn

from stratum.moe import load_balancing_loss_func, patch_moe_block_for_router_logits, pop_router_logits


class MoePatchTests(unittest.TestCase):
    def test_each_patched_block_keeps_its_own_forward(self):
        class Lfm2MoeSparseMoeBlock(nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset
                self.gate = nn.Linear(3, 2, bias=False)

            def forward(self, hidden_states):
                return hidden_states + self.offset

        module_name = "transformers.models.lfm2_moe.modeling_lfm2_moe"
        package_name = "transformers.models.lfm2_moe"
        old_module = sys.modules.get(module_name)
        old_package = sys.modules.get(package_name)
        fake_package = types.ModuleType(package_name)
        fake_module = types.ModuleType(module_name)
        fake_module.Lfm2MoeSparseMoeBlock = Lfm2MoeSparseMoeBlock
        sys.modules[package_name] = fake_package
        sys.modules[module_name] = fake_module
        try:
            first = Lfm2MoeSparseMoeBlock(offset=1.0)
            second = Lfm2MoeSparseMoeBlock(offset=2.0)
            with torch.no_grad():
                first.gate.weight.fill_(1.0)
                second.gate.weight.fill_(2.0)

            model = nn.ModuleList([first, second])
            self.assertEqual(patch_moe_block_for_router_logits(model), 2)

            hidden = torch.ones(1, 2, 3)
            first_hidden = first(hidden)
            second_hidden = second(hidden)
            first_router, = pop_router_logits(first)
            second_router, = pop_router_logits(second)

            self.assertTrue(torch.equal(first_hidden, hidden + 1.0))
            self.assertTrue(torch.equal(second_hidden, hidden + 2.0))
            self.assertTrue(torch.equal(first_router, torch.full((2, 2), 3.0)))
            self.assertTrue(torch.equal(second_router, torch.full((2, 2), 6.0)))
        finally:
            if old_package is None:
                sys.modules.pop(package_name, None)
            else:
                sys.modules[package_name] = old_package
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module

    def test_qwen_style_tuple_gate_output_uses_logits_tensor(self):
        class TupleGate(nn.Module):
            def forward(self, hidden_states):
                logits = hidden_states[:, :2]
                scores = torch.ones_like(logits)
                indices = torch.zeros(hidden_states.shape[0], 1, dtype=torch.long)
                return logits, scores, indices

        class Lfm2MoeSparseMoeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = TupleGate()

            def forward(self, hidden_states):
                return hidden_states

        module_name = "transformers.models.lfm2_moe.modeling_lfm2_moe"
        package_name = "transformers.models.lfm2_moe"
        old_module = sys.modules.get(module_name)
        old_package = sys.modules.get(package_name)
        fake_package = types.ModuleType(package_name)
        fake_module = types.ModuleType(module_name)
        fake_module.Lfm2MoeSparseMoeBlock = Lfm2MoeSparseMoeBlock
        sys.modules[package_name] = fake_package
        sys.modules[module_name] = fake_module
        try:
            block = Lfm2MoeSparseMoeBlock()
            self.assertEqual(patch_moe_block_for_router_logits(block), 1)
            hidden = torch.randn(1, 3, 4)
            output = block(hidden)
            router_logits, = pop_router_logits(block)
            self.assertTrue(torch.equal(output, hidden))
            self.assertTrue(torch.equal(router_logits, hidden.reshape(-1, 4)[:, :2]))
        finally:
            if old_package is None:
                sys.modules.pop(package_name, None)
            else:
                sys.modules[package_name] = old_package
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module

    def test_load_balancing_loss_matches_transformers_formula_and_has_grad(self):
        gate_logits = (
            torch.tensor(
                [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0]],
                requires_grad=True,
            ),
            torch.tensor(
                [[2.0, 0.5, 1.0], [0.0, 1.0, 4.0], [1.0, 3.0, 0.0]],
                requires_grad=True,
            ),
        )
        loss = load_balancing_loss_func(gate_logits, num_experts=3, top_k=2)

        concatenated = torch.cat(gate_logits, dim=0)
        routing_weights = torch.softmax(concatenated, dim=-1)
        _, selected = torch.topk(routing_weights, 2, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected, num_classes=3)
        tokens_per_expert = expert_mask.float().mean(dim=0)
        router_prob_per_expert = routing_weights.mean(dim=0)
        expected = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * 3

        self.assertTrue(torch.allclose(loss, expected, atol=1e-6, rtol=1e-6))
        loss.backward()
        self.assertIsNotNone(gate_logits[0].grad)
        self.assertIsNotNone(gate_logits[1].grad)


if __name__ == "__main__":
    unittest.main()
