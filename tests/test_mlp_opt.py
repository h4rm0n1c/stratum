import unittest

import torch
import torch.nn as nn

from stratum.model.mlp_opt import MemoryFlatFrozenMLP


class StandardDenseMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class FfnDenseMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_gate_proj = nn.Linear(4, 6, bias=False)
        self.ffn_up_proj = nn.Linear(4, 6, bias=False)
        self.ffn_down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        return self.ffn_down_proj(
            self.act_fn(self.ffn_gate_proj(hidden_states)) * self.ffn_up_proj(hidden_states)
        )


def _freeze(module):
    for param in module.parameters():
        param.requires_grad_(False)
    return module


class MemoryFlatFrozenMLPTests(unittest.TestCase):
    def assert_matches_unwrapped(self, module):
        torch.manual_seed(1234)
        module = _freeze(module)
        wrapped = MemoryFlatFrozenMLP(module, token_chunk_size=3)

        expected_input = torch.randn(2, 7, 4, requires_grad=True)
        actual_input = expected_input.detach().clone().requires_grad_(True)

        expected = module(expected_input)
        actual = wrapped(actual_input)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        expected.square().sum().backward()
        actual.square().sum().backward()
        self.assertTrue(
            torch.allclose(actual_input.grad, expected_input.grad, atol=1e-6, rtol=1e-6)
        )

    def test_standard_projection_names_match_unwrapped_mlp(self):
        self.assert_matches_unwrapped(StandardDenseMlp())

    def test_lfm_ffn_projection_names_match_unwrapped_mlp(self):
        self.assert_matches_unwrapped(FfnDenseMlp())

    def test_moe_experts_are_rejected_explicitly(self):
        class MoeLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_exps = nn.ModuleList()

        with self.assertRaisesRegex(TypeError, "MoE"):
            MemoryFlatFrozenMLP(MoeLike(), token_chunk_size=2)


if __name__ == "__main__":
    unittest.main()
