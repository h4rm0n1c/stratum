import unittest

import torch
import torch.nn as nn

from stratum.model.mlp_opt import (
    MemoryFlatFrozenMLP,
    MemoryFlatFrozenModule,
    TokenChunkedModule,
    enable_decoder_mlp_token_chunking,
    enable_memory_flat_frozen_mlp,
)
from stratum.upload import _build_ckpt_key_fn


class RecordingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states):
        self.calls.append(tuple(hidden_states.shape))
        return hidden_states + 1


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


class FakeLayer(nn.Module):
    def __init__(self, attr_name: str, module: nn.Module):
        super().__init__()
        setattr(self, attr_name, module)


class FakeCore(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)


class FakeMoe(nn.Module):
    def __init__(self, *, trainable: bool = False):
        super().__init__()
        self.gate_exps = nn.Parameter(torch.empty(1), requires_grad=trainable)

    def forward(self, hidden_states):
        return hidden_states * 2


def _freeze(module):
    for param in module.parameters():
        param.requires_grad_(False)
    return module


class TestMlpOptimizations(unittest.TestCase):
    def assert_dense_matches_unwrapped(self, module):
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
        self.assert_dense_matches_unwrapped(StandardDenseMlp())

    def test_lfm_ffn_projection_names_match_unwrapped_mlp(self):
        self.assert_dense_matches_unwrapped(FfnDenseMlp())

    def test_token_chunked_module_chunks_packed_2d_tokens(self):
        module = RecordingModule()
        wrapped = TokenChunkedModule(module, token_chunk_size=4)

        x = torch.zeros(10, 3)
        y = wrapped(x)

        self.assertEqual(tuple(y.shape), (10, 3))
        self.assertEqual(module.calls, [(4, 3), (4, 3), (2, 3)])

    def test_token_chunking_patches_lfm_feed_forward_attr(self):
        feed_forward = RecordingModule()
        model = FakeCore([FakeLayer("feed_forward", feed_forward)])

        patched = enable_decoder_mlp_token_chunking(model, token_chunk_size=4)

        self.assertEqual(patched, 1)
        self.assertIsInstance(model.model.layers[0].feed_forward, TokenChunkedModule)

    def test_memory_flat_uses_generic_wrapper_for_frozen_moe_feed_forward(self):
        model = FakeCore([FakeLayer("feed_forward", FakeMoe())])

        patched = enable_memory_flat_frozen_mlp(model, token_chunk_size=4)

        self.assertEqual(patched, 1)
        self.assertIsInstance(model.model.layers[0].feed_forward, MemoryFlatFrozenModule)

    def test_memory_flat_falls_back_to_token_chunking_for_trainable_moe_feed_forward(self):
        model = FakeCore([FakeLayer("feed_forward", FakeMoe(trainable=True))])

        patched = enable_memory_flat_frozen_mlp(model, token_chunk_size=4)

        self.assertEqual(patched, 1)
        self.assertIsInstance(model.model.layers[0].feed_forward, TokenChunkedModule)

    def test_generic_memory_flat_module_matches_forward_and_input_grad(self):
        module = FakeMoe()
        wrapped = MemoryFlatFrozenModule(module, token_chunk_size=4)
        x = torch.randn(10, 3, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)

        y = wrapped(x)
        ref_y = module(ref_x)
        y.sum().backward()
        ref_y.sum().backward()

        self.assertTrue(torch.equal(y, ref_y))
        self.assertTrue(torch.equal(x.grad, ref_x.grad))

    def test_wrapper_names_are_canonicalized_for_checkpoint_lookup(self):
        feed_forward = nn.Module()
        feed_forward.w1 = nn.Linear(3, 3, bias=False)
        model = FakeCore([FakeLayer("feed_forward", feed_forward)])

        enable_decoder_mlp_token_chunking(model, token_chunk_size=4)

        ckpt_keys = _build_ckpt_key_fn(model)
        keys = ckpt_keys("model.layers.0.feed_forward.module.w1.weight")
        self.assertIn("model.layers.0.feed_forward.w1.weight", keys)
        self.assertNotIn("model.layers.0.feed_forward.module.w1.weight", keys)


if __name__ == "__main__":
    unittest.main()
