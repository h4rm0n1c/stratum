import sys
import types
import unittest

import torch
import torch.nn as nn

from stratum.model.blocked_loss import BlockedPostfixCausalLMLoss
from stratum.moe import load_balancing_loss_func


def _install_fake_transformers():
    transformers = sys.modules.setdefault("transformers", types.ModuleType("transformers"))
    models = sys.modules.setdefault("transformers.models", types.ModuleType("transformers.models"))
    setattr(transformers, "models", models)

    modeling_outputs = types.ModuleType("transformers.modeling_outputs")

    class CausalLMOutputWithPast:
        def __init__(self, loss=None, logits=None, **kwargs):
            self.loss = loss
            self.logits = logits
            for key, value in kwargs.items():
                setattr(self, key, value)

    modeling_outputs.CausalLMOutputWithPast = CausalLMOutputWithPast
    sys.modules["transformers.modeling_outputs"] = modeling_outputs

    masking_utils = types.ModuleType("transformers.masking_utils")
    masking_utils.create_causal_mask = lambda *args, **kwargs: None
    masking_utils.create_sliding_window_causal_mask = lambda *args, **kwargs: None
    sys.modules["transformers.masking_utils"] = masking_utils

    llama_pkg = types.ModuleType("transformers.models.llama")
    llama_mod = types.ModuleType("transformers.models.llama.modeling_llama")
    llama_mod.repeat_kv = lambda hidden, n_rep: hidden
    sys.modules["transformers.models.llama"] = llama_pkg
    sys.modules["transformers.models.llama.modeling_llama"] = llama_mod

    lfm_pkg = types.ModuleType("transformers.models.lfm2_moe")
    lfm_mod = types.ModuleType("transformers.models.lfm2_moe.modeling_lfm2_moe")

    class Lfm2MoeAttention(nn.Module):
        pass

    lfm_mod.Lfm2MoeAttention = Lfm2MoeAttention
    lfm_mod.apply_rotary_pos_emb = lambda *args, **kwargs: None
    lfm_mod.is_fast_path_available = False
    sys.modules["transformers.models.lfm2_moe"] = lfm_pkg
    sys.modules["transformers.models.lfm2_moe.modeling_lfm2_moe"] = lfm_mod

    qwen_pkg = types.ModuleType("transformers.models.qwen3_5")
    qwen_mod = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")

    class Qwen3_5Attention(nn.Module):
        pass

    qwen_mod.Qwen3_5Attention = Qwen3_5Attention
    qwen_mod.apply_rotary_pos_emb = lambda *args, **kwargs: None
    sys.modules["transformers.models.qwen3_5"] = qwen_pkg
    sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"] = qwen_mod


_install_fake_transformers()
from stratum.model.lfm25 import LFM25ForCausalLMPostfix, LFM25ForCausalLMWrappedLayer  # noqa: E402
from stratum.model.qwen35 import Qwen35ForCausalLMPostfix, Qwen35ForCausalLMWrappedLayer  # noqa: E402


class _CoreModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.model = types.SimpleNamespace(
            norm=nn.LayerNorm(hidden_size),
            output_norm=nn.LayerNorm(hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output = self.lm_head
        self.config = types.SimpleNamespace(
            vocab_size=vocab_size,
            num_experts=3,
            num_experts_per_tok=2,
        )
        for param in self.model.norm.parameters():
            param.requires_grad_(False)
        for param in self.model.output_norm.parameters():
            param.requires_grad_(False)
        for param in self.lm_head.parameters():
            param.requires_grad_(False)


class PostfixAuxLossTests(unittest.TestCase):
    def _exercise(self, postfix_cls, norm_attr):
        torch.manual_seed(2026)
        hidden_size = 5
        vocab_size = 11
        core = _CoreModel(hidden_size, vocab_size)
        norm = getattr(core.model, norm_attr)
        hidden = torch.randn(1, 6, hidden_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (1, 6))
        router_logits = (
            torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]], requires_grad=True),
            torch.tensor([[2.0, 0.5, 1.0], [0.0, 1.0, 4.0]], requires_grad=True),
        )
        coef = 0.25

        postfix = postfix_cls(
            core,
            postfix_loss_token_chunk_size=2,
            router_aux_loss_coef=coef,
        )
        output = postfix(
            (
                hidden,
                None,
                None,
                None,
                {"_router_logits": list(router_logits)},
                labels,
                0,
            )
        )

        base_loss = BlockedPostfixCausalLMLoss.apply(
            hidden,
            labels,
            norm,
            core.lm_head,
            vocab_size,
            2,
            -100,
            False,
            False,
        )
        aux_loss = load_balancing_loss_func(router_logits, num_experts=3, top_k=2)
        expected = base_loss + coef * aux_loss
        self.assertTrue(torch.allclose(output.loss, expected, atol=1e-6, rtol=1e-6))

    def test_lfm_blocked_postfix_still_adds_router_aux_loss(self):
        self._exercise(LFM25ForCausalLMPostfix, "output_norm")

    def test_qwen_blocked_postfix_still_adds_router_aux_loss(self):
        self._exercise(Qwen35ForCausalLMPostfix, "norm")


class QwenWrappedLayerTests(unittest.TestCase):
    def test_uses_layer_attention_type_mask_when_present(self):
        class RecordingLayer(nn.Module):
            attention_type = "sliding_attention"

            def __init__(self):
                super().__init__()
                self.seen_attention_mask = None

            def forward(self, hidden, **kwargs):
                self.seen_attention_mask = kwargs["attention_mask"]
                return hidden + 1

        layer = RecordingLayer()
        wrapper = Qwen35ForCausalLMWrappedLayer(layer, idx=0)
        hidden = torch.zeros(1, 2, 3)
        sliding_mask = torch.ones(1, 1, 2, 2)
        full_mask = torch.zeros(1, 1, 2, 2)

        output = wrapper(
            (
                hidden,
                {"full_attention": full_mask, "sliding_attention": sliding_mask},
                torch.arange(2).unsqueeze(0),
                (torch.zeros(1), torch.zeros(1)),
                {},
                None,
                0,
            )
        )

        self.assertIs(layer.seen_attention_mask, sliding_mask)
        self.assertTrue(torch.equal(output[0], hidden + 1))

    def test_side_channel_router_logits_are_appended_without_tuple_return(self):
        class FakeMoe(nn.Module):
            def forward(self, hidden):
                self._last_router_logits = hidden.reshape(-1, hidden.shape[-1])[:, :2]
                return hidden + 1

        class DecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.feed_forward = FakeMoe()
                self.seen_kwargs = None

            def forward(self, hidden, **kwargs):
                self.seen_kwargs = kwargs
                return self.feed_forward(hidden)

        layer = DecoderLayer()
        wrapper = LFM25ForCausalLMWrappedLayer(layer, idx=0)
        hidden = torch.randn(1, 3, 4)
        kwargs = {"_router_logits": []}

        output = wrapper(
            (
                hidden,
                {"full_attention": None},
                torch.arange(3).unsqueeze(0),
                (torch.zeros(1), torch.zeros(1)),
                kwargs,
                None,
                0,
            )
        )

        self.assertTrue(torch.equal(output[0], hidden + 1))
        self.assertNotIn("_router_logits", layer.seen_kwargs)
        self.assertEqual(len(kwargs["_router_logits"]), 1)
        self.assertTrue(torch.equal(kwargs["_router_logits"][0], hidden.reshape(-1, 4)[:, :2]))


if __name__ == "__main__":
    unittest.main()
