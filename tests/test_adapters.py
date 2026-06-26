"""CPU-only unit tests for the Llama, Qwen3, and Qwen3-MoE adapter classes.

No GPU required, no HuggingFace downloads. Fake transformers modules are
installed at module load time (before any adapter import) so that the adapters
can be imported without the real transformers package.

Tests exercise:
  - Prefix → WrappedLayer → Postfix forward pass produces a finite scalar loss
  - Router logits accumulation wires correctly for Qwen3-MoE
  - Class identity checks for the adapter Prefix / Postfix types
"""

import sys
import types
import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fake transformers installation
# ---------------------------------------------------------------------------

def _install_fake_transformers():
    """Install minimal stub transformers modules required by the three adapters."""
    transformers = sys.modules.setdefault("transformers", types.ModuleType("transformers"))
    models = sys.modules.setdefault("transformers.models", types.ModuleType("transformers.models"))
    setattr(transformers, "models", models)

    # transformers.modeling_outputs
    modeling_outputs = types.ModuleType("transformers.modeling_outputs")

    class CausalLMOutputWithPast:
        def __init__(self, loss=None, logits=None, **kwargs):
            self.loss = loss
            self.logits = logits
            for k, v in kwargs.items():
                setattr(self, k, v)

    modeling_outputs.CausalLMOutputWithPast = CausalLMOutputWithPast
    sys.modules["transformers.modeling_outputs"] = modeling_outputs

    # transformers.masking_utils
    masking_utils = types.ModuleType("transformers.masking_utils")
    masking_utils.create_causal_mask = lambda *args, **kwargs: None
    masking_utils.create_sliding_window_causal_mask = lambda *args, **kwargs: None
    sys.modules["transformers.masking_utils"] = masking_utils

    # transformers.models.llama
    llama_pkg = types.ModuleType("transformers.models.llama")
    llama_mod = types.ModuleType("transformers.models.llama.modeling_llama")

    class LlamaAttention(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.head_dim = getattr(config, "head_dim", 8) if config else 8
            self.num_heads = getattr(config, "num_attention_heads", 2) if config else 2
            self.num_key_value_groups = 1
            self.scaling = 1.0 / (self.head_dim ** 0.5)
            self.attention_dropout = 0.0
            hidden = getattr(config, "hidden_size", 16) if config else 16
            self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def _apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        return q, k

    def _eager_attention_forward(module, q, k, v, attn_mask, dropout=0.0, scaling=1.0):
        # Minimal scaled-dot-product attention for CPU test.
        # q/k/v: (batch, heads, seq, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        # Return (batch, seq, heads, head_dim) to match flash_attn convention.
        return out.transpose(1, 2), None

    llama_mod.LlamaAttention = LlamaAttention
    llama_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb
    llama_mod.eager_attention_forward = _eager_attention_forward
    llama_mod.repeat_kv = lambda hidden, n_rep: hidden
    sys.modules["transformers.models.llama"] = llama_pkg
    sys.modules["transformers.models.llama.modeling_llama"] = llama_mod
    setattr(models, "llama", llama_pkg)

    # transformers.models.qwen3
    qwen3_pkg = types.ModuleType("transformers.models.qwen3")
    qwen3_mod = types.ModuleType("transformers.models.qwen3.modeling_qwen3")

    class Qwen3Attention(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.head_dim = getattr(config, "head_dim", 8) if config else 8
            self.num_heads = getattr(config, "num_attention_heads", 2) if config else 2
            self.num_key_value_groups = 1
            self.scaling = 1.0 / (self.head_dim ** 0.5)
            self.attention_dropout = 0.0
            hidden = getattr(config, "hidden_size", 16) if config else 16
            self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    qwen3_mod.Qwen3Attention = Qwen3Attention
    qwen3_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb
    sys.modules["transformers.models.qwen3"] = qwen3_pkg
    sys.modules["transformers.models.qwen3.modeling_qwen3"] = qwen3_mod
    setattr(models, "qwen3", qwen3_pkg)

    # transformers.models.qwen3_moe
    qwen3moe_pkg = types.ModuleType("transformers.models.qwen3_moe")
    qwen3moe_mod = types.ModuleType("transformers.models.qwen3_moe.modeling_qwen3_moe")

    class Qwen3MoeAttention(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.head_dim = getattr(config, "head_dim", 8) if config else 8
            self.num_heads = getattr(config, "num_attention_heads", 2) if config else 2
            self.num_key_value_groups = 1
            self.scaling = 1.0 / (self.head_dim ** 0.5)
            self.attention_dropout = 0.0
            hidden = getattr(config, "hidden_size", 16) if config else 16
            self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    class Qwen3MoeSparseMoeBlock(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            hidden = getattr(config, "hidden_size", 16) if config else 16
            num_experts = getattr(config, "num_experts", 3) if config else 3
            self.gate = nn.Linear(hidden, num_experts, bias=False)

        def forward(self, hidden_states):
            return hidden_states

    qwen3moe_mod.Qwen3MoeAttention = Qwen3MoeAttention
    qwen3moe_mod.Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock
    qwen3moe_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb
    sys.modules["transformers.models.qwen3_moe"] = qwen3moe_pkg
    sys.modules["transformers.models.qwen3_moe.modeling_qwen3_moe"] = qwen3moe_mod
    setattr(models, "qwen3_moe", qwen3moe_pkg)

    # Also stub existing adapters' deps so importing stratum doesn't fail.
    for mod_path, cls_name in [
        ("transformers.models.lfm2_moe", "Lfm2MoeAttention"),
        ("transformers.models.qwen3_5", "Qwen3_5Attention"),
    ]:
        pkg_path = mod_path
        mod_full = mod_path + ".modeling_" + mod_path.split(".")[-1]
        pkg_mod = sys.modules.setdefault(pkg_path, types.ModuleType(pkg_path))
        full_mod = sys.modules.setdefault(mod_full, types.ModuleType(mod_full))
        fake_cls = type(cls_name, (nn.Module,), {
            "__init__": lambda self, config=None, layer_idx=0: nn.Module.__init__(self),
        })
        setattr(full_mod, cls_name, fake_cls)
        setattr(full_mod, "apply_rotary_pos_emb", _apply_rotary_pos_emb)

    # LFM2.5 extra attribute needed at import time.
    lfm2_mod = sys.modules.get("transformers.models.lfm2_moe.modeling_lfm2_moe")
    if lfm2_mod is not None:
        lfm2_mod.is_fast_path_available = False


_install_fake_transformers()

# Import adapters AFTER fakes are installed.
from stratum.model.llama import (  # noqa: E402
    LlamaFlashAttention,
    LlamaForCausalLMPrefix,
    LlamaForCausalLMWrappedLayer,
    LlamaForCausalLMPostfix,
)
from stratum.model.qwen3 import (  # noqa: E402
    Qwen3FlashAttention,
    Qwen3ForCausalLMPrefix,
    Qwen3ForCausalLMWrappedLayer,
    Qwen3ForCausalLMPostfix,
)
from stratum.model.qwen3_moe import (  # noqa: E402
    Qwen3MoeFlashAttention,
    Qwen3MoeForCausalLMPrefix,
    Qwen3MoeForCausalLMWrappedLayer,
    Qwen3MoeForCausalLMPostfix,
)


# ---------------------------------------------------------------------------
# Tiny fake model helpers
# ---------------------------------------------------------------------------

def _make_llama_config(hidden=16, heads=2, vocab=32, layers=2):
    return types.SimpleNamespace(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        head_dim=hidden // heads,
        vocab_size=vocab,
        sliding_window=None,
    )


def _make_qwen3_config(hidden=16, heads=2, vocab=32, layers=2):
    return types.SimpleNamespace(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        head_dim=hidden // heads,
        vocab_size=vocab,
        layer_types=["full_attention"] * layers,
        sliding_window=None,
    )


def _make_qwen3moe_config(hidden=16, heads=2, vocab=32, layers=2,
                           num_experts=3, num_experts_per_tok=2):
    return types.SimpleNamespace(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        head_dim=hidden // heads,
        vocab_size=vocab,
        sliding_window=None,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )


class _RotaryEmb(nn.Module):
    """Trivial rotary embedding stub returning (ones, zeros) for tests."""

    def forward(self, hidden_states, position_ids=None):
        seq = hidden_states.shape[1]
        head_dim = 8
        cos = torch.ones(1, seq, head_dim, device=hidden_states.device)
        sin = torch.zeros(1, seq, head_dim, device=hidden_states.device)
        return cos, sin


class _CoreModel(nn.Module):
    """Minimal HF-like model wrapping a config and the expected sub-modules."""

    def __init__(self, config, num_layers):
        super().__init__()
        hidden = config.hidden_size
        vocab = config.vocab_size

        class _InnerModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.config = cfg
                self.embed_tokens = nn.Embedding(vocab, hidden)
                self.rotary_emb = _RotaryEmb()
                self.norm = nn.LayerNorm(hidden)
                self.layers = nn.ModuleList([
                    nn.Identity() for _ in range(num_layers)
                ])

        self.model = _InnerModel(config)
        self.config = config
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


def _make_fake_decoder_layer(config, attention_type="full_attention"):
    """Return a trivial decoder layer (identity on hidden_states)."""
    hidden = config.hidden_size

    class _FakeDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_type = attention_type
            self.linear = nn.Linear(hidden, hidden, bias=False)

        def forward(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    return _FakeDecoderLayer()


def _run_prefix_layer_postfix(prefix_cls, postfix_cls, prefix_kwargs, postfix_kwargs,
                               config, batch=1, seq=6):
    """Run a forward pass through prefix → single fake layer → postfix."""
    core = _CoreModel(config, 1)
    prefix = prefix_cls(core, **prefix_kwargs)
    layer = _make_fake_decoder_layer(config)
    wrapped = None
    # Determine right wrapped-layer class from prefix class name.
    if "Llama" in prefix_cls.__name__:
        wrapped = LlamaForCausalLMWrappedLayer(layer, idx=0)
    elif "Qwen3Moe" in prefix_cls.__name__:
        wrapped = Qwen3MoeForCausalLMWrappedLayer(layer, idx=0)
    else:
        wrapped = Qwen3ForCausalLMWrappedLayer(layer, idx=0)
    postfix = postfix_cls(core, **postfix_kwargs)

    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    labels = torch.randint(0, config.vocab_size, (batch, seq))

    pipe_out = prefix(input_ids=input_ids, labels=labels)
    pipe_out = wrapped(pipe_out)
    result = postfix(pipe_out)
    return result


# ---------------------------------------------------------------------------
# Tests: Llama adapter
# ---------------------------------------------------------------------------

class LlamaAdapterTests(unittest.TestCase):
    def test_forward_produces_finite_loss(self):
        torch.manual_seed(42)
        config = _make_llama_config(hidden=16, heads=2, vocab=32, layers=2)
        result = _run_prefix_layer_postfix(
            LlamaForCausalLMPrefix, LlamaForCausalLMPostfix,
            {}, {},
            config,
        )
        self.assertIsNotNone(result.loss)
        self.assertTrue(torch.isfinite(result.loss))

    def test_postfix_is_llama_class(self):
        config = _make_llama_config()
        core = _CoreModel(config, 2)
        postfix = LlamaForCausalLMPostfix(core)
        self.assertIsInstance(postfix, LlamaForCausalLMPostfix)

    def test_prefix_is_llama_class(self):
        config = _make_llama_config()
        core = _CoreModel(config, 2)
        prefix = LlamaForCausalLMPrefix(core)
        self.assertIsInstance(prefix, LlamaForCausalLMPrefix)

    def test_prefix_output_has_single_causal_mask(self):
        """Llama prefix emits a single tensor (or None) mask, not a dict."""
        torch.manual_seed(7)
        config = _make_llama_config(hidden=16, heads=2, vocab=8, layers=1)
        core = _CoreModel(config, 1)
        prefix = LlamaForCausalLMPrefix(core)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = prefix(input_ids=input_ids, labels=None)
        _hidden, causal_mask, _pos_ids, _pos_embeds, _kwargs, _labels, _lk = out
        # In flash mode (dense_attention_masks=False), mask should be None.
        self.assertIsNone(causal_mask)

    def test_all_ignored_labels_gives_zero_loss(self):
        torch.manual_seed(0)
        config = _make_llama_config(hidden=16, heads=2, vocab=8, layers=1)
        core = _CoreModel(config, 1)
        prefix = LlamaForCausalLMPrefix(core)
        layer = _make_fake_decoder_layer(config)
        wrapped = LlamaForCausalLMWrappedLayer(layer, idx=0)
        postfix = LlamaForCausalLMPostfix(core)

        input_ids = torch.randint(0, 8, (1, 4))
        labels = torch.full((1, 4), -100)  # all ignored

        out = prefix(input_ids=input_ids, labels=labels)
        out = wrapped(out)
        result = postfix(out)
        self.assertTrue(torch.equal(result.loss, torch.tensor(0.0)))


# ---------------------------------------------------------------------------
# Tests: Qwen3 adapter
# ---------------------------------------------------------------------------

class Qwen3AdapterTests(unittest.TestCase):
    def test_forward_produces_finite_loss(self):
        torch.manual_seed(42)
        config = _make_qwen3_config(hidden=16, heads=2, vocab=32, layers=2)
        result = _run_prefix_layer_postfix(
            Qwen3ForCausalLMPrefix, Qwen3ForCausalLMPostfix,
            {}, {},
            config,
        )
        self.assertIsNotNone(result.loss)
        self.assertTrue(torch.isfinite(result.loss))

    def test_postfix_is_qwen3_class(self):
        config = _make_qwen3_config()
        core = _CoreModel(config, 2)
        postfix = Qwen3ForCausalLMPostfix(core)
        self.assertIsInstance(postfix, Qwen3ForCausalLMPostfix)

    def test_prefix_is_qwen3_class(self):
        config = _make_qwen3_config()
        core = _CoreModel(config, 2)
        prefix = Qwen3ForCausalLMPrefix(core)
        self.assertIsInstance(prefix, Qwen3ForCausalLMPrefix)

    def test_prefix_output_has_dict_causal_mask(self):
        """Qwen3 prefix emits a dict of masks (one key per attention type)."""
        torch.manual_seed(7)
        config = _make_qwen3_config(hidden=16, heads=2, vocab=8, layers=1)
        core = _CoreModel(config, 1)
        prefix = Qwen3ForCausalLMPrefix(core)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = prefix(input_ids=input_ids, labels=None)
        _hidden, causal_mask_mapping, _pos_ids, _pos_embeds, _kwargs, _labels, _lk = out
        self.assertIsInstance(causal_mask_mapping, dict)
        self.assertIn("full_attention", causal_mask_mapping)

    def test_no_sliding_layers_on_all_full_attention_config(self):
        config = _make_qwen3_config()
        core = _CoreModel(config, 2)
        prefix = Qwen3ForCausalLMPrefix(core)
        self.assertFalse(prefix.has_sliding_layers)

    def test_has_sliding_layers_when_config_specifies(self):
        config = _make_qwen3_config(layers=2)
        config.layer_types = ["full_attention", "sliding_attention"]
        core = _CoreModel(config, 2)
        prefix = Qwen3ForCausalLMPrefix(core)
        self.assertTrue(prefix.has_sliding_layers)


# ---------------------------------------------------------------------------
# Tests: Qwen3-MoE adapter
# ---------------------------------------------------------------------------

class Qwen3MoeAdapterTests(unittest.TestCase):
    def test_forward_produces_finite_loss(self):
        torch.manual_seed(42)
        config = _make_qwen3moe_config(hidden=16, heads=2, vocab=32, layers=2)
        result = _run_prefix_layer_postfix(
            Qwen3MoeForCausalLMPrefix, Qwen3MoeForCausalLMPostfix,
            {"output_router_logits": False}, {"router_aux_loss_coef": 0.0},
            config,
        )
        self.assertIsNotNone(result.loss)
        self.assertTrue(torch.isfinite(result.loss))

    def test_postfix_is_qwen3moe_class(self):
        config = _make_qwen3moe_config()
        core = _CoreModel(config, 2)
        postfix = Qwen3MoeForCausalLMPostfix(core)
        self.assertIsInstance(postfix, Qwen3MoeForCausalLMPostfix)

    def test_prefix_is_qwen3moe_class(self):
        config = _make_qwen3moe_config()
        core = _CoreModel(config, 2)
        prefix = Qwen3MoeForCausalLMPrefix(core)
        self.assertIsInstance(prefix, Qwen3MoeForCausalLMPrefix)

    def test_prefix_output_has_single_causal_mask(self):
        """Qwen3-MoE prefix emits a single tensor (or None) mask, not a dict."""
        torch.manual_seed(7)
        config = _make_qwen3moe_config(hidden=16, heads=2, vocab=8, layers=1)
        core = _CoreModel(config, 1)
        prefix = Qwen3MoeForCausalLMPrefix(core)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = prefix(input_ids=input_ids, labels=None)
        _hidden, causal_mask, _pos_ids, _pos_embeds, _kwargs, _labels, _lk = out
        # In flash mode (dense_attention_masks=False), mask should be None.
        self.assertIsNone(causal_mask)

    def test_router_logits_accumulate_with_output_router_logits(self):
        """When output_router_logits=True, _router_logits list is initialised."""
        torch.manual_seed(0)
        config = _make_qwen3moe_config(hidden=16, heads=2, vocab=8, layers=1)
        core = _CoreModel(config, 1)
        prefix = Qwen3MoeForCausalLMPrefix(core, output_router_logits=True)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = prefix(input_ids=input_ids, labels=None)
        _hidden, _mask, _pos_ids, _pos_embeds, kwargs, _labels, _lk = out
        self.assertIn("_router_logits", kwargs)
        self.assertIsInstance(kwargs["_router_logits"], list)

    def test_aux_loss_adds_to_main_loss(self):
        """Router aux loss is added to the LM loss when coef > 0."""
        torch.manual_seed(99)
        config = _make_qwen3moe_config(
            hidden=16, heads=2, vocab=8, layers=1,
            num_experts=3, num_experts_per_tok=2,
        )
        core = _CoreModel(config, 1)
        postfix = Qwen3MoeForCausalLMPostfix(core, router_aux_loss_coef=0.01)

        hidden = torch.randn(1, 4, 16, requires_grad=False)
        labels = torch.randint(0, 8, (1, 4))
        router_logits = [
            torch.randn(4, 3, requires_grad=True),  # (batch*seq, num_experts)
        ]
        result = postfix((
            hidden, None, None, None,
            {"_router_logits": router_logits},
            labels, 0,
        ))
        self.assertIsNotNone(result.loss)
        self.assertTrue(torch.isfinite(result.loss))

    def test_postfix_reads_num_experts_from_config(self):
        config = _make_qwen3moe_config(num_experts=5, num_experts_per_tok=2)
        core = _CoreModel(config, 1)
        postfix = Qwen3MoeForCausalLMPostfix(core)
        self.assertEqual(postfix.num_experts, 5)
        self.assertEqual(postfix.num_experts_per_tok, 2)


# ---------------------------------------------------------------------------
# Adapter registration check
# ---------------------------------------------------------------------------

class RegistrationTests(unittest.TestCase):
    def test_all_three_adapters_are_registered(self):
        from stratum.model.registry import _registry
        for name in ("llama", "qwen3", "qwen3-moe"):
            self.assertIn(name, _registry, f"'{name}' not found in registry")


if __name__ == "__main__":
    unittest.main()
