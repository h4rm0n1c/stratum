from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn

from stratum.model.registry import ModelArch
from stratum.upload import NF4_ATTR


class _TinyBackbone(nn.Module):
    def __init__(self, *, device: str):
        super().__init__()
        self.embed = nn.Linear(2, 2, bias=False, device=device, dtype=torch.float16)
        self.layers = nn.ModuleList(
            [nn.Linear(2, 2, bias=False, device=device, dtype=torch.float16)]
        )
        self.head = nn.Linear(2, 2, bias=False, device=device, dtype=torch.float16)


class _TinyHFModel(nn.Module):
    def __init__(self, *, device: str = "meta"):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.model = _TinyBackbone(device=device)
        for param in self.parameters():
            param.requires_grad_(False)


class _TinyArch(ModelArch):
    def build_prefix(self, model: nn.Module, **kwargs) -> nn.Module:
        return model.model.embed

    def build_wrapped_layer(self, layer: nn.Module, idx: int, **kwargs) -> nn.Module:
        return layer

    def build_postfix(self, model: nn.Module, **kwargs) -> nn.Module:
        return model.model.head

    def get_num_layers(self, config) -> int:
        return config.num_hidden_layers


def _mark_nf4_staged(module: nn.Module, **kwargs):
    for param in module.parameters():
        setattr(param, NF4_ATTR, object())


class RegistryStagedLoadTest(TestCase):
    def test_meta_nf4_staged_load_uses_explicit_hf_source(self):
        from stratum.upload import NF4Stats

        model = _TinyHFModel(device="meta")

        with patch(
            "stratum.upload.stream_load_and_quantize_module",
            return_value=NF4Stats(),
        ) as stream_fn:
            _TinyArch().build(
                model,
                device_ids=[0],
                use_nf4=True,
                hf_model_name_or_path="example/model",
                verbose=False,
            )

        self.assertEqual(stream_fn.call_count, 3)
        self.assertEqual(
            [call.args[1] for call in stream_fn.call_args_list],
            ["example/model", "example/model", "example/model"],
        )

    def test_meta_nf4_staged_load_requires_hf_source(self):
        from stratum.upload import NF4Stats

        model = _TinyHFModel(device="meta")

        with (
            patch("stratum.upload.stream_load_and_quantize_module", return_value=NF4Stats()),
            self.assertRaisesRegex(RuntimeError, "hf_model_name_or_path"),
        ):
            _TinyArch().build(
                model,
                device_ids=[0],
                use_nf4=True,
                verbose=False,
            )

    def test_meta_nf4_cache_path_passes_cache_dir_and_min_numel(self):
        from stratum.upload import NF4Stats

        model = _TinyHFModel(device="meta")

        with patch(
            "stratum.upload.stream_load_and_quantize_module",
            return_value=NF4Stats(),
        ) as stream_fn:
            _TinyArch().build(
                model,
                device_ids=[0],
                use_nf4=True,
                nf4_cache_dir="/tmp/stratum-nf4-cache",
                nf4_min_numel=123,
                hf_model_name_or_path="example/model",
                verbose=False,
            )

        self.assertEqual(stream_fn.call_count, 3)
        self.assertTrue(
            all(call.kwargs["min_numel"] == 123 for call in stream_fn.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["cache_dir"] == "/tmp/stratum-nf4-cache" for call in stream_fn.call_args_list)
        )
